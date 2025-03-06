import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import List

import datasets
import pandas as pd
import torch
import typer
from datasets import load_dataset
from pydantic import BaseModel
from typing_extensions import Annotated

from chrisbase.data import AppTyper, JobTimer, NewProjectEnv
from chrisbase.io import LoggingFormat, LoggerWriter, new_path, log_table
from chrisbase.time import from_timestamp, now_stamp
from chrisdata.ner import GenNERSampleWrapper
from gner import extract_predictions3, parser
from progiter import ProgIter
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Global settings
logger: logging.Logger = logging.getLogger("gner")


def predict(
        device: Annotated[str, typer.Option("--device")] = ...,
        input_file: Annotated[str, typer.Option("--input_file")] = ...,  # "data/ZSE-validation.jsonl", "data/ZSE-test.jsonl"
        pretrained: Annotated[str, typer.Option("--pretrained")] = "dyyyyyyyy/GNER-T5-base",  # "dyyyyyyyy/GNER-T5-large", "output-lfs/ZSE-jihee-BL-dl012/FlanT5-Base-BL/checkpoint-9900", "output-lfs/ZSE-yuyang-BL-lirs-b1/checkpoint-9900"
        max_generation_tokens: Annotated[int, typer.Option("--max_generation_tokens")] = 640,
        generation_by_sample: Annotated[bool, typer.Option("--generation_by_sample/--generation_by_beam")] = ...,
        generation_num_return: Annotated[int, typer.Option("--num_generation")] = ...,
        generation_temperature: Annotated[float, typer.Option("--temperature")] = None,
        generation_top_p: Annotated[float, typer.Option("--top_p")] = None,
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "ZSE-predict",
        output_file: Annotated[str, typer.Option("--output_file")] = "pred.jsonl",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "predict-loggings.out",
        verbose: Annotated[int, typer.Option("--verbose")] = 1,
):
    assert (generation_by_sample and generation_temperature is not None and generation_top_p is not None) or (not generation_by_sample), f"Invalid generation parameters: by_sample={generation_by_sample}, temperature={generation_temperature}, top_p={generation_top_p}"
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        output_name=output_name,
        output_home=output_home,
        output_file=new_path(
            output_file,
            pre=Path(input_file).stem,
            post=f'by_sample-num={generation_num_return}-temp={generation_temperature}-top_p={generation_top_p}'
            if generation_by_sample else f'by_beam-num={generation_num_return}'
        ),
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging.INFO,
        logging_format=LoggingFormat.CHECK_24,
    )
    env.setup_logger(env.logging_level)

    with JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=verbose):
        input_dataset = load_dataset("json", data_files=str(input_file), split="train")
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16).to(device)
        model.eval()

        logger.info(f"Saving predictions to {(env.output_dir / env.output_file)} for {len(input_dataset)} samples")
        with (
            ProgIter(input_dataset, desc=f"Predicting:", stream=LoggerWriter(logger), verbose=2) as progress,
            (env.output_dir / env.output_file).open("w") as out,
            torch.no_grad(),
        ):
            num_example_outputs = 0
            num_prediction_outputs = 0
            for example in progress:
                example = GenNERSampleWrapper.model_validate(example)
                example.id = example.id or example.instance.id
                model_input = tokenizer(example.instance.instruction_inputs, return_tensors="pt").to(device)
                model_outputs = model.generate(
                    **model_input,
                    max_new_tokens=max_generation_tokens,
                    num_return_sequences=generation_num_return,
                    do_sample=generation_by_sample,
                    num_beams=1 if generation_by_sample else generation_num_return,
                    temperature=generation_temperature if generation_by_sample else None,
                    top_p=generation_top_p if generation_by_sample else None,
                )
                example.instance.prediction_outputs = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in model_outputs]
                num_prediction_outputs += len(example.instance.prediction_outputs)
                num_example_outputs += 1
                progress.set_extra(f"| #example={num_example_outputs}, #prediction={num_prediction_outputs}")
                out.write(example.model_dump_json() + "\n")
        logger.info(f"Saved predictions to {(env.output_dir / env.output_file)}: #example={num_example_outputs}, #prediction={num_prediction_outputs}")


class F1_Metric(BaseModel):
    n_correct: int = 0
    n_pos_gold: int = 0
    n_pos_pred: int = 0

    def __str__(self):
        return f"F1={self.f1:.4f}, Prec={self.prec:.4f}, Rec={self.rec:.4f}, #correct={self.n_correct}, #pos_gold={self.n_pos_gold}, #pos_pred={self.n_pos_pred}"

    def __add__(self, other: "F1_Metric") -> "F1_Metric":
        if not isinstance(other, F1_Metric):
            return NotImplemented
        return F1_Metric(
            n_correct=self.n_correct + other.n_correct,
            n_pos_gold=self.n_pos_gold + other.n_pos_gold,
            n_pos_pred=self.n_pos_pred + other.n_pos_pred,
        )

    @property
    def prec(self):
        return self.n_correct / (self.n_pos_pred + 1e-10)

    @property
    def rec(self):
        return self.n_correct / (self.n_pos_gold + 1e-10)

    @property
    def f1(self):
        return 2 * self.prec * self.rec / (self.prec + self.rec + 1e-10)


def find_increasing_indices(lst):
    if not lst:
        return []

    indices = []
    prev_value = lst[0]

    for i in range(1, len(lst)):
        if lst[i] > prev_value:
            indices.append(i)
            prev_value = lst[i]  # Update previous value to the new threshold

    return indices


def evaluate(
        predction_file: Annotated[str, typer.Option("--predction_file")] = ...,  # "output/ZSE-predict/ZSE-test-pred-by_beam-num=100.jsonl", "output/ZSE-predict/ZSE-validation-pred-by_beam-num=100.jsonl"
        max_examples: Annotated[int, typer.Option("--max_examples")] = -1,
        pretrained: Annotated[str, typer.Option("--pretrained")] = "dyyyyyyyy/GNER-T5-base",  # "dyyyyyyyy/GNER-T5-large", "output-lfs/ZSE-jihee-BL-dl012/FlanT5-Base-BL/checkpoint-9900", "output-lfs/ZSE-yuyang-BL-lirs-b1/checkpoint-9900"
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "ZSE-predict",
        output_file: Annotated[str, typer.Option("--output_file")] = "eval.csv",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "evaluate-loggings.out",
        verbose: Annotated[int, typer.Option("--verbose")] = 1,
):
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        output_name=output_name,
        output_home=output_home,
        output_file=new_path(
            output_file,
            pre=Path(predction_file).stem,
        ),
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging.INFO,
        logging_format=LoggingFormat.CHECK_24,
    )
    env.setup_logger(env.logging_level)

    with (JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=verbose)):
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        input_dataset = load_dataset("json", data_files=str(predction_file), split=datasets.Split.TRAIN)
        all_examples = defaultdict(list)
        for example in input_dataset:
            example = GenNERSampleWrapper.model_validate(example)
            if max_examples <= 0 or len(all_examples[example.dataset]) < max_examples:
                all_examples[example.dataset].append(example)
        max_candidates = max(
            len(example.instance.prediction_outputs)
            for dataset in all_examples
            for example in all_examples[dataset]
        )
        num_dataset = len(all_examples)

        logger.info(f"Evaluating predictions from {predction_file}"
                    f" for {len(all_examples)} dataset ({sum(len(all_examples[d]) for d in all_examples)} samples)")
        all_results = {}
        for di, dataset in enumerate(sorted(all_examples.keys()), start=1):
            dataset_metrics = [F1_Metric()] * max_candidates
            for example in ProgIter(all_examples[dataset], stream=LoggerWriter(logger), verbose=2, time_thresh=5,
                                    desc=f" - [{di:02d}/{num_dataset:02d}] {dataset:<20s}:"):
                example_metrics: List[F1_Metric] = []
                for prediction_output in example.instance.prediction_outputs:
                    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
                    words = example.instance.words
                    labels = example.instance.labels
                    predictions = extract_predictions3(prediction_output, example=example, tokenizer=tokenizer)
                    gold_tuples = parser(words, labels)
                    pred_tuples = parser(words, predictions)
                    for t in pred_tuples:
                        if t in gold_tuples:
                            n_correct += 1
                        n_pos_pred += 1
                    n_pos_gold += len(gold_tuples)
                    curr_metric = F1_Metric(
                        n_correct=n_correct,
                        n_pos_gold=n_pos_gold,
                        n_pos_pred=n_pos_pred,
                    )
                    if not example_metrics:
                        example_metric = curr_metric
                    else:
                        prev_metric = example_metrics[-1]
                        example_metric = curr_metric if curr_metric.f1 > prev_metric.f1 else prev_metric
                    example_metrics.append(example_metric)

                # inc_points = find_increasing_indices([x.f1 for x in example_metrics])
                # logger.info(f"{len(example_metrics)}: {len(inc_points)}: {inc_points}: {[f'{x.f1:.4f}' for x in example_metrics]}")
                for c, example_metric in enumerate(example_metrics):
                    dataset_metrics[c] += example_metric
            all_results[re.split("[-_]", dataset)[-1]] = dataset_metrics

        all_results["average"] = [F1_Metric()] * max_candidates
        for c in range(max_candidates):
            for d in all_results:
                all_results["average"][c] += all_results[d][c]

        dataset_names = list(all_results.keys())
        candidate_numbers = list(range(1, max_candidates + 1))
        f1_data = {dataset: [metric.f1 for metric in all_results[dataset]] for dataset in dataset_names}
        df_results = pd.DataFrame(f1_data, index=candidate_numbers)
        df_results.index.name = "candidate"
        df_results.to_csv(env.output_dir / env.output_file)
        log_table(logger, df_results.map(lambda x: f"{x:.4f}"), tablefmt="orgtbl", border_idx=1)
        logger.info(f"F1 performance results saved to {env.output_dir / env.output_file}")


if __name__ == "__main__":
    AppTyper.run(
        predict,
        evaluate,
    )
