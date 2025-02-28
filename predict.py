import logging
from pathlib import Path
from typing import List

import datasets
import torch
import typer
from datasets import load_dataset
from pydantic import BaseModel
from typing_extensions import Annotated

from chrisbase.data import AppTyper, JobTimer, NewProjectEnv
from chrisbase.io import LoggingFormat, LoggerWriter, new_path
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


class PerformanceMetrics(BaseModel):
    f1: float = 0.0
    prec: float = 0.0
    rec: float = 0.0
    n_correct: int = 0
    n_pos_gold: int = 0
    n_pos_pred: int = 0

    def __str__(self):
        return f"F1={self.f1:.4f}, Prec={self.prec:.4f}, Rec={self.rec:.4f}, #correct={self.n_correct}, #pos_gold={self.n_pos_gold}, #pos_pred={self.n_pos_pred}"


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
        predction_file: Annotated[str, typer.Option("--predction_file")] = "output/ZSE-predict/ZSE-test-pred-by_beam-num=100.jsonl",  # "output/ZSE-predict/ZSE-test-pred-by_beam-num=100.jsonl"
        pretrained: Annotated[str, typer.Option("--pretrained")] = "dyyyyyyyy/GNER-T5-base",  # "dyyyyyyyy/GNER-T5-large", "output-lfs/ZSE-jihee-BL-dl012/FlanT5-Base-BL/checkpoint-9900", "output-lfs/ZSE-yuyang-BL-lirs-b1/checkpoint-9900"
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "ZSE-predict",
        output_file: Annotated[str, typer.Option("--output_file")] = "eval.jsonl",
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

    with JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=verbose):
        input_dataset = load_dataset("json", data_files=str(predction_file), split=datasets.Split.TRAIN)
        # input_dataset = input_dataset.select(range(10))
        tokenizer = AutoTokenizer.from_pretrained(pretrained)

        logger.info(f"Evaluating predictions from {predction_file} for {len(input_dataset)} samples")
        with (
            ProgIter(input_dataset, desc=f"Evaluating:", stream=LoggerWriter(logger), verbose=2) as progress,
            (env.output_dir / env.output_file).open("w") as out,
        ):
            num_example_outputs = 0
            num_prediction_outputs = 0
            for example in progress:
                example = GenNERSampleWrapper.model_validate(example)
                assert example.instance.prediction_outputs and len(example.instance.prediction_outputs) > 0, f"Missing prediction outputs for {example.instance.id}"
                # logger.info(f"Answer: {example.instance.prompt_labels}")
                achievable_perfs: List[PerformanceMetrics] = []
                for prediction_output in example.instance.prediction_outputs:
                    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
                    # logger.info(f"= prediction_output: {prediction_output}")
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
                    prec = n_correct / (n_pos_pred + 1e-10)
                    rec = n_correct / (n_pos_gold + 1e-10)
                    f1 = 2 * prec * rec / (prec + rec + 1e-10)
                    curr_perf = PerformanceMetrics(
                        f1=f1,
                        prec=prec,
                        rec=rec,
                        n_correct=n_correct,
                        n_pos_gold=n_pos_gold,
                        n_pos_pred=n_pos_pred,
                    )
                    if not achievable_perfs:
                        achievable_perf = curr_perf
                    else:
                        prev_perf = achievable_perfs[-1]
                        achievable_perf = curr_perf if curr_perf.f1 > prev_perf.f1 else prev_perf
                    achievable_perfs.append(achievable_perf)

                    # logger.info(f"  - words : {words}")
                    # logger.info(f"  - labels: {labels}")
                    # logger.info(f"  - preds : {predictions}")
                    # logger.info(f"  - gold_tuples: {gold_tuples}")
                    # logger.info(f"  - pred_tuples: {pred_tuples}")
                    # logger.info(f"  - achievable: {achievable_perf}")
                    # logger.info(f"  - currently : {curr_perf}")
                    # logger.info("")

                inc_points = find_increasing_indices([x.f1 for x in achievable_perfs])
                logger.info(f"{len(inc_points)}: {inc_points}: {[f'{x.f1:.4f}' for x in achievable_perfs]}")


if __name__ == "__main__":
    AppTyper.run(
        predict,
        evaluate,
    )
