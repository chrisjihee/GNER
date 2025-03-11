import json
import logging
import re
from collections import defaultdict
from operator import attrgetter
from pathlib import Path
from typing import List, Optional

import datasets
import pandas as pd
import torch
import typer
from datasets import load_dataset
from nltk.metrics.distance import edit_distance
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

from chrisbase.data import AppTyper, JobTimer, NewProjectEnv
from chrisbase.io import LoggingFormat, LoggerWriter, new_path, log_table, files, make_parent_dir, load_json
from chrisbase.time import from_timestamp, now_stamp
from chrisbase.util import grouped
from chrisdata.learn import F1, RegressionSample
from chrisdata.ner import GenNERSampleWrapper, GenNERSample
from gner import NEREvaluator
from progiter import ProgIter
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Global settings
logger: logging.Logger = logging.getLogger("gner")
main = AppTyper()


@main.command("generate_prediction")
def generate_prediction(
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
        logging_file: Annotated[str, typer.Option("--logging_file")] = "generate_prediction.out",
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


def normalized_edit_distance(hyp_text: str, ref_text: str) -> float:
    dist = edit_distance(ref_text, hyp_text)
    max_len = max(len(ref_text), len(hyp_text))
    norm_dist = dist / max_len if max_len else 0.0
    return norm_dist


class PredictionQuality(BaseModel):
    prediction: str
    norm_dist: float
    f1_score: float
    quality: Optional[float] = None

    def calc_quality(self, weight_f1: float = 0.7, weight_nd: float = 0.3, pow_weight: float = 2.0, max_score: float = 5.0):
        qe_score = sum([
            weight_f1 * self.f1_score,
            weight_nd * (1.0 - self.norm_dist),
        ])
        self.quality = round(pow(qe_score, pow_weight) * max_score, 2)
        return self

    def __str__(self):
        return f"Q={self.quality:.2f}, F1={self.f1_score:.4f}, ND={self.norm_dist:.4f}, pred={self.prediction}"


@main.command("convert_to_qe_data")
def convert_to_qe_data(
        generation_file: Annotated[str, typer.Option("--generation_file")] = "output/ZSE-predict/ZSE-test-pred-by_beam-num=100.jsonl",  # "output/ZSE-predict/ZSE-test-pred-by_beam-num=100.jsonl", "output/ZSE-predict/ZSE-validation-pred-by_beam-num=100.jsonl"
        output_file: Annotated[str, typer.Option("--output_file")] = None,
        pretrained: Annotated[str, typer.Option("--pretrained")] = "dyyyyyyyy/GNER-T5-base",  # "dyyyyyyyy/GNER-T5-large", "output-lfs/ZSE-jihee-BL-dl012/FlanT5-Base-BL/checkpoint-9900", "output-lfs/ZSE-yuyang-BL-lirs-b1/checkpoint-9900"
        num_candidates: Annotated[int, typer.Option("--num_candidates")] = 10,
        no_train_split: Annotated[bool, typer.Option("--no_train_split")] = True,
        test_count: Annotated[int, typer.Option("--test_count")] = 200,
        test_ratio: Annotated[float, typer.Option("--test_ratio")] = 1.0,
        test_split: Annotated[str, typer.Option("--test_split")] = "test",
        weight_f1: Annotated[float, typer.Option("--weight_f1")] = 0.7,
        weight_nd: Annotated[float, typer.Option("--weight_nd")] = 0.3,
        pow_weight: Annotated[float, typer.Option("--pow_weight")] = 2.0,
        max_score: Annotated[float, typer.Option("--max_score")] = 5.0,
        output_home: Annotated[str, typer.Option("--output_home")] = "data",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER-QE",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "convert_to_qe_data.out",
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        verbose: Annotated[int, typer.Option("--verbose")] = 1,
):
    if not output_file:
        output_file = Path(generation_file).stem.split("-num=")[0] + ".json"
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        random_seed=random_seed,
        output_name=output_name,
        output_home=output_home,
        output_file=output_file,
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging.INFO,
        logging_format=LoggingFormat.CHECK_24,
    )
    env.setup_logger(env.logging_level)

    with (JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=verbose)):
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        input_dataset = load_dataset("json", data_files=str(generation_file), split=datasets.Split.TRAIN)
        raw_examples = defaultdict(list)
        test_examples = defaultdict(list)
        train_examples = defaultdict(list)
        for example in input_dataset:
            example = GenNERSampleWrapper.model_validate(example)
            example.id = example.id or example.instance.id
            raw_examples[example.dataset].append(example)
        for sub in raw_examples:
            test_size = test_count if test_count > 0 else test_ratio
            if isinstance(test_size, float) and test_size >= 1.0:
                train_set, test_set = [], raw_examples[sub]
            else:
                train_set, test_set = train_test_split(raw_examples[sub], test_size=test_size, random_state=env.random_seed)
                if no_train_split:
                    train_set = []
            train_examples[sub] = sorted(train_set, key=lambda x: int(x.id))
            test_examples[sub] = sorted(test_set, key=lambda x: int(x.id))

        logger.info(f"Converting the generated predictions into QE data"
                    f" for {len(raw_examples)} dataset ({sum(map(len, raw_examples.values()))} samples = {sum(map(len, train_examples.values()))} + {sum(map(len, test_examples.values()))})")

        converted_dataset = defaultdict(list)
        for split, examples in {
            "train": train_examples,
            test_split: test_examples,
        }.items():
            idx = 0
            for i, sub in enumerate(sorted(examples.keys()), start=1):
                for example in ProgIter(examples[sub], stream=LoggerWriter(logger), verbose=2, time_thresh=5,
                                        desc=f" - [{i:02d}/{len(examples):02d}] ({split:<5s}) {sub:<20s}:"):
                    reference = GenNERSample.get_prompt_labels(example.instance.words, example.instance.labels)
                    candiates = sorted(set(example.instance.prediction_outputs[: num_candidates]))
                    # print(f"prompt_labels: {reference}")
                    scored_candidates = [
                        PredictionQuality(
                            prediction=candidate,
                            norm_dist=normalized_edit_distance(candidate, reference),
                            f1_score=NEREvaluator().evaluate_prediction(candidate, example, tokenizer).f1
                        ).calc_quality(weight_f1=weight_f1, weight_nd=weight_nd, pow_weight=pow_weight, max_score=max_score)
                        for candidate in candiates
                    ]
                    # for j, x in enumerate(sorted(scored_candidates, key=attrgetter('quality'), reverse=True), start=1):
                    #     print(f"{j:02d}: {x}")

                    for scored_candidate in scored_candidates:
                        converted_dataset[split].append(
                            RegressionSample(
                                sentence1=scored_candidate.prediction,
                                sentence2=' '.join(example.instance.words),
                                label=scored_candidate.quality,
                                idx=idx,
                            )
                        )
                        idx += 1

        for split in converted_dataset:
            output_file = new_path(env.output_file, post=f"num={num_candidates}-{split}")
            with Path(env.output_dir / output_file).open("w") as out:
                for example in converted_dataset[split]:
                    out.write(example.model_dump_json() + "\n")
            logger.info(f"QE data[{split}] saved to {env.output_dir / output_file}")


@main.command("rerank_by_predict_results")
def rerank_by_predict_results(
        generation_file: Annotated[str, typer.Option("--generation_file")] = "output/ZSE-predict/ZSE-validation-pred-by_beam-num=100.jsonl",  # "output/ZSE-predict/ZSE-test-pred-by_beam-num=100.jsonl", "output/ZSE-predict/ZSE-validation-pred-by_beam-num=100.jsonl"
        regression_input_files: Annotated[str, typer.Option("--regression_input_files")] = "data/GNER-QE/ZSE-validation-pred-by_beam-num=*-val.json",
        regression_output_files: Annotated[str, typer.Option("--regression_output_files")] = "output/GNER-QE/**/checkpoint-*/pred/predict_results_*",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "dyyyyyyyy/GNER-T5-base",  # "dyyyyyyyy/GNER-T5-large", "output-lfs/ZSE-jihee-BL-dl012/FlanT5-Base-BL/checkpoint-9900", "output-lfs/ZSE-yuyang-BL-lirs-b1/checkpoint-9900"
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "ZSE-rerank",
        output_file: Annotated[str, typer.Option("--output_file")] = "rerank.jsonl",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "rerank_by_predict_results.out",
        verbose: Annotated[int, typer.Option("--verbose")] = 1,
):
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        output_name=output_name,
        output_home=output_home,
        output_file=new_path(
            output_file,
            pre=Path(generation_file).stem,
        ),
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging.INFO,
        logging_format=LoggingFormat.CHECK_24,
    )
    env.setup_logger(env.logging_level)

    with (JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=verbose)):
        # tokenizer = AutoTokenizer.from_pretrained(pretrained)
        gner_data = defaultdict(GenNERSampleWrapper)
        with Path(generation_file).open() as generation_data:
            for line in generation_data:
                example = GenNERSampleWrapper.model_validate_json(line)
                example.id = example.instance.id
                # example.instance.prediction_outputs = []
                gner_data[' '.join(example.instance.words)] = example
        logger.info(f"Loaded {len(gner_data)} samples from {generation_file}")

        regression_inputs = {k: list(vs)[0] for k, vs in grouped(files(regression_input_files), key=lambda x: x.stem)}
        for k, regression_input_file in regression_inputs.items():
            regression_input_samples = []
            with Path(regression_input_file).open() as regression_input:
                for line in regression_input:
                    regression_input_samples.append(RegressionSample.model_validate_json(line))
            regression_inputs[k] = regression_input_samples
        logger.info(f"Loaded {sum([len(x) for x in regression_inputs.values()])} regression input samples: {len(regression_inputs)} regression inputs")

        regression_outputs = {k: list(vs) for k, vs in grouped(files(regression_output_files), key=lambda x: x.parent.parent)}
        logger.info(f"Merging and reranking {sum([len(x) for x in regression_outputs.values()])} outputs: {len(regression_outputs)} regression models * {len(regression_inputs)} regression inputs")
        for model in regression_outputs:
            logger.info(f"[model] {model}")

            # Copy the evaluation results
            model_eval_file = model / "pred" / "eval_results.json"
            assert model_eval_file.exists() and model_eval_file.is_file(), f"Missing evaluation results: {model_eval_file}"
            make_parent_dir(env.output_dir / model.parent.name / model.name / "eval_results.json").write_text(json.dumps(load_json(model_eval_file), indent=4))

            for regression_output_file in regression_outputs[model]:
                regression_input_name = regression_output_file.stem.split("predict_results_")[-1]
                regression_input_samples = regression_inputs[regression_input_name]
                regression_output_labels = pd.read_csv(regression_output_file, delimiter="\t")['prediction'].tolist()
                assert len(regression_input_samples) == len(regression_output_labels), f"Length mismatch: input({len(regression_input_samples)}) != output({len(regression_output_labels)})"
                for input_sample, output_label in zip(regression_input_samples, regression_output_labels):
                    input_sample.label = output_label
                reranked_samples = {k: sorted(vs, key=attrgetter('label'), reverse=True) for k, vs in grouped(regression_input_samples, attrgetter='sentence2')}
                reranked_gner_data = []
                for input_sentence, reranked in reranked_samples.items():
                    example = gner_data[input_sentence].model_copy(deep=True)
                    example.instance.prediction_outputs = [x.sentence1 for x in reranked]
                    # reranked = [x.sentence1 for x in reranked]
                    # previous = example.instance.prediction_outputs
                    # for reranked_sample, previous_sample in zip(reranked, previous[:len(reranked)]):
                    #     previous_f1 = NEREvaluator().evaluate_prediction(previous_sample, example, tokenizer)
                    #     reranked_f1 = NEREvaluator().evaluate_prediction(reranked_sample, example, tokenizer)
                    #     logger.info(f"{previous_f1} -> {reranked_f1}")
                    # logger.info("--------------------")
                    reranked_gner_data.append(example)
                reranked_gner_data = sorted(reranked_gner_data, key=lambda x: (x.dataset, int(x.id)))
                num_candidate = max([len(x.instance.prediction_outputs) for x in reranked_gner_data])
                # sum_candidate = sum([len(x.instance.prediction_outputs) for x in reranked_gner_data])
                # logger.info(f"  => {len(reranked_gner_data)}, {num_candidate}, {sum_candidate}")

                reranked_data_file = make_parent_dir(env.output_dir / model.parent.name / model.name / new_path(env.output_file, post=f"cand={num_candidate}"))
                with reranked_data_file.open("w") as out:
                    for example in reranked_gner_data:
                        out.write(example.model_dump_json() + "\n")
                logger.info(f"- {regression_output_file} => {reranked_data_file}")


@main.command("check_possibility")
def check_possibility(
        generation_file: Annotated[str, typer.Argument()] = ...,  # "output/ZSE-predict/ZSE-validation-pred-by_beam-num=100.jsonl",  "output/ZSE-predict/ZSE-test-pred-by_beam-num=100.jsonl"
        pretrained: Annotated[str, typer.Option("--pretrained")] = "dyyyyyyyy/GNER-T5-base",  # "dyyyyyyyy/GNER-T5-large", "output-lfs/ZSE-jihee-BL-dl012/FlanT5-Base-BL/checkpoint-9900", "output-lfs/ZSE-yuyang-BL-lirs-b1/checkpoint-9900"
        max_examples: Annotated[int, typer.Option("--max_examples")] = -1,
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = None,
        output_file: Annotated[str, typer.Option("--output_file")] = "eval.csv",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "check_possibility.out",
        verbose: Annotated[int, typer.Option("--verbose")] = 1,
):
    generation_file = Path(generation_file)
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        output_name=output_name,
        output_home=output_home,
        output_file=new_path(
            output_file,
            pre=generation_file.stem,
        ),
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging.INFO,
        logging_format=LoggingFormat.CHECK_24,
    )
    env.setup_logger(env.logging_level)

    with (JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=verbose)):
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        input_dataset = load_dataset("json", data_files=str(generation_file), split=datasets.Split.TRAIN)
        all_examples = defaultdict(list)
        for example in input_dataset:
            example = GenNERSampleWrapper.model_validate(example)
            example.id = example.id or example.instance.id
            if max_examples <= 0 or len(all_examples[example.dataset]) < max_examples:
                all_examples[example.dataset].append(example)
        max_candidates = max(
            len(example.instance.prediction_outputs)
            for dataset in all_examples
            for example in all_examples[dataset]
        )
        num_dataset = len(all_examples)

        logger.info(f"Checking the possibility of predictions from {generation_file}"
                    f" for {len(all_examples)} dataset ({sum(len(all_examples[d]) for d in all_examples)} samples)")
        evaluation_file = generation_file.parent / "eval_results.json"
        evaluation_results = None
        if evaluation_file.exists() and evaluation_file.is_file():
            logger.info(f"Loading evaluation results from {evaluation_file}")
            evaluation_results = load_json(evaluation_file)
        all_results = {}
        for di, dataset in enumerate(sorted(all_examples.keys()), start=1):
            dataset_metrics = [F1()] * max_candidates
            for example in ProgIter(all_examples[dataset], stream=LoggerWriter(logger), verbose=2, time_thresh=5,
                                    desc=f" - [{di:02d}/{num_dataset:02d}] {dataset:<20s}:"):
                candiates_best: List[F1] = []
                for prediction_output in example.instance.prediction_outputs:
                    example_f1 = NEREvaluator().evaluate_prediction(prediction_output, example, tokenizer)
                    if not candiates_best:
                        best_curr = example_f1
                    else:
                        best_prev = candiates_best[-1]
                        best_curr = example_f1 if example_f1.f1 > best_prev.f1 else best_prev
                    candiates_best.append(best_curr)

                # inc_points = find_increasing_indices([x.f1 for x in candiates_best])
                # logger.info(f"{len(candiates_best)}: {len(inc_points)}: {inc_points}: {[f'{x.f1:.4f}' for x in candiates_best]}")
                for c, example_metric in enumerate(candiates_best):
                    dataset_metrics[c] += example_metric
            all_results[re.split("[-_]", dataset)[-1]] = dataset_metrics

        all_results["average"] = [F1()] * max_candidates
        for c in range(max_candidates):
            for d in all_results:
                all_results["average"][c] += all_results[d][c]

        dataset_names = list(all_results.keys())
        candidate_numbers = list(range(1, max_candidates + 1))
        performance_data = {dataset: [metric.f1 for metric in all_results[dataset]] for dataset in dataset_names}
        performance_df = pd.DataFrame(performance_data, index=candidate_numbers)
        performance_df.index.name = "candidate"
        if evaluation_results:
            for k, v in evaluation_results.items():
                performance_df[k] = v
        performance_df.to_csv(Path(generation_file).parent / env.output_file)
        log_table(logger, performance_df.map(lambda x: f"{x:.4f}"), tablefmt="orgtbl", border_idx=1)
        logger.info(f"F1 performance results saved to {env.output_dir / env.output_file}")


if __name__ == "__main__":
    main()
