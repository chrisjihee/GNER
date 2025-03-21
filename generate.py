import difflib
import json
import logging
import os
import random
import re
from collections import defaultdict, OrderedDict
from operator import attrgetter
from pathlib import Path
from typing import List

import datasets
import pandas as pd
import torch
import typer
from datasets import Dataset, load_dataset
from typing_extensions import Annotated

from chrisbase.data import FileOption, InputOption, JobTimer, FileStreamer, NewProjectEnv, AppTyper
from chrisbase.io import LoggingFormat, new_path, LoggerWriter, log_table, files, make_parent_dir, load_json
from chrisbase.time import from_timestamp, now_stamp
from chrisbase.util import grouped
from chrisdata.learn import F1, Sum, RegressionSample
from chrisdata.ner import GenNERSampleWrapper, GenNERSample
from gner import NEREvaluator
from gner.gner_evaluator import extract_predictions3, PredictionQuality
from progiter import ProgIter
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Global settings
logger: logging.Logger = logging.getLogger("gner")
main = AppTyper()


def compute_diffs(base_sent, alter_sents, ws=1):
    """
    Function to compute the differing spans between the base sentence and the other candidates
    :param base_sent: base sentence
    :param alter_sents: list of candidate sentences
    """
    edits = OrderedDict()
    edit_positions = OrderedDict()
    base_hyp = base_sent.split(" ")
    for candidate in alter_sents:
        alter_hyp = candidate.split(" ")
        # compute the differing spans between the base sentence and the candidate
        s = difflib.SequenceMatcher(None, base_hyp, alter_hyp)
        for op, s_start_ind, s_end_ind, t_start_ind, t_end_ind in s.get_opcodes():
            if op != "equal":
                # if op != "replace":
                s_start_ind -= ws
                t_start_ind -= ws
                s_end_ind += ws
                t_end_ind += ws
                source_span = " ".join(base_hyp[max(s_start_ind, 0): s_end_ind])
                target_span = " ".join(alter_hyp[max(t_start_ind, 0): t_end_ind])
                # add the differing spans to the edits dictionary
                if source_span not in edits:
                    edits[source_span] = [target_span]
                    # save the start index of the initial span to the edit positions dictionary
                    edit_positions[source_span] = [max(s_start_ind - 1, 0)]
                else:
                    if target_span not in edits[source_span]:
                        edits[source_span].append(target_span)
    # sort the edits based on the edit positions
    sorted_positions = OrderedDict(sorted(edit_positions.items(), key=lambda item: item[1]))
    # sorted_edits = OrderedDict((key, edits[key]) for key in sorted_positions)
    sorted_edits = [(key, edits[key]) for key in sorted_positions]
    return sorted_edits


@main.command("generate_hybrid_prediction")
def generate_hybrid_prediction(
        device: Annotated[str, typer.Option("--device")] = ...,
        input_file: Annotated[str, typer.Option("--input_file")] = ...,  # "data/ZSE-validation-sampled-N70.jsonl"
        input_start: Annotated[int, typer.Option("--input_start")] = 0,
        input_limit: Annotated[int, typer.Option("--input_limit")] = -1,
        output_file: Annotated[str, typer.Option("--output_file")] = "data/GNER-QE/hybrid_gen.jsonl",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER-QE",
        output_home: Annotated[str, typer.Option("--output_home")] = "data",
        sr_inst_file: Annotated[str, typer.Option("--sr_inst_file")] = "configs/instruction/GNER-EQ-SR.txt",
        mr_inst_file: Annotated[str, typer.Option("--mr_inst_file")] = "configs/instruction/GNER-EQ-MR.txt",
        sr_generation_amount: Annotated[int, typer.Option("--generation_amount")] = 5,
        mr_generation_amount: Annotated[int, typer.Option("--generation_amount")] = 10,
        # max_example_per_quality: Annotated[int, typer.Option("--max_example_per_quality")] = 20,
        generation_by_sample: Annotated[bool, typer.Option("--generation_by_sample/--generation_by_beam")] = False,
        generation_temp: Annotated[float, typer.Option("--temp")] = 1.5,
        generation_top_p: Annotated[float, typer.Option("--top_p")] = 0.9,
        generation_tokens: Annotated[int, typer.Option("--generation_tokens")] = 640,
        pretrained: Annotated[str, typer.Option("--pretrained")] = "output-lfs/train_ZSE-HR207842/GnerT5-Base-HR207842/checkpoint-17052",
        do_check_possibility: Annotated[bool, typer.Option("--do_check_possibility/--no_check_possibility")] = False,
        weight_f1: Annotated[float, typer.Option("--weight_f1")] = 0.7,
        weight_ed: Annotated[float, typer.Option("--weight_ed")] = 0.3,
        pow_weight: Annotated[float, typer.Option("--pow_weight")] = 2.0,
        max_score: Annotated[float, typer.Option("--max_score")] = 5.0,
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        logging_file: Annotated[str, typer.Option("--logging_file")] = "generate_hybrid_prediction.out",
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    input_file = Path(input_file)
    output_file = Path(output_file)
    sr_inst_file = Path(sr_inst_file)
    mr_inst_file = Path(mr_inst_file)
    assert (generation_by_sample and generation_temp is not None and generation_top_p is not None) or (not generation_by_sample), f"Invalid generation parameters: by_sample={generation_by_sample}, temperature={generation_temp}, top_p={generation_top_p}"
    assert sr_inst_file.is_file() and mr_inst_file.is_file(), f"Invalid instruction files: sr={sr_inst_file}, mr={mr_inst_file}"
    assert input_file.is_file(), f"Invalid input file: {input_file}"
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        output_name=output_name,
        output_home=output_home,
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging_level,
        logging_format=LoggingFormat.CHECK_24,
        random_seed=random_seed,
    )
    input_opt = InputOption(
        start=input_start,
        limit=input_limit,
        file=FileOption.from_path(path=input_file, required=True),
    )
    output_file = (env.output_dir if output_file.parent == Path() else output_file.parent) / new_path(
        output_file.name,
        pre=f"{input_file.stem}{f'-from-{input_start:05d}' if input_start > 0 else ''}{f'-to-{input_start + input_limit:05d}' if input_limit > 0 else ''}",
        post=f"by_sample-amount={sr_generation_amount}x{mr_generation_amount}-temp={generation_temp}-top_p={generation_top_p}"
        if generation_by_sample else f"by_beam-amount={sr_generation_amount}x{mr_generation_amount}"
    )
    sr_inst_temp = sr_inst_file.read_text()
    mr_inst_temp = mr_inst_file.read_text()

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
        FileStreamer(input_opt.file) as input_file,
        FileStreamer(FileOption.from_path(path=output_file, mode="w")) as output_file,
    ):
        input_data = input_opt.ready_inputs(input_file, total=len(input_file))
        with (
            ProgIter(input_data.items, total=input_data.num_item, desc=f"Generating to {output_file.path}:", stream=LoggerWriter(logger, level=logging_level), verbose=3) as progress,
            torch.no_grad(),
        ):
            f1_sum = F1()
            hyps_sum = Sum()
            for line in progress:
                example = GenNERSampleWrapper.model_validate_json(line)
                example.instance.id = example.id = example.instance.id or example.id
                example.label_list = [str(x).replace(" ", "_").upper() for x in example.label_list]
                example.instance.labels = [str(x).replace(" ", "_").upper() for x in example.instance.labels]
                possible_labels = [tag for entity_type in example.label_list for tag in (f"B-{entity_type}", f"I-{entity_type}")] + ["O"]
                if len(example.instance.words) != len(example.instance.labels) or any(label not in possible_labels for label in example.instance.labels):
                    continue
                sentence = " ".join(example.instance.words)
                logger.debug("=" * 80)
                logger.debug(f"[Sentence] {sentence}")

                # Generate by single-round for all entity types
                entity_types = ", ".join(example.label_list)
                possible_labels = [tag for entity_type in example.label_list for tag in (f"B-{entity_type}", f"I-{entity_type}")] + ["O"]
                final_words, final_labels = example.instance.words, [x if x in possible_labels else "O" for x in example.instance.labels]
                prompt_labels = GenNERSample.get_prompt_labels(final_words, final_labels)
                instruction_inputs = sr_inst_temp.format(entity_types=entity_types, sentence=sentence)
                sr_example = GenNERSampleWrapper(
                    id=f"{example.id}.S",
                    dataset=example.dataset,
                    split=example.split,
                    label_list=example.label_list,
                    instance=GenNERSample(
                        id=f"{example.id}.S",
                        group=f"{example.id}",
                        words=final_words,
                        labels=final_labels,
                        target_label="*",
                        prompt_labels=prompt_labels,
                        instruction_inputs=instruction_inputs,
                    )
                )
                model_input = tokenizer(sr_example.instance.instruction_inputs, return_tensors="pt").to(device)
                model_outputs = model.generate(
                    **model_input,
                    max_new_tokens=generation_tokens,
                    num_return_sequences=sr_generation_amount,
                    do_sample=generation_by_sample,
                    num_beams=1 if generation_by_sample else sr_generation_amount,
                    temperature=generation_temp if generation_by_sample else None,
                    top_p=generation_top_p if generation_by_sample else None,
                )

                valid_prediction_labels = list()
                for model_output in model_outputs:
                    decoded_output = tokenizer.decode(model_output, skip_special_tokens=True).strip()
                    prediction_labels = extract_predictions3(decoded_output, example=sr_example, tokenizer=tokenizer)
                    if len(sr_example.instance.words) != len(prediction_labels) or any(label not in possible_labels for label in prediction_labels):
                        continue
                    if prediction_labels not in valid_prediction_labels:
                        valid_prediction_labels.append(prediction_labels)
                sr_example.instance.prediction_outputs = [GenNERSample.get_prompt_labels(example.instance.words, labels) for labels in valid_prediction_labels]
                logger.debug(f"  * Generate {'single-round':20s} : {len(model_outputs):2d} -> {len(sr_example.instance.prediction_outputs):2d}")

                # Generate by multi-round for each entity type
                mr_examples = []
                for i, entity_type in enumerate(example.label_list if mr_inst_temp else [], start=1):
                    possible_labels = [tag for tag in (f"B-{entity_type}", f"I-{entity_type}")] + ["O"]
                    final_words, final_labels = example.instance.words, [x if x in possible_labels else "O" for x in example.instance.labels]
                    prompt_labels = GenNERSample.get_prompt_labels(final_words, final_labels)
                    instruction_inputs = mr_inst_temp.format(entity_type=entity_type, sentence=sentence)
                    mr_example = GenNERSampleWrapper(
                        id=f"{example.id}.M{i}",
                        dataset=example.dataset,
                        split=example.split,
                        label_list=example.label_list,
                        instance=GenNERSample(
                            id=f"{example.id}.M{i}",
                            group=f"{example.id}",
                            words=final_words,
                            labels=final_labels,
                            target_label=entity_type,
                            prompt_labels=prompt_labels,
                            instruction_inputs=instruction_inputs,
                        )
                    )
                    model_input = tokenizer(mr_example.instance.instruction_inputs, return_tensors="pt").to(device)
                    model_outputs = model.generate(
                        **model_input,
                        max_new_tokens=generation_tokens,
                        num_return_sequences=mr_generation_amount,
                        do_sample=generation_by_sample,
                        num_beams=1 if generation_by_sample else mr_generation_amount,
                        temperature=generation_temp if generation_by_sample else None,
                        top_p=generation_top_p if generation_by_sample else None,
                    )

                    valid_prediction_labels = list()
                    for model_output in model_outputs:
                        decoded_output = tokenizer.decode(model_output, skip_special_tokens=True).strip()
                        prediction_labels = extract_predictions3(decoded_output, example=mr_example, tokenizer=tokenizer)
                        if len(mr_example.instance.words) != len(prediction_labels) or any(label not in possible_labels for label in prediction_labels):
                            continue
                        if prediction_labels not in valid_prediction_labels:
                            valid_prediction_labels.append(prediction_labels)
                    mr_example.instance.prediction_outputs = [GenNERSample.get_prompt_labels(example.instance.words, labels) for labels in valid_prediction_labels]
                    logger.debug(f"  * Generate {entity_type:20s} : {len(model_outputs):2d} -> {len(mr_example.instance.prediction_outputs):2d}")

                    mr_examples.append(mr_example)

                # Combine the predictions by replacing the spans in the base prediction
                all_hyps = list()
                for base_idx in range(len(sr_example.instance.prediction_outputs)):
                    base_hyp = sr_example.instance.prediction_outputs[base_idx]
                    sr_hyps = [x for x in sr_example.instance.prediction_outputs if x != base_hyp]
                    if len(sr_hyps) > 0:
                        for init_span, alter_spans in compute_diffs(base_hyp, sr_hyps):
                            for span in alter_spans:
                                new_hyp = base_hyp.replace(init_span, span)
                                if new_hyp not in all_hyps:
                                    all_hyps.append(new_hyp)
                    for mr_example in mr_examples:
                        mr_hyps = mr_example.instance.prediction_outputs
                        for init_span, alter_spans in compute_diffs(base_hyp, mr_hyps):
                            for span in alter_spans:
                                new_hyp = base_hyp.replace(init_span, span)
                                if new_hyp not in all_hyps:
                                    all_hyps.append(new_hyp)
                logger.debug(f"  * Combined {'all candidates':20s} : {len(all_hyps):d}")
                hyps_sum += len(all_hyps)
                progress.set_extra(f"| avg_hyp={hyps_sum.avg:.0f}")
                example.instance.prediction_outputs = all_hyps
                output_file.fp.write(example.model_dump_json() + "\n")

                # Calculate the quality of the predictions (if necessary)
                if do_check_possibility:
                    reference = GenNERSample.get_prompt_labels(sr_example.instance.words, sr_example.instance.labels)
                    quality_hyps = list()
                    for candidate in all_hyps:
                        f1_info = PredictionQuality.calc_f1_info(candidate, example, tokenizer)
                        edit_dist = PredictionQuality.calc_edit_dist(candidate, reference)
                        quality = PredictionQuality.calc_quality(f1_info.f1, edit_dist, weight_f1=weight_f1, weight_ed=weight_ed, pow_weight=pow_weight, max_score=max_score)
                        quality_hyp = PredictionQuality(
                            id=example.id,
                            dataset=example.dataset,
                            sentence=sentence,
                            prediction=candidate,
                            f1_info=f1_info,
                            edit_dist=edit_dist,
                            quality=quality,
                        )
                        quality_hyps.append(quality_hyp)
                    quality_hyps = sorted(quality_hyps, key=lambda x: x.quality, reverse=True)
                    for hyp in quality_hyps[:5]:
                        logger.debug(f"    - {hyp}")
                    f1_sum += quality_hyps[0].f1_info
                    progress.set_extra(f"| avg_hyp={hyps_sum.avg:.0f}, max_f1={f1_sum.f1:.3f}")
                progress.display_message()

        logger.info(f"Saved generated and combined predictions to {output_file.path}: #example={hyps_sum.count}, sum_hyp={hyps_sum.sum:.0f}, avg_hyp={hyps_sum.avg:.0f}, max_f1={f1_sum.f1:.3f}")


@main.command("convert_to_qe_data")
def convert_to_qe_data(
        input_files: Annotated[str, typer.Argument()] = ...,  # "data/GNER-QE/pile-ner-sampled-N19988-part1/*.jsonl", "data/GNER-QE/ZSE-validation-sampled-N210/*.jsonl", "data/GNER-QE/ZSE-test-sampled-N700/*.jsonl"
        output_file: Annotated[str, typer.Option("--output_file")] = "data/GNER-QE/quality_est.jsonl",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER-QE",
        output_home: Annotated[str, typer.Option("--output_home")] = "data",
        pretrained: Annotated[str, typer.Option("--pretrained")] = "output-lfs/train_ZSE-HR207842/GnerT5-Base-HR207842/checkpoint-17052",
        max_sample_per_quality: Annotated[int, typer.Option("--max_sample_per_quality")] = 3,
        weight_f1: Annotated[float, typer.Option("--weight_f1")] = 0.7,
        weight_ed: Annotated[float, typer.Option("--weight_ed")] = 0.3,
        pow_weight: Annotated[float, typer.Option("--pow_weight")] = 2.0,
        max_score: Annotated[float, typer.Option("--max_score")] = 5.0,
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        max_workers: Annotated[int, typer.Option("--max_workers")] = int(os.cpu_count() / 4),
        logging_file: Annotated[str, typer.Option("--logging_file")] = "convert_to_qe_data.out",
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    input_files = Path(input_files)
    output_file = Path(output_file)
    input_file_list = files(input_files)
    assert input_file_list, f"Invalid input files: {input_files}"
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        output_name=output_name,
        output_home=output_home,
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging_level,
        logging_format=LoggingFormat.CHECK_24,
        random_seed=random_seed,
        max_workers=max_workers,
    )
    output_file = (env.output_dir if output_file.parent == Path() else output_file.parent) / new_path(
        output_file.name,
        pre=input_files.parent.name,
        post=f"max_sampled={max_sample_per_quality}"
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    datasets.disable_progress_bar()
    logger.info(f"Convert {input_files}[{len(input_file_list)}] => {output_file}")

    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
        FileStreamer(FileOption.from_path(path=output_file, mode="w")) as output_file,
    ):
        for input_file in input_file_list:
            with FileStreamer(FileOption.from_path(path=input_file)) as input_file:
                logger.info(f"[input_file] {input_file.path}: #item={len(input_file)}")
                with ProgIter(input_file, total=len(input_file), desc=f"  - Converting {input_file.path.stem}:", stream=LoggerWriter(logger, level=logging_level), verbose=3) as progress:
                    f1_sum = F1()
                    combined_sum = Sum()
                    sampled_sum = Sum()
                    for line in progress:
                        example = GenNERSampleWrapper.model_validate_json(line)
                        sentence = " ".join(example.instance.words)
                        reference = GenNERSample.get_prompt_labels(example.instance.words, example.instance.labels)
                        combined_hyps = example.instance.prediction_outputs
                        if len(combined_hyps) == 0:
                            continue
                        logger.debug("=" * 80)
                        logger.debug(f"[Sentence] {sentence}")
                        logger.debug(f"  = #Combined : {len(combined_hyps)}")
                        logger.debug(f"  = Gold : {reference}")
                        combined_sum += len(combined_hyps)

                        # Calculate the quality of the predictions
                        quality_hyps = Dataset.from_dict({
                            "prediction": combined_hyps,
                            "sentence": [sentence] * len(combined_hyps),
                            "dataset": [example.dataset] * len(combined_hyps),
                            "id": [example.id] * len(combined_hyps),
                        })

                        def calc_metrics(row):
                            prediction = row["prediction"]
                            f1_info = PredictionQuality.calc_f1_info(prediction, example, tokenizer)
                            edit_dist = PredictionQuality.calc_edit_dist(prediction, reference)
                            quality = PredictionQuality.calc_quality(f1_info.f1, edit_dist, weight_f1, weight_ed, pow_weight, max_score)
                            return {
                                "f1_info": f1_info.model_dump(),
                                "edit_dist": edit_dist,
                                "quality": quality,
                            }

                        quality_hyps = quality_hyps.map(calc_metrics, num_proc=min(env.max_workers, len(quality_hyps)), desc="Calculating metrics")
                        quality_hyps = quality_hyps.sort("quality", reverse=True)
                        quality_hyps = [PredictionQuality.model_validate(hyp) for hyp in quality_hyps]
                        grouped_hyps = {k: list(vs) for k, vs in grouped(quality_hyps, key=lambda x: x.quality)}
                        sampled_hyps = list()
                        for quality in sorted(grouped_hyps.keys(), reverse=True):
                            num_sampling = min(len(grouped_hyps[quality]), max_sample_per_quality) if max_sample_per_quality > 0 else len(grouped_hyps[quality])
                            for hyp in random.sample(grouped_hyps[quality], num_sampling):
                                sampled_hyps.append(hyp)
                        sampled_sum += len(sampled_hyps)
                        for hyp in sampled_hyps[:10]:
                            logger.debug(f"  = Hypo : {hyp}")
                        logger.debug(f"  = #Sampled : {len(sampled_hyps)}")

                        f1_sum += sampled_hyps[0].f1_info
                        progress.set_extra(f"| avg_combined={combined_sum.avg:.0f}, avg_sampled={sampled_sum.avg:.0f}, max_f1={f1_sum.f1:.3f}")

                        for i, hyp in enumerate(sampled_hyps, start=1):
                            output_file.fp.write(
                                RegressionSample(
                                    sentence1=hyp.prediction,
                                    sentence2=hyp.sentence,
                                    label=hyp.quality,
                                    id=f"{example.id}.{i:d}",
                                ).model_dump_json() + "\n"
                            )
                        logger.debug(f"  = #Saved : {len(sampled_hyps)}")
                        logger.debug("-" * 80)
                        logger.debug("")


@main.command("convert_to_qe_data_seq2seq")
def convert_to_qe_data_seq2seq(
        input_files: Annotated[str, typer.Argument()] = ...,  # "data/GNER-QE/pile-ner-sampled-N19988-part1/*.jsonl", "data/GNER-QE/ZSE-validation-sampled-N210/*.jsonl", "data/GNER-QE/ZSE-test-sampled-N700/*.jsonl"
        output_file: Annotated[str, typer.Option("--output_file")] = "data/GNER-QE-seq2seq/quality_est.jsonl",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER-QE-seq2seq",
        output_home: Annotated[str, typer.Option("--output_home")] = "data",
        pretrained: Annotated[str, typer.Option("--pretrained")] = "output-lfs/train_ZSE-HR207842/GnerT5-Base-HR207842/checkpoint-17052",
        max_sample_per_quality: Annotated[int, typer.Option("--max_sample_per_quality")] = 3,
        weight_f1: Annotated[float, typer.Option("--weight_f1")] = 0.7,
        weight_ed: Annotated[float, typer.Option("--weight_ed")] = 0.3,
        pow_weight: Annotated[float, typer.Option("--pow_weight")] = 2.0,
        max_score: Annotated[float, typer.Option("--max_score")] = 5.0,
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        max_workers: Annotated[int, typer.Option("--max_workers")] = int(os.cpu_count() / 4),
        logging_file: Annotated[str, typer.Option("--logging_file")] = "convert_to_qe_data.out",
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    input_files = Path(input_files)
    output_file = Path(output_file)
    input_file_list = files(input_files)
    assert input_file_list, f"Invalid input files: {input_files}"
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        output_name=output_name,
        output_home=output_home,
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging_level,
        logging_format=LoggingFormat.CHECK_24,
        random_seed=random_seed,
        max_workers=max_workers,
    )
    output_file = (env.output_dir if output_file.parent == Path() else output_file.parent) / new_path(
        output_file.name,
        pre=input_files.parent.name,
        post=f"max_sampled={max_sample_per_quality}"
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    datasets.disable_progress_bar()
    logger.info(f"Convert {input_files}[{len(input_file_list)}] => {output_file}")


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


@main.command("rerank_by_predict_results")
def rerank_by_predict_results(
        generation_file: Annotated[str, typer.Option("--generation_file")] = "data/GNER-QE/ZSE-test-sampled-N700/ZSE-test-sampled-N700-hybrid_gen-by_beam-amount=5x10.jsonl",
        regression_input_files: Annotated[str, typer.Option("--regression_input_files")] = "data/GNER-QE/ZSE-test-sampled-N700-quality_est-max_sampled=0.json",
        regression_output_files: Annotated[str, typer.Option("--regression_output_files")] = "output-lfs/GNER-QE-HR-*/**/checkpoint-*/pred/predict_results_*",
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
        generation_file: Annotated[str, typer.Argument()] = ...,  # "output/ZSE-generate/ZSE-validation-pred-by_beam-num=100.jsonl",  "output/ZSE-generate/ZSE-test-pred-by_beam-num=100.jsonl"
        pretrained: Annotated[str, typer.Option("--pretrained")] = "output-lfs/train_ZSE-HR207842/GnerT5-Base-HR207842/checkpoint-17052",  # "dyyyyyyyy/GNER-T5-base", "dyyyyyyyy/GNER-T5-large", "output-lfs/train_ZSE-HR207842/GnerT5-Base-HR207842/checkpoint-17052"
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
        logger.info(f"F1 performance results saved to {Path(generation_file).parent / env.output_file}")


if __name__ == "__main__":
    main()
