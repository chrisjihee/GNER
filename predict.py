import logging

import torch
import typer
from datasets import load_dataset
from typing_extensions import Annotated

from chrisbase.data import AppTyper, JobTimer, NewProjectEnv
from chrisbase.io import LoggingFormat, LoggerWriter, new_path
from chrisbase.time import from_timestamp, now_stamp
from chrisdata.ner import GenNERSampleWrapper
from progiter import ProgIter
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Global settings
logger: logging.Logger = logging.getLogger("gner")


def main(
        device: Annotated[str, typer.Option("--device")] = "cuda:0",
        pretrained: Annotated[str, typer.Option("--pretrained")] = "dyyyyyyyy/GNER-T5-base",  # "dyyyyyyyy/GNER-T5-large", "output-lfs/ZSE-jihee-BL-dl012/FlanT5-Base-BL/checkpoint-9900", "output-lfs/ZSE-yuyang-BL-lirs-b1/checkpoint-9900"
        eval_file: Annotated[str, typer.Option("--eval_file")] = "data/ZSE-validation.jsonl",  # "data/gner/each-sampled/crossner_ai-dev=100.jsonl",
        max_generation_tokens: Annotated[int, typer.Option("--max_generation_tokens")] = 640,
        generation_by_sample: Annotated[bool, typer.Option("--generation_by_sample")] = False,
        generation_num_return: Annotated[int, typer.Option("--generation_num_beams")] = 100,
        generation_temperature: Annotated[float, typer.Option("--generation_temperature")] = 2.0,
        generation_top_p: Annotated[float, typer.Option("--generation_top_p")] = 0.9,
        run_version: Annotated[str, typer.Option("--run_version")] = "ZSE-predict",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER",
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_file: Annotated[str, typer.Option("--output_file")] = "ZSE-validation-pred.jsonl",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "predict-loggings.out",
        verbose: Annotated[int, typer.Option("--verbose")] = 1,
):
    stamp = now_stamp()
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        run_version=run_version,
        output_name=output_name,
        output_home=output_home,
        output_file=new_path(
            output_file, post=f'by_sample-num={generation_num_return}-temp={generation_temperature}-top_p={generation_top_p}'
            if generation_by_sample else f'by_beam-num={generation_num_return}'
        ),
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging.INFO,
        logging_format=LoggingFormat.CHECK_24,
    )
    env.setup_logger(env.logging_level)

    with JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=verbose):
        eval_dataset = load_dataset("json", data_files=str(eval_file), split="train")
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16).to(device)
        model.eval()

        with (
            ProgIter(eval_dataset, desc=f"Predicting:", stream=LoggerWriter(logger), verbose=2) as progress,
            (env.output_dir / env.output_file).open("w") as out,
            torch.no_grad(),
        ):
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
                progress.set_extra(f"| #outputs={num_prediction_outputs}")
                out.write(example.model_dump_json() + "\n")


if __name__ == "__main__":
    AppTyper.run(main)
