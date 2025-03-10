import json
import logging
from pathlib import Path

import pandas as pd
import typer
from typing_extensions import Annotated

from chrisbase.data import AppTyper, JobTimer, NewProjectEnv
from chrisbase.io import files, dirs, merge_dicts, pop_keys
from chrisbase.util import grouped

# Global settings
main = AppTyper()
logger: logging.Logger = logging.getLogger("gner")


def normalize_epoch(df: pd.DataFrame, unit_epoch: float = 0.5) -> pd.DataFrame:
    df["mapped_epoch"] = round(df["epoch"] / unit_epoch) * unit_epoch
    df["epoch_diff"] = (df["epoch"] - df["mapped_epoch"]).abs()
    df = df.loc[df.groupby("mapped_epoch")["epoch_diff"].idxmin()].copy()
    df.drop(columns=["epoch_diff", "epoch"], inplace=True)
    df.rename(columns={"mapped_epoch": "epoch"}, inplace=True)
    df["epoch"] = df["epoch"].round(1)
    return df


@main.command("summarize")
def summarize(
        input_dirs: Annotated[str, typer.Argument()] = ...,  # "output/GNER-supervised/*/*"
        csv_filename: Annotated[str, typer.Option("--csv_filename")] = "train-metrics-*.csv",
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
        unit_epoch: Annotated[float, typer.Option("--unit_epoch")] = 0.5,
):
    env = NewProjectEnv(logging_level=logging_level)
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
    ):
        no_interest_columns = ['eval_average', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']
        for input_dir in dirs(input_dirs):
            output_file = input_dir.with_suffix(".csv")
            logger.info("[input_dir] %s", input_dir)
            dfs = []
            for input_file in files(input_dir / csv_filename):
                df = pd.read_csv(input_file)
                eval_columns = [col for col in df.columns if col.startswith("eval_") and col not in no_interest_columns]
                df = df.dropna(subset=eval_columns, how="all")[["epoch"] + eval_columns]
                if df.shape[0] == 0:
                    continue
                dfs.append(normalize_epoch(df, unit_epoch=unit_epoch))
            if dfs:
                merged_df = pd.concat(dfs, ignore_index=True)
                merged_df = merged_df.groupby("epoch").first().reset_index()
                eval_columns = [col for col in merged_df.columns if col.startswith("eval_")]
                merged_df["eval_average"] = merged_df[eval_columns].mean(axis=1)
                merged_df = merged_df[["epoch", "eval_average"] + eval_columns]
                merged_df.to_csv(output_file, index=False)
                logger.info(f"            >> {len(dfs)} files are merged into {output_file}")
            else:
                logger.info(f"            >> No files or no evaluation output in input folder")


@main.command("compare")
def compare(
        input_files: Annotated[str, typer.Argument()] = ...,  # "output/GNER-supervised/*/*.csv"
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    env = NewProjectEnv(logging_level=logging_level)
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
    ):
        grouped_results = {k: list(v) for k, v in grouped(files(input_files), key=lambda x: Path(x).parent)}
        for group_name, group_evals in grouped_results.items():
            output_file = group_name.with_suffix(".csv")
            logger.info("[group_name] %s", group_name)
            for x in group_evals:
                logger.info("    - [file] %s", x)

            best_rows = []
            for csv_file in group_evals:
                df = pd.read_csv(csv_file)
                df["experiment_id"] = csv_file.stem
                best_idx = df['eval_average'].idxmax()
                best_row = df.loc[best_idx]
                best_rows.append(best_row)
            best_df = pd.DataFrame(best_rows)
            best_df.reset_index(drop=True, inplace=True)
            rename_map = {x: x.replace("eval_", "") for x in best_df.columns}
            rename_map = {k: f"{'best_' if v in ["epoch", "average"] else ''}{v}" for k, v in rename_map.items()}
            best_df.rename(columns=rename_map, inplace=True)
            first_columns = ["experiment_id", "best_average", "best_epoch"]
            new_columns = first_columns + [x for x in best_df.columns if x not in first_columns]
            best_df = best_df[new_columns]
            best_df.sort_values(by="experiment_id", inplace=True)
            best_df.to_csv(output_file, index=False)


@main.command("summarize_trainer_state_json")
def summarize_trainer_state_json(
        input_dirs: Annotated[str, typer.Argument()] = ...,  # "output/GLUE-STSb/**", "output/NER-CONLL/**", "output/GNER-QE/**"
        task_type: Annotated[str, typer.Argument()] = "regression",  # "classification"
        json_filename: Annotated[str, typer.Option("--json_filename")] = "trainer_state.json",
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    env = NewProjectEnv(logging_level=logging_level)
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
    ):
        output_file = Path(input_dirs).parent.with_suffix(".csv")
        common_columns1 = ["model", "step", "epoch"]
        common_columns2 = ["eval_loss", "train_loss", "eval_runtime", "train_runtime"]
        if task_type == "regression":
            interest_columns = common_columns1 + ["eval_pearson", "eval_spearmanr"] + common_columns2
        else:
            interest_columns = common_columns1 + ["eval_accuracy", "eval_f1"] + common_columns2
        logger.info(f"Interest Columns: {', '.join(interest_columns)}")

        model_dfs = []
        for input_dir in dirs(input_dirs):
            for input_file in [x for x in files(input_dir / json_filename) if not x.parent.name.startswith("checkpoint-")]:
                logger.info("[input_file] %s", input_file)
                trainer_state = json.load(input_file.open())
                log_history = trainer_state.get("log_history", [])

                metrics_per_step = {k: merge_dicts(*vs) for k, vs in grouped(log_history, itemgetter="step")}
                model_df = pd.DataFrame(metrics_per_step).transpose()
                model_df["model"] = input_dir.name
                model_dfs.append(model_df[interest_columns])
        all_model_df = pd.concat(model_dfs)
        all_model_df.to_csv(output_file, index=False)
        logger.info("Summary is saved to %s", output_file)


@main.command("summarize_generation_eval_csv")
def summarize_generation_eval_csv(
        input_dirs: Annotated[str, typer.Argument()] = "output/ZSE-predict",  # "output/ZSE-predict"
        csv_filename: Annotated[str, typer.Option("--csv_filename")] = ...,  # "ZSE-test-*-eval.csv", "ZSE-validation-*-eval.csv"
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    env = NewProjectEnv(logging_level=logging_level)
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
    ):
        output_file = Path(input_dirs).with_suffix(".csv")
        interest_columns = ['method', 'candidate', 'average',
                            'ai', 'literature', 'music', 'politics', 'science', 'movie', 'restaurant']

        all_dfs = []
        for input_dir in dirs(input_dirs):
            logger.info("[input_dir] %s", input_dir)
            for input_file in files(input_dir / csv_filename):
                logger.info("  [input_file] %s", input_file)
                df = pd.read_csv(input_file)
                df["method"] = input_file.name.replace("-eval.csv", "").replace("-num=100", "").split("by_")[-1]
                df = df[df['candidate'].isin([1, 2, 3, 4, 5]) | (df['candidate'] % 10 == 0)]
                df = df[interest_columns]
                all_dfs.append(df)
        all_dfs = pd.concat(all_dfs)
        all_dfs.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
