import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional, Dict

import datasets
import pandas as pd
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing_extensions import Self

from chrisbase.data import NewCommonArguments
from chrisbase.util import to_dataframe
from transformers import Seq2SeqTrainingArguments

logger = logging.getLogger(__name__)


class CustomDataArguments(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data_dir: str | Path | None = Field(default=None)
    data_config_dir: str | Path | None = Field(default=None)
    instruct_file: str | Path | None = Field(default=None)
    train_file: str | Path | None = Field(default=None)
    eval_file: str | Path | None = Field(default=None)
    pred_file: str | Path | None = Field(default=None)
    pretrained: str | Path = Field(default=None)
    max_train_samples: int = Field(default=-1)
    max_study_samples: int = Field(default=-1)
    max_eval_samples: int = Field(default=-1)
    max_pred_samples: int = Field(default=-1)
    use_cache_data: bool = Field(default=True)
    progress_seconds: float = Field(default=2.0)
    max_source_length: int = Field(default=512)
    max_target_length: int = Field(default=512)
    max_generation_tokens: int = Field(default=1280)
    ignore_pad_token_for_loss: bool = Field(default=True)
    write_predictions: bool = Field(default=False)
    data_files: Dict[datasets.Split, Optional[Path]] = Field(default=None, init=False)

    @model_validator(mode='after')
    def after(self) -> Self:
        self.data_dir = Path(self.data_dir) if self.data_dir else None
        self.data_config_dir = Path(self.data_config_dir) if self.data_config_dir else None
        self.instruct_file = Path(self.instruct_file) if self.instruct_file else None
        self.train_file = Path(self.train_file) if self.train_file else None
        self.eval_file = Path(self.eval_file) if self.eval_file else None
        self.pred_file = Path(self.pred_file) if self.pred_file else None
        self.pretrained = Path(self.pretrained) if self.pretrained else None
        self.data_files = {
            datasets.Split.TRAIN: self.train_file,
            datasets.Split.VALIDATION: self.eval_file,
            datasets.Split.TEST: self.pred_file,
        }
        return self

    def data_file(self, split: datasets.Split) -> Optional[Path]:
        assert split in self.data_files, f"Split {split} not in {self.data_files.keys()}"
        return self.data_files[split]

    def cache_file(self, split: datasets.Split, data_size: int, tokenizer_path: str) -> Optional[str]:
        data_file = self.data_file(split)
        if data_file:
            return str(data_file.parent / ".cache" / f"{data_file.stem}={data_size}={tokenizer_path.replace('/', '--')}.tmp")


@dataclass
class ExSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    logging_epochs: float = field(
        default=0.1,
        metadata={"help": "Log every X epochs."},
    )
    eval_epochs: float = field(
        default=0.1,
        metadata={"help": "Run an evaluation every X epochs."},
    )
    save_epochs: float = field(
        default=0.1,
        metadata={"help": "Save checkpoint every X epochs."},
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Use Flash Attention 2."},
    )


class TrainingArgumentsForAccelerator(NewCommonArguments):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: CustomDataArguments = Field(default=None)
    train: ExSeq2SeqTrainingArguments = Field(default=None)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.__class__.__name__, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.train, data_prefix="train", sorted_keys=True),
        ]).reset_index(drop=True)
        return df
