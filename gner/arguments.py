import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing_extensions import Self

from chrisbase.data import NewCommonArguments
from chrisbase.util import to_dataframe
from transformers import Seq2SeqTrainingArguments

logger = logging.getLogger(__name__)


class CustomDataArguments(BaseModel):
    data_dir: str | Path | None = Field(default=None)
    data_config_dir: str | Path | None = Field(default=None)
    instruct_file: str | Path | None = Field(default=None)
    train_file: str | Path | None = Field(default=None)
    study_file: str | Path | None = Field(default=None)
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
    write_predictions: bool = Field(default=False)
    ignore_pad_token_for_loss: bool = Field(default=True)

    @model_validator(mode='after')
    def after(self) -> Self:
        self.pretrained = Path(self.pretrained) if self.pretrained else None
        self.train_file = Path(self.train_file) if self.train_file else None
        self.study_file = Path(self.study_file) if self.study_file else None
        self.eval_file = Path(self.eval_file) if self.eval_file else None
        self.pred_file = Path(self.pred_file) if self.pred_file else None
        return self

    @property
    def cache_train_dir(self) -> Optional[Path]:
        if self.train_file:
            return self.train_file.parent / ".cache"

    @property
    def cache_study_dir(self) -> Optional[Path]:
        if self.study_file:
            return self.study_file.parent / ".cache"

    @property
    def cache_eval_dir(self) -> Optional[Path]:
        if self.eval_file:
            return self.eval_file.parent / ".cache"

    @property
    def cache_pred_dir(self) -> Optional[Path]:
        if self.pred_file:
            return self.pred_file.parent / ".cache"

    def cache_train_path(self, suffix: str) -> Optional[str]:
        if self.train_file:
            return str(self.cache_train_dir / f"{self.train_file.stem}={suffix}.tmp")

    def cache_study_path(self, suffix: str) -> Optional[str]:
        if self.study_file:
            return str(self.cache_study_dir / f"{self.study_file.stem}={suffix}.tmp")

    def cache_eval_path(self, suffix: str) -> Optional[str]:
        if self.eval_file:
            return str(self.cache_eval_dir / f"{self.eval_file.stem}={suffix}.tmp")

    def cache_pred_path(self, suffix: str) -> Optional[str]:
        if self.pred_file:
            return str(self.cache_pred_dir / f"{self.pred_file.stem}={suffix}.tmp")


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
