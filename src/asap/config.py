from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_FILE = Path("config/config.yml")

class Paths(BaseModel):
    raw_csv: Path
    processed_dir: Path
    model_dir: Path
    mlm_dir: Path
    cls_dir: Path
    onnx_dir: Path
    experiments_dir: Path

class DataConfig(BaseModel):
    text_column: str
    label_column: str
    val_fraction: float
    test_fraction: float
    stratify: bool
    dedup_on_normalized_text: bool
    min_tokens: int
    max_tokens: int
    labels: dict[int, str]

class MlmConfig(BaseModel):
    base_model: str
    mlm_probability: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_length: int
    use_fp16: bool
    max_rows: int | None = None
    normalize_text: bool = True

class ClassifierConfig(BaseModel):
    base_model: str
    num_labels: int
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_length: int
    use_fp16: bool
    early_stopping_patience: int
    metric_for_best_model: str
    greater_is_better: bool
    save_total_limit: int

class TrainingConfig(BaseModel):
    mlm: MlmConfig
    classifier: ClassifierConfig

class InferenceConfig(BaseModel):
    model_dir: Path
    model_file: str
    provider: Literal["CPUExecutionProvider", "CUDAExecutionProvider"]
    intra_op_num_threads: int
    max_length: int
    batch_size: int
    truncate: bool

class ApiConfig(BaseModel):
    host: str
    port: int
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"]
    cors_origins: list[str]
    max_text_chars: int
    max_batch_items: int

class TrackingConfig(BaseModel):
    uri: str
    artifact_dir: Path
    experiment: str
    registered_model: str
    primary_metric: str
    greater_is_better: bool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASAP_",
        env_nested_delimiter="__",
        env_file=".env",
        yaml_file=CONFIG_FILE,
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    project: dict[str, Any]
    paths: Paths
    data: DataConfig
    training: TrainingConfig
    inference: InferenceConfig
    api: ApiConfig
    tracking: TrackingConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()