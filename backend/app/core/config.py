import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Resume Screening System"
    api_prefix: str = "/api"
    random_seed: int = 42
    dataset_slug: str = "rhythmghai/resume-screening-dataset-200k-candidates"
    local_dataset_path: str | None = None
    data_dir: Path = Field(default=ROOT_DIR / "data")
    models_dir: Path = Field(default=ROOT_DIR / "models")
    default_dataset_cache_dir: Path = Field(default=ROOT_DIR / "data" / "kaggle")
    latest_bundle_path: Path = Field(default=ROOT_DIR / "models" / "latest" / "bundle.joblib")
    latest_metadata_path: Path = Field(default=ROOT_DIR / "models" / "latest" / "metadata.json")
    latest_predictions_path: Path = Field(default=ROOT_DIR / "models" / "latest" / "ranked_candidates.csv")
    experiments_log_path: Path = Field(default=ROOT_DIR / "models" / "experiments.json")
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ]
    )
    allowed_origin_regex: str = r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|(\d{1,3}\.){3}\d{1,3})(:\d+)?$"
    ranking_export_name: str = "candidate_ranking_export.csv"
    preview_page_size: int = 20
    eda_sample_size: int = 8000
    ranking_sample_size: int = 15000
    default_training_profile: str = "balanced"
    max_cpu_workers: int = Field(default_factory=lambda: max((os.cpu_count() or 4) - 2, 1))
    torch_data_loader_workers: int = Field(default_factory=lambda: min(max((os.cpu_count() or 4) // 2, 2), 12))
    torch_prefetch_factor: int = 4
    torch_pin_memory: bool = True
    torch_persistent_workers: bool = True
    torch_compile_enabled: bool = True
    torch_amp_enabled: bool = True
    torch_allow_tf32: bool = True
    torch_use_data_parallel: bool = True
    torch_eval_batch_size: int = 65536
    server_max_classification_models: list[str] = Field(
        default_factory=lambda: ["catboost", "xgboost", "hist_gradient_boosting", "extra_trees", "torch_mlp"]
    )
    server_max_regression_models: list[str] = Field(
        default_factory=lambda: ["catboost", "xgboost", "hist_gradient_boosting", "extra_trees", "torch_mlp"]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    (settings.models_dir / "latest").mkdir(parents=True, exist_ok=True)
    (settings.models_dir / "runs").mkdir(parents=True, exist_ok=True)
    return settings
