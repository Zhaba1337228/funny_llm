from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


TaskType = Literal["classification", "regression"]
TrainingProfile = Literal["rapid", "balanced", "max_accuracy", "server_max"]


class KPIEntry(BaseModel):
    label: str
    value: str | float | int
    delta: str | None = None
    tone: str = "neutral"


class PreviewResponse(BaseModel):
    total_rows: int
    page: int
    page_size: int
    rows: list[dict[str, Any]]
    columns: list[str]


class TrainingRequest(BaseModel):
    task_type: TaskType = "classification"
    training_profile: TrainingProfile = "balanced"
    model_name: str = "random_forest"
    resume_from_run_id: str | None = None
    resume_rounds: int | None = None
    target_column: str | None = None
    feature_columns: list[str] | None = None
    models_to_compare: list[str] | None = None
    test_size: float = 0.2
    validation_size: float = 0.2
    random_seed: int = 42
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    neural_net: dict[str, Any] = Field(default_factory=dict)
    save_as_best: bool = True

    @field_validator("test_size", "validation_size")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if not 0.05 <= value <= 0.4:
            raise ValueError("Split ratios must be between 0.05 and 0.4.")
        return value


class TrainingStatusResponse(BaseModel):
    run_id: str | None = None
    status: str
    progress: float = 0.0
    current_step: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    logs: list[str] = Field(default_factory=list)
    device: str | None = None
    active_model_name: str | None = None
    task_type: TaskType | None = None
    can_stop: bool = False


class PredictionRequest(BaseModel):
    features: dict[str, Any]


class PredictionResponse(BaseModel):
    candidate_score: float
    hire_probability: float
    recommendation: str
    explanation: str
    strengths: list[str]
    weaknesses: list[str]
    feature_contributions: list[dict[str, Any]]


class ModelSelectionRequest(BaseModel):
    run_id: str


class InfoCard(BaseModel):
    title: str
    value: str | int | float
    subtitle: str | None = None
    status: str | None = None


class ModelVersion(BaseModel):
    run_id: str
    created_at: str
    model_name: str
    task_type: TaskType
    target_column: str
    metrics: dict[str, Any]
    is_active: bool = False


class ModelCatalogEntry(BaseModel):
    name: str
    label: str
    task_types: list[TaskType]
    kind: str
    description: str
    default_params: dict[str, Any]
