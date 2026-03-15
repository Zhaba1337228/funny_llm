from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None


@dataclass(slots=True)
class ModelDescriptor:
    name: str
    label: str
    task_types: list[str]
    kind: str
    description: str
    default_params: dict


MODEL_CATALOG: list[ModelDescriptor] = [
    ModelDescriptor(
        name="logistic_regression",
        label="Logistic Regression",
        task_types=["classification"],
        kind="sklearn",
        description="Fast linear baseline with calibrated class probabilities.",
        default_params={"max_iter": 300, "class_weight": "balanced", "C": 1.0},
    ),
    ModelDescriptor(
        name="random_forest",
        label="Random Forest",
        task_types=["classification", "regression"],
        kind="sklearn",
        description="Robust tree ensemble with strong nonlinear performance.",
        default_params={"n_estimators": 220, "max_depth": 14, "min_samples_leaf": 2},
    ),
    ModelDescriptor(
        name="extra_trees",
        label="Extra Trees",
        task_types=["classification", "regression"],
        kind="sklearn",
        description="High-variance ensemble useful for tabular ranking tasks.",
        default_params={"n_estimators": 260, "max_depth": 18, "min_samples_leaf": 2},
    ),
    ModelDescriptor(
        name="hist_gradient_boosting",
        label="Hist Gradient Boosting",
        task_types=["classification", "regression"],
        kind="sklearn",
        description="Fast boosting baseline that scales well to large tabular datasets.",
        default_params={"max_depth": 8, "learning_rate": 0.08, "max_iter": 180},
    ),
    ModelDescriptor(
        name="linear_regression",
        label="Linear Regression",
        task_types=["regression"],
        kind="sklearn",
        description="Transparent baseline for synthetic candidate score regression.",
        default_params={},
    ),
    ModelDescriptor(
        name="torch_mlp",
        label="PyTorch Neural Network",
        task_types=["classification", "regression"],
        kind="torch",
        description="GPU-capable multilayer perceptron for tabular modeling.",
        default_params={
            "hidden_dims": [512, 256, 128],
            "dropout": 0.12,
            "epochs": 32,
            "batch_size": 4096,
            "lr": 0.0008,
            "early_stopping_patience": 8,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
        },
    ),
]

if CatBoostClassifier is not None and CatBoostRegressor is not None:
    MODEL_CATALOG.append(
        ModelDescriptor(
            name="catboost",
            label="CatBoost",
            task_types=["classification", "regression"],
            kind="sklearn",
            description="Strong tabular gradient boosting model with native GPU support.",
            default_params={"iterations": 1200, "depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 4.0},
        )
    )

if XGBClassifier is not None and XGBRegressor is not None:
    MODEL_CATALOG.append(
        ModelDescriptor(
            name="xgboost",
            label="XGBoost",
            task_types=["classification", "regression"],
            kind="sklearn",
            description="Gradient boosting model with excellent tabular accuracy.",
            default_params={
                "n_estimators": 900,
                "max_depth": 8,
                "learning_rate": 0.04,
                "subsample": 0.92,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.4,
            },
        )
    )


def list_model_catalog() -> list[dict]:
    return [asdict(descriptor) for descriptor in MODEL_CATALOG]


def get_descriptor(model_name: str) -> ModelDescriptor:
    for descriptor in MODEL_CATALOG:
        if descriptor.name == model_name:
            return descriptor
    raise ValueError(f"Unsupported model '{model_name}'.")


def get_default_models(task_type: str) -> list[str]:
    preferred = ["catboost", "xgboost", "hist_gradient_boosting", "torch_mlp", "extra_trees"]
    available = [item.name for item in MODEL_CATALOG if task_type in item.task_types]
    ordered = [name for name in preferred if name in available]
    for name in available:
        if name not in ordered:
            ordered.append(name)
    return ordered[:4]


def build_sklearn_model(
    model_name: str,
    task_type: str,
    random_seed: int,
    hyperparameters: dict | None = None,
    prefer_gpu: bool = False,
    gpu_count: int = 0,
    cpu_threads: int | None = None,
):
    params = {**get_descriptor(model_name).default_params, **(hyperparameters or {})}

    def filtered(estimator):
        valid_keys = set(estimator.get_params(deep=True).keys())
        safe_params = {key: value for key, value in params.items() if key in valid_keys}
        estimator.set_params(**safe_params)
        return estimator

    if model_name == "logistic_regression":
        return filtered(LogisticRegression(random_state=random_seed, n_jobs=None))
    if model_name == "random_forest":
        if task_type == "classification":
            return filtered(RandomForestClassifier(random_state=random_seed, n_jobs=cpu_threads or -1))
        return filtered(RandomForestRegressor(random_state=random_seed, n_jobs=cpu_threads or -1))
    if model_name == "extra_trees":
        if task_type == "classification":
            return filtered(ExtraTreesClassifier(random_state=random_seed, n_jobs=cpu_threads or -1))
        return filtered(ExtraTreesRegressor(random_state=random_seed, n_jobs=cpu_threads or -1))
    if model_name == "hist_gradient_boosting":
        if task_type == "classification":
            return filtered(HistGradientBoostingClassifier(random_state=random_seed))
        return filtered(HistGradientBoostingRegressor(random_state=random_seed))
    if model_name == "linear_regression":
        return filtered(LinearRegression())
    if model_name == "catboost" and CatBoostClassifier is not None and CatBoostRegressor is not None:
        base_params = {
            "random_seed": random_seed,
            "verbose": False,
            "thread_count": cpu_threads or -1,
        }
        if prefer_gpu and gpu_count > 0:
            base_params["task_type"] = "GPU"
            if gpu_count > 1:
                base_params["devices"] = ":".join(str(index) for index in range(gpu_count))
        else:
            base_params["task_type"] = "CPU"
        if task_type == "classification":
            return filtered(CatBoostClassifier(**base_params))
        return filtered(CatBoostRegressor(**base_params))
    if model_name == "xgboost" and XGBClassifier is not None and XGBRegressor is not None:
        base_params = {
            "random_state": random_seed,
            "tree_method": "hist",
            "n_jobs": cpu_threads or 0,
        }
        if prefer_gpu and gpu_count > 0:
            base_params["device"] = "cuda"
        if task_type == "classification":
            return filtered(XGBClassifier(eval_metric="logloss", **base_params))
        return filtered(XGBRegressor(**base_params))
    raise ValueError(f"Model '{model_name}' is not available for task '{task_type}'.")
