from __future__ import annotations

from copy import deepcopy
from typing import Any

from backend.app.core.config import Settings
from backend.app.schemas.api import TrainingRequest


TRAINING_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "rapid": {
        "classification": {
            "preferred_model": "hist_gradient_boosting",
            "models_to_compare": ["hist_gradient_boosting", "random_forest"],
            "hyperparameters": {"n_estimators": 260, "max_depth": 8, "learning_rate": 0.08},
            "neural_net": {
                "hidden_dims": [256, 128],
                "dropout": 0.15,
                "epochs": 16,
                "batch_size": 2048,
                "lr": 0.001,
                "early_stopping_patience": 5,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
            },
        },
        "regression": {
            "preferred_model": "hist_gradient_boosting",
            "models_to_compare": ["hist_gradient_boosting", "random_forest"],
            "hyperparameters": {"n_estimators": 260, "max_depth": 8, "learning_rate": 0.08},
            "neural_net": {
                "hidden_dims": [256, 128],
                "dropout": 0.15,
                "epochs": 16,
                "batch_size": 2048,
                "lr": 0.001,
                "early_stopping_patience": 5,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
            },
        },
    },
    "balanced": {
        "classification": {
            "preferred_model": "random_forest",
            "models_to_compare": ["catboost", "xgboost", "hist_gradient_boosting", "extra_trees", "torch_mlp"],
            "hyperparameters": {"n_estimators": 600, "max_depth": 10, "learning_rate": 0.05, "subsample": 0.9},
            "neural_net": {
                "hidden_dims": [512, 256, 128],
                "dropout": 0.12,
                "epochs": 32,
                "batch_size": 4096,
                "lr": 0.0008,
                "early_stopping_patience": 8,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
            },
        },
        "regression": {
            "preferred_model": "hist_gradient_boosting",
            "models_to_compare": ["catboost", "xgboost", "hist_gradient_boosting", "extra_trees", "torch_mlp"],
            "hyperparameters": {"n_estimators": 600, "max_depth": 10, "learning_rate": 0.05, "subsample": 0.9},
            "neural_net": {
                "hidden_dims": [512, 256, 128],
                "dropout": 0.12,
                "epochs": 32,
                "batch_size": 4096,
                "lr": 0.0008,
                "early_stopping_patience": 8,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
            },
        },
    },
    "max_accuracy": {
        "classification": {
            "preferred_model": "xgboost",
            "models_to_compare": ["catboost", "xgboost", "hist_gradient_boosting", "extra_trees", "torch_mlp"],
            "hyperparameters": {
                "n_estimators": 1200,
                "max_depth": 10,
                "learning_rate": 0.035,
                "subsample": 0.93,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.6,
                "max_bin": 512,
            },
            "neural_net": {
                "hidden_dims": [768, 512, 256, 128],
                "dropout": 0.1,
                "epochs": 48,
                "batch_size": 8192,
                "lr": 0.0007,
                "early_stopping_patience": 10,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
            },
        },
        "regression": {
            "preferred_model": "xgboost",
            "models_to_compare": ["catboost", "xgboost", "hist_gradient_boosting", "extra_trees", "torch_mlp"],
            "hyperparameters": {
                "n_estimators": 1200,
                "max_depth": 10,
                "learning_rate": 0.035,
                "subsample": 0.93,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.6,
                "max_bin": 512,
            },
            "neural_net": {
                "hidden_dims": [768, 512, 256, 128],
                "dropout": 0.1,
                "epochs": 48,
                "batch_size": 8192,
                "lr": 0.0007,
                "early_stopping_patience": 10,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
            },
        },
    },
    "server_max": {
        "classification": {
            "preferred_model": "catboost",
            "models_to_compare": ["catboost", "xgboost", "torch_mlp"],
            "hyperparameters": {
                "iterations": 2200,
                "n_estimators": 1800,
                "max_depth": 12,
                "depth": 10,
                "learning_rate": 0.025,
                "subsample": 0.95,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.8,
                "l2_leaf_reg": 5.0,
                "max_bin": 512,
            },
            "neural_net": {
                "hidden_dims": [1024, 768, 512, 256, 128],
                "dropout": 0.08,
                "epochs": 72,
                "batch_size": 16384,
                "lr": 0.0006,
                "early_stopping_patience": 14,
                "weight_decay": 0.00008,
                "gradient_clip_norm": 1.0,
            },
        },
        "regression": {
            "preferred_model": "catboost",
            "models_to_compare": ["catboost", "xgboost", "torch_mlp"],
            "hyperparameters": {
                "iterations": 2200,
                "n_estimators": 1800,
                "max_depth": 12,
                "depth": 10,
                "learning_rate": 0.025,
                "subsample": 0.95,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.8,
                "l2_leaf_reg": 5.0,
                "max_bin": 512,
            },
            "neural_net": {
                "hidden_dims": [1024, 768, 512, 256, 128],
                "dropout": 0.08,
                "epochs": 72,
                "batch_size": 16384,
                "lr": 0.0006,
                "early_stopping_patience": 14,
                "weight_decay": 0.00008,
                "gradient_clip_norm": 1.0,
            },
        },
    },
}


def list_training_profiles() -> list[dict[str, str]]:
    return [
        {"name": "rapid", "label": "Rapid", "description": "Fast baseline training with lower compute cost."},
        {"name": "balanced", "label": "Balanced", "description": "Good default for most local environments."},
        {"name": "max_accuracy", "label": "Max Accuracy", "description": "Heavier compare set with stronger tuning."},
        {"name": "server_max", "label": "Server Max", "description": "Aggressive profile for multi-GPU, high-core servers."},
    ]


def apply_training_profile(
    request: TrainingRequest,
    settings: Settings,
    available_models: list[str],
    gpu_count: int,
) -> TrainingRequest:
    profile_name = request.training_profile or settings.default_training_profile
    profile = deepcopy(TRAINING_PROFILES.get(profile_name, TRAINING_PROFILES["balanced"])[request.task_type])

    if profile_name == "server_max":
        configured_models = (
            settings.server_max_classification_models
            if request.task_type == "classification"
            else settings.server_max_regression_models
        )
        if configured_models:
            profile["models_to_compare"] = configured_models
            profile["preferred_model"] = configured_models[0]

    if profile_name == "server_max" and gpu_count > 1:
        profile["neural_net"]["batch_size"] = int(profile["neural_net"]["batch_size"] * gpu_count)

    effective = request.model_copy(deep=True)
    if not effective.model_name or effective.model_name not in available_models:
        preferred = profile["preferred_model"]
        effective.model_name = preferred if preferred in available_models else available_models[0]

    if not effective.models_to_compare:
        effective.models_to_compare = [model for model in profile["models_to_compare"] if model in available_models]

    effective.hyperparameters = {**profile["hyperparameters"], **effective.hyperparameters}
    effective.neural_net = {**profile["neural_net"], **effective.neural_net}
    return effective
