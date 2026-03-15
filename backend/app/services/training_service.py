from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.app.core.config import Settings
from backend.app.ml.model_factory import build_sklearn_model, get_descriptor, list_model_catalog
from backend.app.ml.preprocessing import build_preprocessor
from backend.app.ml.torch_model import TorchRuntimeConfig, TorchTabularPredictor
from backend.app.ml.training_profiles import apply_training_profile, list_training_profiles
from backend.app.schemas.api import TrainingRequest
from backend.app.services.dataset_service import DatasetService, SCORING_COLUMN
from backend.app.utils.serialization import round_float, to_builtin


BEST_MODEL_METRIC = {
    "classification": ("f1", max),
    "regression": ("r2", max),
}


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class TrainingStatus:
    run_id: str | None = None
    status: str = "idle"
    progress: float = 0.0
    current_step: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    logs: list[str] = field(default_factory=list)
    device: str | None = None
    active_model_name: str | None = None
    task_type: str | None = None
    can_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingService:
    def __init__(self, dataset_service: DatasetService, settings: Settings) -> None:
        self.dataset_service = dataset_service
        self.settings = settings
        self._gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        try:
            torch.set_num_threads(max(int(settings.max_cpu_workers), 1))
            torch.set_num_interop_threads(max(min(int(settings.max_cpu_workers) // 2, 8), 1))
        except RuntimeError:
            pass
        self._status = TrainingStatus()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._active_bundle: dict[str, Any] | None = None
        self._latest_results: dict[str, Any] | None = None
        self._prediction_cache: pd.DataFrame | None = None
        self._load_latest_bundle()

    def _set_status(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._status, key, value)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._status.logs = [*self._status.logs[-79:], f"[{timestamp}] {message}"]

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = self._status.to_dict()
        if status["started_at"] and status["status"] == "training":
            started_at = datetime.fromisoformat(status["started_at"])
            status["elapsed_seconds"] = round((datetime.now(tz=timezone.utc) - started_at).total_seconds(), 1)
        return status

    def list_models(self) -> dict[str, Any]:
        active_run_id = self._active_bundle.get("run_id") if self._active_bundle else None
        versions: list[dict[str, Any]] = []
        runs_dir = self.settings.models_dir / "runs"
        if runs_dir.exists():
            for metadata_file in sorted(runs_dir.glob("*/metadata.json"), reverse=True):
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                versions.append(
                    {
                        "run_id": metadata["run_id"],
                        "created_at": metadata["trained_at"],
                        "model_name": metadata["model_name"],
                        "task_type": metadata["task_type"],
                        "target_column": metadata["target_column"],
                        "metrics": metadata["metrics"],
                        "is_active": metadata["run_id"] == active_run_id,
                        "resumable": metadata["model_name"] in {"catboost", "xgboost", "torch_mlp"},
                    }
                )
        experiments = []
        if self.settings.experiments_log_path.exists():
            experiments = json.loads(self.settings.experiments_log_path.read_text(encoding="utf-8"))
        return {
            "catalog": list_model_catalog(),
            "training_profiles": list_training_profiles(),
            "saved_versions": versions,
            "recent_experiments": experiments[-12:][::-1],
        }

    def _available_model_names(self, task_type: str) -> list[str]:
        return [entry["name"] for entry in list_model_catalog() if task_type in entry["task_types"]]

    def _load_run_bundle(self, run_id: str) -> dict[str, Any]:
        run_bundle_path = self.settings.models_dir / "runs" / run_id / "bundle.joblib"
        if not run_bundle_path.exists():
            raise FileNotFoundError(f"Model run '{run_id}' was not found.")
        return joblib.load(run_bundle_path)

    def _resume_supported(self, model_name: str) -> bool:
        return model_name in {"catboost", "xgboost", "torch_mlp"}

    def _prepare_request(self, request: TrainingRequest) -> TrainingRequest:
        if request.resume_from_run_id:
            bundle = self._load_run_bundle(request.resume_from_run_id)
            model_name = bundle["model_name"]
            if not self._resume_supported(model_name):
                raise ValueError(f"Resume training is not supported for model '{model_name}'.")

            effective_request = request.model_copy(deep=True)
            effective_request.task_type = bundle["task_type"]
            effective_request.model_name = model_name
            effective_request.target_column = bundle["target_column"]
            effective_request.feature_columns = bundle["feature_columns"]
            effective_request.models_to_compare = [model_name]
            effective_request.training_profile = bundle.get("training_profile", effective_request.training_profile)
            if not effective_request.hyperparameters:
                effective_request.hyperparameters = dict(bundle.get("hyperparameters", {}))
            else:
                effective_request.hyperparameters = {**bundle.get("hyperparameters", {}), **effective_request.hyperparameters}
            if not effective_request.neural_net:
                effective_request.neural_net = dict(bundle.get("neural_net", {}))
            else:
                effective_request.neural_net = {**bundle.get("neural_net", {}), **effective_request.neural_net}

            if effective_request.resume_rounds:
                if model_name == "torch_mlp":
                    effective_request.neural_net["epochs"] = int(effective_request.resume_rounds)
                elif model_name == "catboost":
                    effective_request.hyperparameters["iterations"] = int(effective_request.resume_rounds)
                elif model_name == "xgboost":
                    effective_request.hyperparameters["n_estimators"] = int(effective_request.resume_rounds)
            return effective_request

        available_models = self._available_model_names(request.task_type)
        effective_request = apply_training_profile(
            request=request,
            settings=self.settings,
            available_models=available_models,
            gpu_count=self._gpu_count,
        )
        if not effective_request.models_to_compare:
            effective_request.models_to_compare = [effective_request.model_name]
        if effective_request.model_name not in effective_request.models_to_compare:
            effective_request.models_to_compare = [effective_request.model_name, *effective_request.models_to_compare]
        return effective_request

    def _torch_runtime_config(self) -> TorchRuntimeConfig:
        return TorchRuntimeConfig(
            data_loader_workers=self.settings.torch_data_loader_workers,
            pin_memory=self.settings.torch_pin_memory,
            persistent_workers=self.settings.torch_persistent_workers,
            prefetch_factor=self.settings.torch_prefetch_factor,
            eval_batch_size=self.settings.torch_eval_batch_size,
            amp_enabled=self.settings.torch_amp_enabled,
            allow_tf32=self.settings.torch_allow_tf32,
            compile_enabled=self.settings.torch_compile_enabled,
            use_data_parallel=self.settings.torch_use_data_parallel,
        )

    def _describe_estimator_device(self, pipeline: Pipeline) -> str:
        model = pipeline.named_steps.get("model")
        if model is None:
            return "cpu"
        params = model.get_params(deep=True)
        device = str(params.get("device", "")).lower()
        task_type = str(params.get("task_type", "")).lower()
        devices = params.get("devices")
        if device.startswith("cuda"):
            return device
        if task_type == "gpu":
            return f"gpu:{devices}" if devices else "gpu"
        return "cpu"

    def _resolve_columns(self, request: TrainingRequest, dataframe: pd.DataFrame) -> tuple[list[str], str]:
        profile = self.dataset_service.get_profile()
        if request.task_type == "classification":
            target_column = request.target_column or profile.classification_target
            if not target_column:
                raise ValueError("No classification target column is available for this dataset.")
        else:
            target_column = request.target_column or profile.regression_target

        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")

        feature_columns = request.feature_columns or [*profile.numeric_columns, *profile.categorical_columns]
        if not feature_columns:
            raise ValueError("No feature columns were selected.")
        unknown_columns = [column for column in feature_columns if column not in dataframe.columns]
        if unknown_columns:
            raise ValueError(f"Unknown feature columns: {', '.join(unknown_columns)}")
        return feature_columns, target_column

    def start_training(self, request: TrainingRequest) -> dict[str, Any]:
        with self._lock:
            if self._status.status == "training":
                raise RuntimeError("Training is already running.")
            self._status = TrainingStatus(
                run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                status="training",
                progress=0.0,
                current_step="Initializing training pipeline",
                started_at=utc_now_iso(),
                logs=[],
                task_type=request.task_type,
                can_stop=True,
            )
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_training, args=(request,), daemon=True)
        self._worker.start()
        return self.get_status()

    def stop_training(self) -> dict[str, Any]:
        if self._status.status != "training":
            return self.get_status()
        self._stop_event.set()
        self._append_log("Stop requested. The current fit will finish before shutting down.")
        self._set_status(current_step="Stop requested")
        return self.get_status()

    def _split_dataset(self, features: pd.DataFrame, target: pd.Series, request: TrainingRequest):
        stratify_target = target if request.task_type == "classification" else None
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            target,
            test_size=request.test_size,
            random_state=request.random_seed,
            stratify=stratify_target,
        )
        if request.validation_size > 0:
            adjusted_val_ratio = request.validation_size / (1 - request.test_size)
            stratify_train = train_y if request.task_type == "classification" else None
            train_x, val_x, train_y, val_y = train_test_split(
                train_x,
                train_y,
                test_size=adjusted_val_ratio,
                random_state=request.random_seed,
                stratify=stratify_train,
            )
        else:
            val_x, val_y = test_x, test_y
        return train_x, val_x, test_x, train_y, val_y, test_y

    def _classification_metrics(self, y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
        y_pred = (probabilities >= 0.5).astype(int)
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        max_points = 300
        if len(fpr) > max_points:
            step = max(len(fpr) // max_points, 1)
            indices = list(range(0, len(fpr), step))
            if indices[-1] != len(fpr) - 1:
                indices.append(len(fpr) - 1)
            fpr = fpr[indices]
            tpr = tpr[indices]
            thresholds = thresholds[indices]
        return {
            "accuracy": round_float(accuracy_score(y_true, y_pred), 4),
            "precision": round_float(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round_float(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round_float(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round_float(roc_auc_score(y_true, probabilities), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "roc_curve": {
                "fpr": [round_float(value, 6) for value in fpr.tolist()],
                "tpr": [round_float(value, 6) for value in tpr.tolist()],
                "thresholds": [round_float(value, 6) for value in thresholds.tolist()],
            },
        }

    def _regression_metrics(self, y_true: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
        sample_size = min(400, len(y_true))
        return {
            "mae": round_float(mean_absolute_error(y_true, predictions), 4),
            "rmse": round_float(mean_squared_error(y_true, predictions, squared=False), 4),
            "r2": round_float(r2_score(y_true, predictions), 4),
            "prediction_error": {
                "y_true": [round_float(value, 3) for value in y_true[:sample_size]],
                "y_pred": [round_float(value, 3) for value in predictions[:sample_size]],
            },
        }

    def _compute_feature_importance(
        self,
        artifact_type: str,
        model_object: Any,
        task_type: str,
        sample_x: pd.DataFrame,
        sample_y: pd.Series,
        feature_columns: list[str],
    ) -> list[dict[str, Any]]:
        if artifact_type == "sklearn":
            estimator = model_object.named_steps.get("model") if isinstance(model_object, Pipeline) else model_object
            native_importances = None
            if estimator is not None and hasattr(estimator, "feature_importances_"):
                native_importances = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
            elif estimator is not None and hasattr(estimator, "coef_"):
                coefficients = np.asarray(getattr(estimator, "coef_"), dtype=float)
                native_importances = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)

            if native_importances is not None and native_importances.size > 0:
                importances = native_importances[: len(feature_columns)]
            else:
                scoring = "f1" if task_type == "classification" else "r2"
                result = permutation_importance(
                    model_object,
                    sample_x,
                    sample_y,
                    n_repeats=2,
                    random_state=self.settings.random_seed,
                    n_jobs=max(min(self.settings.max_cpu_workers // 2, 12), 1),
                    scoring=scoring,
                )
                importances = result.importances_mean
        else:
            correlations = []
            sample_frame = sample_x.copy()
            for column in feature_columns:
                if pd.api.types.is_numeric_dtype(sample_frame[column]):
                    corr = np.corrcoef(sample_frame[column].astype(float), sample_y.astype(float))[0, 1]
                    correlations.append(0.0 if np.isnan(corr) else abs(corr))
                else:
                    target_mean = (
                        pd.DataFrame({"feature": sample_frame[column].astype(str), "target": sample_y})
                        .groupby("feature")["target"]
                        .mean()
                    )
                    spread = float(target_mean.max() - target_mean.min()) if not target_mean.empty else 0.0
                    correlations.append(spread)
            importances = np.array(correlations)

        normalized = importances / max(float(importances.sum()), 1e-9)
        ranking = sorted(
            [{"feature": feature_columns[idx], "importance": round_float(value, 4)} for idx, value in enumerate(normalized)],
            key=lambda item: item["importance"],
            reverse=True,
        )
        return ranking

    def _compute_directionality(self, dataframe: pd.DataFrame, feature_columns: list[str], target_column: str) -> dict[str, Any]:
        directionality: dict[str, Any] = {"numeric": {}, "categorical": {}, "feature_stats": {}}
        target_series = dataframe[target_column]
        overall_mean = float(target_series.mean())

        for column in feature_columns:
            series = dataframe[column]
            if pd.api.types.is_numeric_dtype(series):
                corr = np.corrcoef(series.astype(float), target_series.astype(float))[0, 1]
                directionality["numeric"][column] = 1 if np.nan_to_num(corr) >= 0 else -1
                directionality["feature_stats"][column] = {
                    "median": round_float(series.median(), 4),
                    "std": round_float(series.std(ddof=0) or 1.0, 4),
                }
            else:
                mapping = dataframe.groupby(column)[target_column].mean().to_dict()
                directionality["categorical"][column] = {
                    str(key): round_float(float(value - overall_mean), 4)
                    for key, value in mapping.items()
                }
                directionality["feature_stats"][column] = {
                    "top": str(series.mode().iloc[0]) if not series.mode().empty else "",
                }
        return directionality

    def _recommendation_from_outputs(self, candidate_score: float, hire_probability: float) -> str:
        if hire_probability >= 0.82 or candidate_score >= 82:
            return "strong_hire"
        if hire_probability >= 0.65 or candidate_score >= 68:
            return "shortlist"
        if hire_probability >= 0.45 or candidate_score >= 50:
            return "maybe"
        return "reject"

    def _train_sklearn_model(
        self,
        model_name: str,
        task_type: str,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        val_x: pd.DataFrame,
        val_y: pd.Series,
        request: TrainingRequest,
        numeric_features: list[str],
        categorical_features: list[str],
        resume_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()

        if resume_bundle:
            previous_pipeline = resume_bundle["model_object"]
            preprocessor = previous_pipeline.named_steps["preprocessor"]
            previous_estimator = previous_pipeline.named_steps["model"]
            estimator = build_sklearn_model(
                model_name,
                task_type,
                request.random_seed,
                request.hyperparameters,
                prefer_gpu=self._gpu_count > 0,
                gpu_count=self._gpu_count,
                cpu_threads=self.settings.max_cpu_workers,
            )
            transformed_train = np.asarray(preprocessor.transform(train_x))
            transformed_val = np.asarray(preprocessor.transform(val_x))

            fit_kwargs: dict[str, Any] = {}
            if model_name == "xgboost" and hasattr(previous_estimator, "get_booster"):
                fit_kwargs["xgb_model"] = previous_estimator.get_booster()
            elif model_name == "catboost":
                fit_kwargs["init_model"] = previous_estimator
            else:
                raise ValueError(f"Resume training is not supported for model '{model_name}'.")

            estimator.fit(transformed_train, train_y, **fit_kwargs)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            if task_type == "classification":
                probabilities = estimator.predict_proba(transformed_val)[:, 1]
                metrics = self._classification_metrics(val_y.to_numpy(), probabilities)
            else:
                predictions = estimator.predict(transformed_val)
                metrics = self._regression_metrics(val_y.to_numpy(), predictions)
        else:
            preprocessor = build_preprocessor(numeric_features, categorical_features)
            estimator = build_sklearn_model(
                model_name,
                task_type,
                request.random_seed,
                request.hyperparameters,
                prefer_gpu=self._gpu_count > 0,
                gpu_count=self._gpu_count,
                cpu_threads=self.settings.max_cpu_workers,
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            pipeline.fit(train_x, train_y)
            if task_type == "classification":
                probabilities = pipeline.predict_proba(val_x)[:, 1]
                metrics = self._classification_metrics(val_y.to_numpy(), probabilities)
            else:
                predictions = pipeline.predict(val_x)
                metrics = self._regression_metrics(val_y.to_numpy(), predictions)

        duration = round(time.perf_counter() - started, 2)
        return {
            "artifact_type": "sklearn",
            "model_object": pipeline,
            "metrics": metrics,
            "history": {},
            "training_time_seconds": duration,
            "device": self._describe_estimator_device(pipeline),
        }

    def _train_torch_model(
        self,
        task_type: str,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        val_x: pd.DataFrame,
        val_y: pd.Series,
        request: TrainingRequest,
        numeric_features: list[str],
        categorical_features: list[str],
        resume_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        neural_defaults = get_descriptor("torch_mlp").default_params
        previous_history: dict[str, list[float]] = {}
        initial_state: dict | None = None

        if resume_bundle:
            model_config = resume_bundle["model_object"]
            preprocessor = model_config["preprocessor"]
            transformed_train = preprocessor.transform(train_x)
            transformed_val = preprocessor.transform(val_x)
            neural_config = {
                **neural_defaults,
                "hidden_dims": model_config["hidden_dims"],
                "dropout": model_config["dropout"],
                **resume_bundle.get("neural_net", {}),
                **request.neural_net,
                **request.hyperparameters,
            }
            previous_history = {key: list(value) for key, value in resume_bundle.get("history", {}).items()}
            initial_state = model_config["state_dict"]
        else:
            preprocessor = build_preprocessor(numeric_features, categorical_features)
            transformed_train = preprocessor.fit_transform(train_x)
            transformed_val = preprocessor.transform(val_x)
            neural_config = {**neural_defaults, **request.neural_net, **request.hyperparameters}

        predictor = TorchTabularPredictor(
            task_type=task_type,
            input_dim=int(transformed_train.shape[1]),
            hidden_dims=[int(value) for value in neural_config.get("hidden_dims", neural_defaults["hidden_dims"])],
            dropout=float(neural_config.get("dropout", neural_defaults["dropout"])),
            runtime_config=self._torch_runtime_config(),
        )
        if initial_state is not None:
            predictor.load_state(initial_state)
        self._set_status(device=predictor.device_label)

        def progress_callback(epoch: int, total_epochs: int, metrics: dict[str, Any]) -> None:
            epoch_progress = 0.25 + (epoch / max(total_epochs, 1)) * 0.55
            self._set_status(progress=min(epoch_progress, 0.86), current_step=f"Training neural network epoch {epoch}/{total_epochs}")
            metric_text = ", ".join(f"{key}={round_float(value, 4)}" for key, value in metrics.items())
            self._append_log(f"Epoch {epoch}/{total_epochs}: {metric_text}")

        started = time.perf_counter()
        fit_result = predictor.fit(
            train_x=np.asarray(transformed_train, dtype=np.float32),
            train_y=train_y.to_numpy(dtype=np.float32),
            val_x=np.asarray(transformed_val, dtype=np.float32),
            val_y=val_y.to_numpy(dtype=np.float32),
            epochs=int(neural_config.get("epochs", neural_defaults["epochs"])),
            batch_size=int(neural_config.get("batch_size", neural_defaults["batch_size"])),
            lr=float(neural_config.get("lr", neural_defaults["lr"])),
            progress_callback=progress_callback,
            stop_callback=self._stop_event.is_set,
            early_stopping_patience=int(neural_config.get("early_stopping_patience", neural_defaults["early_stopping_patience"])),
            weight_decay=float(neural_config.get("weight_decay", neural_defaults["weight_decay"])),
            gradient_clip_norm=float(neural_config.get("gradient_clip_norm", neural_defaults["gradient_clip_norm"])),
        )
        duration = round(time.perf_counter() - started, 2)

        if task_type == "classification":
            probabilities = predictor.predict_proba(np.asarray(transformed_val, dtype=np.float32))[:, 1]
            metrics = self._classification_metrics(val_y.to_numpy(), probabilities)
        else:
            predictions = predictor.predict(np.asarray(transformed_val, dtype=np.float32))
            metrics = self._regression_metrics(val_y.to_numpy(), predictions)

        return {
            "artifact_type": "torch",
            "model_object": {
                "preprocessor": preprocessor,
                "state_dict": fit_result.state_dict,
                "input_dim": predictor.input_dim,
                "hidden_dims": fit_result.hidden_dims,
                "dropout": fit_result.dropout,
                "task_type": task_type,
                "inference_device": predictor.device,
            },
            "metrics": metrics,
            "history": {
                key: [*previous_history.get(key, []), *fit_result.history.get(key, [])]
                for key in set(previous_history) | set(fit_result.history)
            },
            "training_time_seconds": duration,
            "device": fit_result.device,
        }

    def _predict_with_bundle(self, bundle: dict[str, Any], feature_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if bundle["artifact_type"] == "sklearn":
            pipeline = bundle["model_object"]
            if bundle["task_type"] == "classification":
                probabilities = pipeline.predict_proba(feature_frame)[:, 1]
                return probabilities, probabilities * 100
            predictions = pipeline.predict(feature_frame)
            return np.clip(predictions / 100.0, 0, 1), predictions

        model_config = bundle["model_object"]
        preprocessor = model_config["preprocessor"]
        transformed = np.asarray(preprocessor.transform(feature_frame), dtype=np.float32)
        predictor = TorchTabularPredictor(
            task_type=model_config["task_type"],
            input_dim=model_config["input_dim"],
            hidden_dims=model_config["hidden_dims"],
            dropout=model_config["dropout"],
            device=model_config.get("inference_device") if torch.cuda.is_available() else "cpu",
            runtime_config=self._torch_runtime_config(),
        )
        predictor.load_state(model_config["state_dict"])
        if bundle["task_type"] == "classification":
            probabilities = predictor.predict_proba(transformed)[:, 1]
            return probabilities, probabilities * 100
        predictions = predictor.predict(transformed)
        return np.clip(predictions / 100.0, 0, 1), predictions

    def _build_prediction_cache(self, bundle: dict[str, Any]) -> pd.DataFrame:
        dataframe = self.dataset_service.load_dataframe()
        feature_frame = dataframe[bundle["feature_columns"]]
        probabilities, scores = self._predict_with_bundle(bundle, feature_frame)
        ranking = dataframe[["candidate_id"]].copy()
        ranking["predicted_score"] = np.clip(scores, 0, 100).round(3)
        ranking["hire_probability"] = np.clip(probabilities, 0, 1).round(4)
        ranking["recommendation"] = [
            self._recommendation_from_outputs(candidate_score=float(score), hire_probability=float(probability))
            for score, probability in zip(ranking["predicted_score"], ranking["hire_probability"], strict=False)
        ]
        ranking = ranking.sort_values(["predicted_score", "hire_probability"], ascending=[False, False]).reset_index(drop=True)
        ranking["rank"] = np.arange(1, len(ranking) + 1)
        merged = ranking.merge(dataframe, on="candidate_id", how="left")
        return merged

    def _summarize_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": bundle["run_id"],
            "trained_at": bundle["trained_at"],
            "task_type": bundle["task_type"],
            "training_profile": bundle.get("training_profile", self.settings.default_training_profile),
            "model_name": bundle["model_name"],
            "target_column": bundle["target_column"],
            "feature_columns": bundle["feature_columns"],
            "metrics": bundle["metrics"],
            "history": bundle["history"],
            "comparison": bundle["comparison"],
            "feature_importance": bundle["feature_importance"],
            "synthetic_mode": bundle["synthetic_mode"],
            "device": bundle["device"],
        }

    def _persist_bundle(self, bundle: dict[str, Any]) -> None:
        runs_dir = self.settings.models_dir / "runs" / bundle["run_id"]
        latest_dir = self.settings.models_dir / "latest"
        runs_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = runs_dir / "bundle.joblib"
        metadata_path = runs_dir / "metadata.json"
        predictions_path = runs_dir / "ranked_candidates.csv"

        joblib.dump(bundle, bundle_path)
        bundle["prediction_cache"].to_csv(predictions_path, index=False)

        metadata = self._summarize_bundle(bundle)
        metadata["metrics"] = to_builtin(metadata["metrics"])
        metadata["history"] = to_builtin(metadata["history"])
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        joblib.dump(bundle, self.settings.latest_bundle_path)
        bundle["prediction_cache"].to_csv(self.settings.latest_predictions_path, index=False)
        self.settings.latest_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._append_experiment(metadata)

    def _append_experiment(self, metadata: dict[str, Any]) -> None:
        experiments: list[dict[str, Any]] = []
        if self.settings.experiments_log_path.exists():
            experiments = json.loads(self.settings.experiments_log_path.read_text(encoding="utf-8"))
        experiments.append(
            {
                "run_id": metadata["run_id"],
                "trained_at": metadata["trained_at"],
                "task_type": metadata["task_type"],
                "training_profile": metadata.get("training_profile", self.settings.default_training_profile),
                "model_name": metadata["model_name"],
                "target_column": metadata["target_column"],
                "metrics": metadata["metrics"],
                "device": metadata["device"],
            }
        )
        self.settings.experiments_log_path.write_text(json.dumps(experiments[-50:], indent=2), encoding="utf-8")

    def _load_latest_bundle(self) -> None:
        if not self.settings.latest_bundle_path.exists():
            return
        try:
            bundle = joblib.load(self.settings.latest_bundle_path)
        except Exception:
            return
        self._active_bundle = bundle
        self._latest_results = self._summarize_bundle(bundle)
        if self.settings.latest_predictions_path.exists():
            self._prediction_cache = pd.read_csv(self.settings.latest_predictions_path)

    def select_model(self, run_id: str) -> dict[str, Any]:
        run_bundle_path = self.settings.models_dir / "runs" / run_id / "bundle.joblib"
        if not run_bundle_path.exists():
            raise FileNotFoundError(f"Model run '{run_id}' was not found.")
        bundle = joblib.load(run_bundle_path)
        self._active_bundle = bundle
        self._latest_results = self._summarize_bundle(bundle)
        self._prediction_cache = bundle["prediction_cache"]
        joblib.dump(bundle, self.settings.latest_bundle_path)
        if self._prediction_cache is not None:
            self._prediction_cache.to_csv(self.settings.latest_predictions_path, index=False)
        self.settings.latest_metadata_path.write_text(json.dumps(self._latest_results, indent=2), encoding="utf-8")
        return self._latest_results

    def get_results(self) -> dict[str, Any]:
        if not self._latest_results:
            raise RuntimeError("No trained model is available yet.")
        return self._latest_results

    def get_comparison(self) -> dict[str, Any]:
        if not self._latest_results:
            return {"comparison": [], "best_model": None}
        return {
            "comparison": self._latest_results.get("comparison", []),
            "best_model": self._latest_results.get("model_name"),
        }

    def _run_training(self, request: TrainingRequest) -> None:
        try:
            effective_request = self._prepare_request(request)
            dataframe = self.dataset_service.load_dataframe()
            feature_columns, target_column = self._resolve_columns(effective_request, dataframe)
            resume_bundle = self._load_run_bundle(effective_request.resume_from_run_id) if effective_request.resume_from_run_id else None
            self._append_log(
                f"Dataset loaded with {len(dataframe):,} candidates, target '{target_column}', "
                f"profile '{effective_request.training_profile}', GPUs={self._gpu_count}, CPU workers={self.settings.max_cpu_workers}."
            )
            if resume_bundle:
                self._append_log(f"Resuming from run '{resume_bundle['run_id']}' using model '{resume_bundle['model_name']}'.")
            self._set_status(progress=0.08, current_step="Preparing train/validation/test split")

            work_frame = dataframe[feature_columns + [target_column]].dropna(subset=[target_column]).copy()
            features = work_frame[feature_columns]
            target = work_frame[target_column]

            train_x, val_x, test_x, train_y, val_y, test_y = self._split_dataset(features, target, effective_request)
            numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(features[column])]
            categorical_features = [column for column in feature_columns if column not in numeric_features]

            models_to_compare = effective_request.models_to_compare or [effective_request.model_name]
            if effective_request.model_name not in models_to_compare:
                models_to_compare = [effective_request.model_name, *models_to_compare]
            results_table: list[dict[str, Any]] = []
            trained_bundles: dict[str, dict[str, Any]] = {}

            for index, model_name in enumerate(models_to_compare, start=1):
                if self._stop_event.is_set():
                    raise InterruptedError("Training was stopped by the user.")
                descriptor = get_descriptor(model_name)
                if effective_request.task_type not in descriptor.task_types:
                    continue
                self._append_log(f"Training model {index}/{len(models_to_compare)}: {descriptor.label}.")
                self._set_status(
                    current_step=f"Training {descriptor.label}",
                    active_model_name=model_name,
                    progress=0.12 + ((index - 1) / max(len(models_to_compare), 1)) * 0.22,
                )

                if descriptor.kind == "torch":
                    outcome = self._train_torch_model(
                        task_type=effective_request.task_type,
                        train_x=train_x,
                        train_y=train_y,
                        val_x=val_x,
                        val_y=val_y,
                        request=effective_request,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        resume_bundle=resume_bundle if resume_bundle and model_name == resume_bundle["model_name"] else None,
                    )
                else:
                    outcome = self._train_sklearn_model(
                        model_name=model_name,
                        task_type=effective_request.task_type,
                        train_x=train_x,
                        train_y=train_y,
                        val_x=val_x,
                        val_y=val_y,
                        request=effective_request,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        resume_bundle=resume_bundle if resume_bundle and model_name == resume_bundle["model_name"] else None,
                    )

                if self._stop_event.is_set():
                    raise InterruptedError("Training was stopped by the user.")

                if effective_request.task_type == "classification":
                    if outcome["artifact_type"] == "sklearn":
                        test_values = outcome["model_object"].predict_proba(test_x)[:, 1]
                    else:
                        test_values, _ = self._predict_with_bundle(
                            {
                                **outcome,
                                "task_type": effective_request.task_type,
                                "model_name": model_name,
                                "feature_columns": feature_columns,
                            },
                            test_x,
                        )
                    self._set_status(
                        current_step=f"Evaluating {descriptor.label}",
                        active_model_name=model_name,
                        progress=0.36 + ((index - 1) / max(len(models_to_compare), 1)) * 0.18,
                    )
                    test_metrics = self._classification_metrics(test_y.to_numpy(), np.asarray(test_values))
                else:
                    if outcome["artifact_type"] == "sklearn":
                        test_values = outcome["model_object"].predict(test_x)
                    else:
                        _, test_values = self._predict_with_bundle(
                            {
                                **outcome,
                                "task_type": effective_request.task_type,
                                "model_name": model_name,
                                "feature_columns": feature_columns,
                            },
                            test_x,
                        )
                    self._set_status(
                        current_step=f"Evaluating {descriptor.label}",
                        active_model_name=model_name,
                        progress=0.36 + ((index - 1) / max(len(models_to_compare), 1)) * 0.18,
                    )
                    test_metrics = self._regression_metrics(test_y.to_numpy(), np.asarray(test_values))

                sample_count = min(self.settings.ranking_sample_size, len(test_x))
                sample_x = test_x.iloc[:sample_count]
                sample_y = test_y.iloc[:sample_count]
                self._set_status(
                    current_step=f"Computing feature importance for {descriptor.label}",
                    active_model_name=model_name,
                    progress=0.46 + ((index - 1) / max(len(models_to_compare), 1)) * 0.18,
                )
                self._append_log(f"Computing feature importance for {descriptor.label}.")
                feature_importance = self._compute_feature_importance(
                    artifact_type=outcome["artifact_type"],
                    model_object=outcome["model_object"],
                    task_type=effective_request.task_type,
                    sample_x=sample_x,
                    sample_y=sample_y,
                    feature_columns=feature_columns,
                )

                trained_bundle = {
                    "run_id": self._status.run_id,
                    "trained_at": utc_now_iso(),
                    "artifact_type": outcome["artifact_type"],
                    "task_type": effective_request.task_type,
                    "training_profile": effective_request.training_profile,
                    "model_name": model_name,
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                    "metrics": test_metrics,
                    "history": outcome["history"],
                    "feature_importance": feature_importance,
                    "directionality": self._compute_directionality(work_frame, feature_columns, target_column),
                    "synthetic_mode": target_column == SCORING_COLUMN,
                    "device": outcome["device"],
                    "model_object": outcome["model_object"],
                    "comparison": [],
                    "hyperparameters": effective_request.hyperparameters,
                    "neural_net": effective_request.neural_net,
                    "resumed_from_run_id": resume_bundle["run_id"] if resume_bundle else None,
                }
                trained_bundles[model_name] = trained_bundle

                comparator_key, _ = BEST_MODEL_METRIC[effective_request.task_type]
                results_table.append(
                    {
                        "model_name": model_name,
                        "label": descriptor.label,
                        "kind": descriptor.kind,
                        "training_time_seconds": outcome["training_time_seconds"],
                        "device": outcome["device"],
                        "metrics": test_metrics,
                        "score_for_selection": test_metrics[comparator_key],
                        "score_direction": "max",
                    }
                )
                self._append_log(
                    f"{descriptor.label} finished in {outcome['training_time_seconds']}s with "
                    f"{comparator_key}={test_metrics[comparator_key]}."
                )

            if not results_table:
                raise RuntimeError("No compatible models were trained.")

            comparator_key, reducer = BEST_MODEL_METRIC[effective_request.task_type]
            best_entry = reducer(results_table, key=lambda item: item["score_for_selection"])
            active_name = effective_request.model_name if effective_request.model_name in trained_bundles else best_entry["model_name"]
            if effective_request.save_as_best:
                active_name = best_entry["model_name"]

            bundle = trained_bundles[active_name]
            bundle["comparison"] = results_table

            self._append_log(f"Best model selected: {bundle['model_name']} on {bundle['device']}.")
            self._set_status(progress=0.9, current_step="Building full candidate ranking cache", device=bundle["device"])
            prediction_cache = self._build_prediction_cache(bundle)
            bundle["prediction_cache"] = prediction_cache
            self._prediction_cache = prediction_cache
            self._active_bundle = bundle
            self._latest_results = self._summarize_bundle(bundle)

            self._persist_bundle(bundle)
            self._set_status(
                status="trained",
                progress=1.0,
                current_step="Training complete",
                finished_at=utc_now_iso(),
                active_model_name=bundle["model_name"],
                device=bundle["device"],
                can_stop=False,
            )
            self._append_log(f"Active model set to '{bundle['model_name']}'.")
        except InterruptedError as exc:
            self._set_status(status="stopped", finished_at=utc_now_iso(), can_stop=False)
            self._append_log(str(exc))
        except Exception as exc:
            self._set_status(status="failed", finished_at=utc_now_iso(), can_stop=False)
            self._append_log(f"Training failed: {exc}")

    def _feature_title(self, name: str) -> str:
        labels = {
            "cgpa": "academic performance",
            "internships": "internship experience",
            "projects": "project portfolio",
            "programming_languages": "technical stack breadth",
            "certifications": "certification profile",
            "experience_years": "professional experience",
            "hackathons": "hackathon exposure",
            "research_papers": "research depth",
            "skills_score": "technical skills score",
            "soft_skills_score": "soft-skills score",
            "resume_length_words": "resume depth",
            "education_level": "education level",
            "university_tier": "university tier",
            "company_type": "company fit preference",
        }
        return labels.get(name, name.replace("_", " "))

    def _generate_explanation(self, bundle: dict[str, Any], candidate_row: pd.Series) -> dict[str, Any]:
        importance_lookup = {item["feature"]: float(item["importance"]) for item in bundle["feature_importance"]}
        directionality = bundle["directionality"]
        contributions: list[dict[str, Any]] = []
        for feature in bundle["feature_columns"]:
            importance = importance_lookup.get(feature, 0.0)
            if feature in directionality["numeric"]:
                stats = directionality["feature_stats"].get(feature, {})
                median = float(stats.get("median", 0.0))
                std = max(float(stats.get("std", 1.0)), 1e-6)
                delta = (float(candidate_row[feature]) - median) / std
                impact = delta * directionality["numeric"][feature] * importance
            else:
                category_effect = directionality["categorical"].get(feature, {}).get(str(candidate_row[feature]), 0.0)
                impact = float(category_effect) * importance
            contributions.append(
                {
                    "feature": feature,
                    "label": self._feature_title(feature),
                    "impact": round_float(impact, 4),
                    "importance": round_float(importance, 4),
                    "value": to_builtin(candidate_row[feature]),
                }
            )
        contributions.sort(key=lambda item: abs(float(item["impact"])), reverse=True)
        strengths = [f"Strong {item['label']}" for item in contributions if item["impact"] > 0][:3]
        weaknesses = [f"Weaker {item['label']}" for item in contributions if item["impact"] < 0][:3]

        overview_parts = []
        if strengths:
            overview_parts.append(", ".join(strengths).lower().replace("strong ", ""))
        if weaknesses:
            overview_parts.append(", ".join(weaknesses).lower().replace("weaker ", ""))
        explanation = (
            "Prediction is driven mostly by " + "; ".join(overview_parts[:2])
            if overview_parts
            else "Prediction is based on the full candidate profile with balanced feature contributions."
        )
        return {
            "explanation": explanation,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "feature_contributions": contributions[:8],
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._active_bundle:
            raise RuntimeError("No trained model is available.")
        dataframe = self.dataset_service.load_dataframe()
        row = {}
        for feature in self._active_bundle["feature_columns"]:
            if feature not in payload:
                if feature in dataframe.columns:
                    if dataframe[feature].dtype == "object":
                        row[feature] = dataframe[feature].mode().iloc[0]
                    else:
                        row[feature] = float(dataframe[feature].median())
                else:
                    row[feature] = None
                continue
            row[feature] = payload[feature]

        feature_frame = pd.DataFrame([row])
        probabilities, scores = self._predict_with_bundle(self._active_bundle, feature_frame)
        candidate_score = float(np.clip(scores[0], 0, 100))
        hire_probability = float(np.clip(probabilities[0], 0, 1))
        explanation = self._generate_explanation(self._active_bundle, feature_frame.iloc[0])
        return {
            "candidate_score": round_float(candidate_score, 3),
            "hire_probability": round_float(hire_probability, 4),
            "recommendation": self._recommendation_from_outputs(candidate_score, hire_probability),
            **explanation,
        }

    def get_top_candidates(
        self,
        limit: int = 50,
        search: str | None = None,
        min_score: float | None = None,
        experience_min: float | None = None,
        internships_min: int | None = None,
        certifications_min: int | None = None,
        skill_score_min: float | None = None,
        company_type: str | None = None,
    ) -> dict[str, Any]:
        if self._prediction_cache is None:
            raise RuntimeError("Ranking is unavailable because no model has been trained.")

        ranked = self._prediction_cache.copy()
        if search:
            lowered = search.lower()
            mask = ranked["candidate_id"].astype(str).str.contains(lowered, na=False)
            for column in ["education_level", "university_tier", "company_type"]:
                if column in ranked.columns:
                    mask = mask | ranked[column].astype(str).str.lower().str.contains(lowered, na=False)
            ranked = ranked.loc[mask]
        if min_score is not None:
            ranked = ranked.loc[ranked["predicted_score"] >= min_score]
        if experience_min is not None and "experience_years" in ranked.columns:
            ranked = ranked.loc[ranked["experience_years"] >= experience_min]
        if internships_min is not None and "internships" in ranked.columns:
            ranked = ranked.loc[ranked["internships"] >= internships_min]
        if certifications_min is not None and "certifications" in ranked.columns:
            ranked = ranked.loc[ranked["certifications"] >= certifications_min]
        if skill_score_min is not None and "skills_score" in ranked.columns:
            ranked = ranked.loc[ranked["skills_score"] >= skill_score_min]
        if company_type:
            ranked = ranked.loc[ranked["company_type"].astype(str).str.lower() == company_type.lower()]

        total = len(ranked)
        records = ranked.head(limit).replace({np.nan: None}).to_dict(orient="records")
        return {
            "total_rows": int(total),
            "rows": to_builtin(records),
            "active_model": self._active_bundle["model_name"] if self._active_bundle else None,
            "task_type": self._active_bundle["task_type"] if self._active_bundle else None,
        }

    def get_candidate_detail(self, candidate_id: int) -> dict[str, Any]:
        if self._prediction_cache is None or not self._active_bundle:
            raise RuntimeError("Candidate detail is unavailable because no model has been trained.")

        match = self._prediction_cache.loc[self._prediction_cache["candidate_id"] == candidate_id]
        if match.empty:
            raise KeyError(f"Candidate {candidate_id} was not found.")
        record = match.iloc[0]
        explanation = self._generate_explanation(self._active_bundle, record[self._active_bundle["feature_columns"]])
        return {
            "candidate": to_builtin(record.replace({np.nan: None}).to_dict()),
            **explanation,
        }

    def export_ranking(self) -> Path:
        if self._prediction_cache is None:
            raise RuntimeError("No ranking is available for export.")
        export_path = self.settings.models_dir / self.settings.ranking_export_name
        self._prediction_cache.to_csv(export_path, index=False)
        return export_path
