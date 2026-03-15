from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, task_type: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)
        self.task_type = task_type

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(slots=True)
class TorchRuntimeConfig:
    data_loader_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    eval_batch_size: int = 65536
    amp_enabled: bool = True
    allow_tf32: bool = True
    compile_enabled: bool = True
    use_data_parallel: bool = True


@dataclass(slots=True)
class TorchFitResult:
    state_dict: dict
    history: dict[str, list[float]]
    best_epoch: int
    device: str
    hidden_dims: list[int]
    dropout: float


class TorchTabularPredictor:
    def __init__(
        self,
        task_type: str,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        device: str | None = None,
        runtime_config: TorchRuntimeConfig | None = None,
    ) -> None:
        self.task_type = task_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.runtime = runtime_config or TorchRuntimeConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_device_count = torch.cuda.device_count() if self.device.startswith("cuda") else 0
        self.multi_gpu = self.device.startswith("cuda") and self.cuda_device_count > 1 and self.runtime.use_data_parallel

        if self.device.startswith("cuda") and self.runtime.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        base_model = TabularMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, task_type=task_type).to(self.device)
        self._core_model = base_model
        model: nn.Module = base_model

        if self.multi_gpu:
            model = nn.DataParallel(base_model)
        elif self.device.startswith("cuda") and self.runtime.compile_enabled and hasattr(torch, "compile"):
            try:
                model = torch.compile(base_model)
            except Exception:
                model = base_model

        self.model = model
        self.device_label = f"cuda:0 ({self.cuda_device_count} GPUs)" if self.multi_gpu else self.device

    @property
    def _autocast_device_type(self) -> str:
        return "cuda" if self.device.startswith("cuda") else "cpu"

    def _resolve_loader_workers(self) -> int:
        workers = max(int(self.runtime.data_loader_workers), 0)
        if workers == 0:
            return 0

        if os.name == "nt":
            return 0

        main_module = sys.modules.get("__main__")
        main_file = getattr(main_module, "__file__", "") if main_module else ""
        if not main_file or "<stdin>" in str(main_file):
            return 0

        if not self.device.startswith("cuda"):
            return min(workers, 4)

        return workers

    def _make_grad_scaler(self, amp_enabled: bool):
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler_device = "cuda" if self.device.startswith("cuda") else "cpu"
            try:
                return torch.amp.GradScaler(scaler_device, enabled=amp_enabled)
            except TypeError:
                return torch.amp.GradScaler(device=scaler_device, enabled=amp_enabled)
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def _loader_kwargs(self, shuffle: bool) -> dict:
        workers = self._resolve_loader_workers()
        kwargs = {
            "shuffle": shuffle,
            "drop_last": False,
            "num_workers": workers,
            "pin_memory": bool(self.runtime.pin_memory and self.device.startswith("cuda")),
        }
        if workers > 0:
            kwargs["persistent_workers"] = bool(self.runtime.persistent_workers)
            kwargs["prefetch_factor"] = max(int(self.runtime.prefetch_factor), 2)
        return kwargs

    def _build_loader(self, features: np.ndarray, target: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
        feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
        if target is None:
            dataset = TensorDataset(feature_tensor)
        else:
            target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32))
            dataset = TensorDataset(feature_tensor, target_tensor)
        return DataLoader(dataset, batch_size=batch_size, **self._loader_kwargs(shuffle=shuffle))

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        progress_callback: Callable[[int, int, dict], None] | None = None,
        stop_callback: Callable[[], bool] | None = None,
        early_stopping_patience: int = 8,
        weight_decay: float = 1e-4,
        gradient_clip_norm: float = 1.0,
    ) -> TorchFitResult:
        amp_enabled = bool(self.runtime.amp_enabled and self.device.startswith("cuda"))
        if self.task_type == "classification":
            positives = max(float(np.asarray(train_y).sum()), 1.0)
            negatives = max(float(len(train_y) - positives), 1.0)
            pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self._core_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        scaler = self._make_grad_scaler(amp_enabled=amp_enabled)

        train_loader = self._build_loader(train_x, train_y, batch_size=batch_size, shuffle=True)
        val_loader = self._build_loader(
            val_x,
            val_y,
            batch_size=min(max(batch_size * 2, 2048), int(self.runtime.eval_batch_size)),
            shuffle=False,
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        if self.task_type == "classification":
            history["val_accuracy"] = []

        best_state = None
        best_epoch = 1
        best_val_loss = math.inf
        epochs_without_improvement = 0

        for epoch in range(epochs):
            if stop_callback and stop_callback():
                break

            self.model.train()
            total_loss = 0.0
            total_items = 0
            for batch in train_loader:
                batch_features, batch_target = batch
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_target = batch_target.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self._autocast_device_type,
                    dtype=torch.float16 if self.device.startswith("cuda") else torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    logits = self.model(batch_features)
                    loss = criterion(logits, batch_target)

                scaler.scale(loss).backward()
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(self._core_model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                total_loss += float(loss.item()) * len(batch_target)
                total_items += len(batch_target)

            scheduler.step()
            train_loss = total_loss / max(total_items, 1)
            metrics = self._evaluate_loader(val_loader, criterion, amp_enabled=amp_enabled)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(metrics["loss"])
            if self.task_type == "classification":
                history["val_accuracy"].append(metrics["accuracy"])

            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
                best_state = {key: value.detach().cpu() for key, value in self._core_model.state_dict().items()}
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if progress_callback:
                progress_callback(
                    epoch + 1,
                    epochs,
                    {
                        "train_loss": train_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        **metrics,
                    },
                )

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                break

        if best_state is not None:
            self._core_model.load_state_dict(best_state)

        return TorchFitResult(
            state_dict={key: value.detach().cpu() for key, value in self._core_model.state_dict().items()},
            history=history,
            best_epoch=best_epoch,
            device=self.device_label,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )

    def _evaluate_loader(self, loader: DataLoader, criterion: nn.Module, amp_enabled: bool) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        accuracy_batches: list[float] = []
        with torch.no_grad():
            for batch in loader:
                features = batch[0].to(self.device, non_blocking=True)
                target = batch[1].to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self._autocast_device_type,
                    dtype=torch.float16 if self.device.startswith("cuda") else torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    logits = self.model(features)
                    loss = criterion(logits, target)
                total_loss += float(loss.item()) * len(target)
                total_items += len(target)
                if self.task_type == "classification":
                    predictions = torch.sigmoid(logits)
                    accuracy_batches.append(float(((predictions >= 0.5).float() == target).float().mean().item()))
        metrics = {"loss": total_loss / max(total_items, 1)}
        if self.task_type == "classification":
            metrics["accuracy"] = float(np.mean(accuracy_batches)) if accuracy_batches else 0.0
        return metrics

    def _forward_array(self, features: np.ndarray) -> np.ndarray:
        loader = self._build_loader(
            features,
            None,
            batch_size=min(max(4096, len(features) // 8 or 1), int(self.runtime.eval_batch_size)),
            shuffle=False,
        )
        amp_enabled = bool(self.runtime.amp_enabled and self.device.startswith("cuda"))
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                feature_tensor = batch[0].to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self._autocast_device_type,
                    dtype=torch.float16 if self.device.startswith("cuda") else torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    logits = self.model(feature_tensor)
                outputs.append(logits.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self._forward_array(features)
        probabilities = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, features: np.ndarray) -> np.ndarray:
        predictions = self._forward_array(features)
        if self.task_type == "classification":
            return (1 / (1 + np.exp(-predictions)) >= 0.5).astype(int)
        return predictions

    def load_state(self, state_dict: dict) -> None:
        self._core_model.load_state_dict(state_dict)
