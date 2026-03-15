from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings
from backend.app.services.dataset_service import DatasetService
from backend.app.utils.serialization import round_float, to_builtin


class EDAService:
    def __init__(self, dataset_service: DatasetService, settings: Settings) -> None:
        self.dataset_service = dataset_service
        self.settings = settings

    def _sample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if len(dataframe) <= self.settings.eda_sample_size:
            return dataframe
        return dataframe.sample(self.settings.eda_sample_size, random_state=self.settings.random_seed)

    def _histogram(self, series: pd.Series, bins: int = 16) -> dict[str, Any]:
        counts, edges = np.histogram(series.astype(float), bins=bins)
        centers = [(edges[idx] + edges[idx + 1]) / 2 for idx in range(len(edges) - 1)]
        return {
            "x": [round_float(value, 3) for value in centers],
            "y": [int(value) for value in counts],
        }

    def _box_stats(self, series: pd.Series) -> dict[str, float]:
        quantiles = series.quantile([0.25, 0.5, 0.75]).to_dict()
        return {
            "min": round_float(series.min(), 3),
            "q1": round_float(quantiles.get(0.25), 3),
            "median": round_float(quantiles.get(0.5), 3),
            "q3": round_float(quantiles.get(0.75), 3),
            "max": round_float(series.max(), 3),
        }

    def get_summary(self) -> dict[str, Any]:
        profile = self.dataset_service.get_profile()
        dataframe = self.dataset_service.load_dataframe()
        sampled = self._sample(dataframe)

        numeric_plots = {}
        box_plots = {}
        for column in profile.numeric_columns:
            numeric_plots[column] = self._histogram(sampled[column])
            box_plots[column] = self._box_stats(sampled[column].astype(float))

        categorical_plots = {}
        for column in profile.categorical_columns:
            counts = sampled[column].astype(str).value_counts().head(12)
            categorical_plots[column] = {
                "x": [str(label) for label in counts.index.tolist()],
                "y": [int(value) for value in counts.tolist()],
            }

        target_column = profile.classification_target or profile.regression_target
        target_distribution = {}
        if target_column:
            target_series = dataframe[target_column]
            if target_series.nunique() <= 20:
                counts = target_series.astype(str).value_counts().sort_index()
                target_distribution = {
                    "x": [str(label) for label in counts.index.tolist()],
                    "y": [int(value) for value in counts.tolist()],
                }
            else:
                target_distribution = self._histogram(target_series.astype(float))

        correlation_columns = profile.numeric_columns + ([profile.classification_target] if profile.classification_target else [])
        numeric_frame = sampled[correlation_columns].corr(numeric_only=True).fillna(0.0)

        return {
            "missing_values": profile.info["missing_values"],
            "target_distribution": target_distribution,
            "class_balance": profile.info["target_distribution"],
            "numeric_distributions": numeric_plots,
            "box_plots": box_plots,
            "categorical_distributions": categorical_plots,
            "correlation": {
                "labels": numeric_frame.columns.tolist(),
                "matrix": numeric_frame.round(3).values.tolist(),
            },
            "dataset_summary": {
                "row_count": profile.info["num_rows"],
                "column_count": profile.info["num_columns"],
                "numeric_feature_count": len(profile.numeric_columns),
                "categorical_feature_count": len(profile.categorical_columns),
            },
            "sample_rows": to_builtin(sampled.head(10).to_dict(orient="records")),
        }
