from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings
from backend.app.utils.serialization import round_float, to_builtin

try:
    import kagglehub
except Exception:  # pragma: no cover - dependency may be missing at runtime
    kagglehub = None


TARGET_CANDIDATES = ["hired", "hire_decision", "shortlisted", "target", "label", "recommendation"]
SCORING_COLUMN = "synthetic_candidate_score"
SYNTHETIC_RECOMMENDATION_COLUMN = "synthetic_recommendation"


@dataclass(slots=True)
class DatasetProfile:
    info: dict[str, Any]
    numeric_columns: list[str]
    categorical_columns: list[str]
    classification_target: str | None
    regression_target: str
    available_tasks: list[str]
    feature_schema: list[dict[str, Any]]


class DatasetService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._dataset_path: Path | None = None
        self._dataframe: pd.DataFrame | None = None
        self._profile: DatasetProfile | None = None

    def _download_or_locate_dataset(self) -> Path:
        if self._dataset_path and self._dataset_path.exists():
            return self._dataset_path

        explicit_path = Path(self.settings.local_dataset_path) if self.settings.local_dataset_path else None
        if explicit_path and explicit_path.exists():
            search_root = explicit_path
        else:
            if kagglehub is None:
                raise RuntimeError("kagglehub is not installed. Install it or set LOCAL_DATASET_PATH.")
            try:
                search_root = Path(kagglehub.dataset_download(self.settings.dataset_slug))
            except Exception as exc:  # pragma: no cover - depends on runtime connectivity
                raise RuntimeError(f"Failed to download dataset from Kaggle: {exc}") from exc

        csv_files = sorted(search_root.rglob("*.csv"), key=lambda item: item.stat().st_size, reverse=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files were found under '{search_root}'.")
        self._dataset_path = csv_files[0]
        return self._dataset_path

    def _build_synthetic_score(self, dataframe: pd.DataFrame) -> pd.Series:
        frame = dataframe.copy()
        education_map = {"Bachelors": 0.72, "Masters": 0.88, "PhD": 1.0}
        tier_map = {"Tier 1": 1.0, "Tier 2": 0.82, "Tier 3": 0.68}
        company_map = {"MNC": 0.92, "Startup": 0.84, "Mid-size": 0.8}

        components = {
            "cgpa": frame.get("cgpa", 0).astype(float).clip(0, 10) / 10.0,
            "internships": frame.get("internships", 0).astype(float).clip(0, 6) / 6.0,
            "projects": frame.get("projects", 0).astype(float).clip(0, 10) / 10.0,
            "programming_languages": frame.get("programming_languages", 0).astype(float).clip(0, 8) / 8.0,
            "certifications": frame.get("certifications", 0).astype(float).clip(0, 8) / 8.0,
            "experience_years": frame.get("experience_years", 0).astype(float).clip(0, 8) / 8.0,
            "hackathons": frame.get("hackathons", 0).astype(float).clip(0, 6) / 6.0,
            "research_papers": frame.get("research_papers", 0).astype(float).clip(0, 5) / 5.0,
            "skills_score": frame.get("skills_score", 0).astype(float).clip(0, 100) / 100.0,
            "soft_skills_score": frame.get("soft_skills_score", 0).astype(float).clip(0, 10) / 10.0,
            "resume_length_words": frame.get("resume_length_words", 0).astype(float).clip(150, 900) / 900.0,
            "education_level": frame.get("education_level", "").map(education_map).fillna(0.76),
            "university_tier": frame.get("university_tier", "").map(tier_map).fillna(0.78),
            "company_type": frame.get("company_type", "").map(company_map).fillna(0.8),
        }
        weights = {
            "cgpa": 0.16,
            "internships": 0.12,
            "projects": 0.1,
            "programming_languages": 0.08,
            "certifications": 0.07,
            "experience_years": 0.14,
            "hackathons": 0.06,
            "research_papers": 0.05,
            "skills_score": 0.12,
            "soft_skills_score": 0.06,
            "resume_length_words": 0.02,
            "education_level": 0.07,
            "university_tier": 0.08,
            "company_type": 0.07,
        }
        score = sum(components[key] * weights[key] for key in weights)
        return (score * 100).clip(0, 100).round(2)

    def _augment_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if SCORING_COLUMN not in dataframe.columns:
            dataframe[SCORING_COLUMN] = self._build_synthetic_score(dataframe)
        if SYNTHETIC_RECOMMENDATION_COLUMN not in dataframe.columns:
            dataframe[SYNTHETIC_RECOMMENDATION_COLUMN] = pd.cut(
                dataframe[SCORING_COLUMN],
                bins=[-1, 45, 65, 80, 101],
                labels=["reject", "maybe", "shortlist", "strong_hire"],
            ).astype(str)
        return dataframe

    def load_dataframe(self, force_refresh: bool = False) -> pd.DataFrame:
        if self._dataframe is not None and not force_refresh:
            return self._dataframe

        dataset_path = self._download_or_locate_dataset()
        dataframe = pd.read_csv(dataset_path)
        if dataframe.empty:
            raise ValueError("The dataset is empty.")
        dataframe = self._augment_dataframe(dataframe)
        self._dataframe = dataframe
        self._profile = None
        return self._dataframe

    def _detect_targets(self, dataframe: pd.DataFrame) -> tuple[str | None, str]:
        lowered = {column.lower(): column for column in dataframe.columns}
        classification_target = None
        for candidate in TARGET_CANDIDATES:
            if candidate in lowered:
                column = lowered[candidate]
                if dataframe[column].dropna().nunique() <= 12:
                    classification_target = column
                    break
        return classification_target, SCORING_COLUMN

    def _build_feature_schema(self, dataframe: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]) -> list[dict[str, Any]]:
        schema: list[dict[str, Any]] = []
        for column in numeric_columns:
            series = dataframe[column]
            schema.append(
                {
                    "name": column,
                    "type": "numeric",
                    "min": round_float(series.min(), 3),
                    "max": round_float(series.max(), 3),
                    "mean": round_float(series.mean(), 3),
                    "median": round_float(series.median(), 3),
                    "description": self._column_description(column),
                }
            )
        for column in categorical_columns:
            series = dataframe[column].astype(str)
            top_values = series.value_counts().head(10)
            schema.append(
                {
                    "name": column,
                    "type": "categorical",
                    "categories": [{"label": key, "count": int(value)} for key, value in top_values.items()],
                    "description": self._column_description(column),
                }
            )
        return schema

    def _column_description(self, column: str) -> str:
        descriptions = {
            "cgpa": "Academic score on a 10-point scale.",
            "internships": "Number of completed internships.",
            "projects": "Number of portfolio or coursework projects.",
            "programming_languages": "Count of programming languages listed.",
            "certifications": "Count of relevant certifications.",
            "experience_years": "Professional experience in years.",
            "hackathons": "Count of hackathon participations.",
            "research_papers": "Count of research publications.",
            "skills_score": "Technical skills strength score.",
            "soft_skills_score": "Soft-skills assessment score.",
            "resume_length_words": "Length of resume in words.",
            "education_level": "Highest education attained.",
            "university_tier": "Tier ranking of the university.",
            "company_type": "Preferred employer type.",
        }
        return descriptions.get(column, column.replace("_", " ").title())

    def get_profile(self) -> DatasetProfile:
        if self._profile is not None:
            return self._profile

        dataframe = self.load_dataframe()
        classification_target, regression_target = self._detect_targets(dataframe)
        ignored_columns = {"candidate_id", regression_target, SYNTHETIC_RECOMMENDATION_COLUMN}
        if classification_target:
            ignored_columns.add(classification_target)

        numeric_columns = [
            column
            for column in dataframe.select_dtypes(include=[np.number]).columns
            if column not in ignored_columns
        ]
        categorical_columns = [
            column
            for column in dataframe.select_dtypes(exclude=[np.number]).columns
            if column not in ignored_columns
        ]

        target_distribution = {}
        if classification_target:
            target_distribution = {
                str(label): round_float(value, 4)
                for label, value in dataframe[classification_target].value_counts(normalize=True).sort_index().items()
            }

        dataset_path = self._dataset_path or self._download_or_locate_dataset()
        info = {
            "dataset_path": str(dataset_path),
            "dataset_name": Path(dataset_path).name,
            "num_rows": int(len(dataframe)),
            "num_columns": int(len(dataframe.columns)),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "feature_types": {column: str(dtype) for column, dtype in dataframe.dtypes.astype(str).items()},
            "missing_values": {column: int(value) for column, value in dataframe.isna().sum().items()},
            "target_distribution": target_distribution,
            "summary_statistics": to_builtin(
                dataframe.describe(include="all").transpose().replace({np.nan: None}).to_dict(orient="index")
            ),
            "target_columns": {
                "classification": classification_target,
                "regression": regression_target,
            },
            "synthetic_target_active": classification_target is None,
            "synthetic_target_logic": (
                "Transparent weighted score derived from academic profile, projects, internships, skills, "
                "certifications, experience, soft skills, resume length, and institution/company profile."
            ),
        }
        available_tasks = ["regression"]
        if classification_target:
            available_tasks.insert(0, "classification")
        self._profile = DatasetProfile(
            info=info,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            classification_target=classification_target,
            regression_target=regression_target,
            available_tasks=available_tasks,
            feature_schema=self._build_feature_schema(dataframe, numeric_columns, categorical_columns),
        )
        return self._profile

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        profile = self.get_profile()
        dataframe = self.load_dataframe()
        numeric_summary = dataframe[profile.numeric_columns].agg(["mean", "median"]).round(2).to_dict()
        return {
            "dataset": profile.info,
            "feature_schema": profile.feature_schema,
            "overview": {
                "candidate_count": profile.info["num_rows"],
                "feature_count": len(profile.numeric_columns) + len(profile.categorical_columns),
                "available_tasks": profile.available_tasks,
                "classification_target": profile.classification_target,
                "regression_target": profile.regression_target,
                "numeric_summary": to_builtin(numeric_summary),
            },
        }

    def get_preview(
        self,
        page: int = 1,
        page_size: int | None = None,
        search: str | None = None,
        sort_by: str | None = None,
        sort_dir: str = "desc",
    ) -> dict[str, Any]:
        dataframe = self.load_dataframe()
        preview = dataframe.copy()

        if search:
            lowered = search.lower()
            search_columns = ["candidate_id"] + [column for column in preview.columns if preview[column].dtype == "object"]
            mask = pd.Series(False, index=preview.index)
            for column in search_columns:
                mask = mask | preview[column].astype(str).str.lower().str.contains(lowered, na=False)
            preview = preview.loc[mask]

        if sort_by and sort_by in preview.columns:
            preview = preview.sort_values(sort_by, ascending=sort_dir.lower() == "asc")

        page_size = page_size or self.settings.preview_page_size
        start = max(page - 1, 0) * page_size
        end = start + page_size
        page_frame = preview.iloc[start:end]
        rows = page_frame.replace({np.nan: None}).to_dict(orient="records")
        return {
            "total_rows": int(len(preview)),
            "page": page,
            "page_size": page_size,
            "rows": to_builtin(rows),
            "columns": list(preview.columns),
        }

    def get_candidate_record(self, candidate_id: int) -> dict[str, Any]:
        dataframe = self.load_dataframe()
        if "candidate_id" not in dataframe.columns:
            raise ValueError("candidate_id column not found in dataset.")
        rows = dataframe.loc[dataframe["candidate_id"] == candidate_id]
        if rows.empty:
            raise KeyError(f"Candidate {candidate_id} was not found.")
        return to_builtin(rows.iloc[0].replace({np.nan: None}).to_dict())
