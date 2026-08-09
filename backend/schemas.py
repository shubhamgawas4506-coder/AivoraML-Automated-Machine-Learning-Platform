"""Pydantic schemas for AivoraML API request/response models."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
from enum import Enum


class TaskType(str, Enum):
    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TrainingApproach(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    semantic_type: str
    n_missing: int
    missing_pct: float
    n_unique: int
    sample_values: List[Any] = Field(default_factory=list)
    stats: Optional[Dict[str, Any]] = None
    top_values: Optional[Dict[str, Any]] = None


class TargetSuggestion(BaseModel):
    column: str
    task_type: str
    reason: str
    score: float


class QualityWarning(BaseModel):
    type: str
    severity: str
    message: str


class DatasetProfile(BaseModel):
    filename: str = ""
    n_rows: int
    n_cols: int
    total_missing: int
    n_duplicates: int
    columns: List[ColumnInfo]
    preview: List[Dict[str, Any]]
    numeric_summary: Optional[Dict[str, Any]] = None
    target_suggestions: List[TargetSuggestion] = Field(default_factory=list)
    warnings: List[QualityWarning] = Field(default_factory=list)


class WorkflowRequest(BaseModel):
    target_column: str
    task_type: TaskType = TaskType.AUTO
    stages: List[str] = Field(
        default_factory=lambda: [
            "cleaning", "eda", "feature_engineering",
            "feature_review", "model_selection", "evaluation"
        ]
    )


class PredictionRequest(BaseModel):
    values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Feature values keyed by the original dataset column name.",
    )


class TrainingJobRequest(BaseModel):
    """Validated configuration for synchronous or background training."""

    target_column: str = Field(min_length=1, max_length=500)
    approach: TrainingApproach = TrainingApproach.BALANCED
    include_eda: bool = False


class AssistantTurn(BaseModel):
    """One bounded conversation turn supplied by the local browser."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class AssistantChatRequest(BaseModel):
    """Validated project-assistant request; raw dataset rows are never accepted."""

    message: str = Field(min_length=1, max_length=4_000)
    view: str = Field(default="workspace", max_length=80)
    strategy: TrainingApproach = TrainingApproach.BALANCED
    target_column: Optional[str] = Field(default=None, max_length=500)
    history: List[AssistantTurn] = Field(default_factory=list, max_length=8)


class ModelResult(BaseModel):
    name: str
    metrics: Dict[str, float]
    is_best: bool = False


class WorkflowResult(BaseModel):
    task_type: str
    target_column: str
    stages_completed: List[str]
    cleaning_actions: List[str]
    eda_insights: List[Dict[str, Any]] = Field(default_factory=list)
    feature_info: Dict[str, Any] = Field(default_factory=dict)
    models_compared: List[ModelResult]
    best_model: str
    metrics: Dict[str, float]
    feature_importance: List[Dict[str, Any]] = Field(default_factory=list)
    split_info: Dict[str, Any]
    tuning_result: Optional[Dict[str, Any]] = None
    decision_record: List[str]
    warnings: List[str] = Field(default_factory=list)
