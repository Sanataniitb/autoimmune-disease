"""Machine-learning classifiers for autoimmune disease diagnosis from RNA-seq expression."""

__version__ = "0.2.0"

from .config import DiseaseConfig
from .data import load_expression, make_synthetic
from .evaluate import classification_metrics, cross_validate_model, fit_final_model
from .models import MODEL_REGISTRY, build_pipeline

__all__ = [
    "DiseaseConfig",
    "MODEL_REGISTRY",
    "build_pipeline",
    "classification_metrics",
    "cross_validate_model",
    "fit_final_model",
    "load_expression",
    "make_synthetic",
]
