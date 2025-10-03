"""Custom SIMBA alignment optimizer with configurable model."""

import logging
from typing import Optional

from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer

logger = logging.getLogger(__name__)


class CustomSIMBAAlignmentOptimizer(SIMBAAlignmentOptimizer):
    """SIMBA optimizer that accepts custom model for alignment.

    This extends MLflow's SIMBAAlignmentOptimizer to support using
    different models for alignment than the default judge model.
    """

    def __init__(self, model: Optional[str] = None, **kwargs):
        """Initialize optimizer with optional custom model.

        Args:
            model: Model identifier (e.g., 'databricks:/my-endpoint')
                   If None, uses default model from parent class.
            **kwargs: Additional arguments passed to parent class.
        """
        if model:
            logger.info(f"Initializing custom SIMBA optimizer with model: {model}")

        super().__init__(model=model, **kwargs)
