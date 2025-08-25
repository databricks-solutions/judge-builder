"""Custom prompt-based judge implementation using MLflow scorers."""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import mlflow

from server.judges.base_judge import BaseJudge
from server.prompts import DEFAULT_JUDGE_PROMPT_TEMPLATE
from server.utils.naming_utils import create_scorer_name, sanitize_judge_name

if TYPE_CHECKING:
    from server.optimizers.custom_prompt_optimizer import CustomPromptOptimizer

logger = logging.getLogger(__name__)

# Minimum number of labeled examples required for optimization
MIN_EXAMPLES_FOR_OPTIMIZATION = 10


class CustomPromptJudge(BaseJudge):
    """Judge implementation using custom MLflow prompt-based scorer."""

    def __init__(
        self,
        name: str,
        user_instructions: str,
        experiment_id: Optional[str] = None,
        optimizer: Optional['CustomPromptOptimizer'] = None,
    ):
        """Initialize CustomPromptJudge."""
        super().__init__(name, user_instructions, experiment_id)

        # Store optimizer as class attribute
        self.optimizer = optimizer

        # Use shared prompt template with system instructions
        self.prompt_template = DEFAULT_JUDGE_PROMPT_TEMPLATE.format(
            system_instructions=self.system_instructions
        )

    def _create_scorer(self) -> Callable:
        """Create the actual custom prompt judge scorer."""
        return self._get_scorer_function()

    def _get_scorer_function(self) -> Callable:
        """Get the shared scorer function implementation."""

        def custom_prompt_judge_scorer(inputs, outputs, trace=None):
            """Custom prompt judge scorer with direct model calling."""
            import json
            import traceback

            try:
                from mlflow.entities import AssessmentError, AssessmentSource, Feedback
                from mlflow.genai.judges import custom_prompt_judge

                # Create the prompt-based judge
                prompt_judge = custom_prompt_judge(
                    name=sanitize_judge_name(self.name),
                    prompt_template=self.prompt_template,
                )

                try:
                    if isinstance(inputs, str):
                        inputs = json.loads(inputs)
                    request = str(inputs.get('request', inputs.get('inputs', inputs)))
                except Exception:
                    request = str(inputs)

                try:
                    if isinstance(outputs, str):
                        outputs = json.loads(outputs)
                    response = str(outputs.get('response', outputs.get('inputs', outputs)))
                except Exception:
                    response = str(outputs)

                # Use the prompt judge to evaluate
                feedback_obj = prompt_judge(request=request, response=response)
                metadata = feedback_obj.metadata or {}
                metadata['version'] = str(self.version)
                feedback_obj.metadata = metadata
                return feedback_obj

            except Exception as e:
                logger.error(f'CustomPromptJudge: Evaluation failed: {str(e)}')
                logger.error(traceback.format_exc())

                # Try to import MLflow classes for error feedback
                try:
                    from mlflow.entities import AssessmentError, AssessmentSource, Feedback

                    return Feedback(
                        name=sanitize_judge_name(self.name),
                        source=AssessmentSource(
                            source_type='LLM_JUDGE', source_id='custom_prompt_judge'
                        ),
                        error=AssessmentError(
                            error_code='EVALUATION_ERROR',
                            error_message=f'Evaluation failed: {str(e)}',
                            stack_trace=traceback.format_exc(),
                        ),
                    )
                except ImportError:
                    # If MLflow imports fail, return a simple error dict
                    return {
                        'error': f'Evaluation failed: {str(e)}',
                        'judge_name': sanitize_judge_name(self.name),
                    }

        return custom_prompt_judge_scorer

    def evaluate(self, inputs: Dict[str, Any], outputs: Dict[str, Any], trace=None):
        """Evaluate using the custom scorer."""
        result = self.scorer_func(inputs, outputs, trace)
        return result

    def register_scorer(self) -> Any:
        """Register the CustomPromptJudge as an MLflow scorer using temporary file approach."""
        try:
            import os
            import tempfile
            from importlib import util

            from mlflow.genai.scorers import scorer

            judge_name = sanitize_judge_name(self.name)
            scorer_name = create_scorer_name(self.name, self.version)

            # Use the prompt template as class attribute
            escaped_prompt_template = repr(self.prompt_template)
            escaped_judge_name = repr(judge_name)

            # Create the scorer function as a complete module - reuse the same logic as _get_scorer_function
            module_code = f"""
def custom_prompt_judge_scorer(inputs, outputs, trace=None):
    '''Custom prompt judge scorer with direct model calling.'''
    import json
    import traceback
    from mlflow.genai.judges import custom_prompt_judge

    # Hardcoded values that will survive serialization
    prompt_template = {escaped_prompt_template}
    judge_name = {escaped_judge_name}

    try:
        # Create the prompt-based judge
        prompt_judge = custom_prompt_judge(
            name=judge_name,
            prompt_template=prompt_template,
        )

        try:
            if isinstance(inputs, str):
                inputs = json.loads(inputs)
            request = str(inputs.get('request', inputs.get("inputs", inputs)))
        except Exception:
            request = str(inputs)

        try:
            if isinstance(outputs, str):
                outputs = json.loads(outputs)
            response = str(outputs.get('response', outputs.get("inputs", outputs)))
        except Exception:
            response = str(outputs)

        # Use the prompt judge to evaluate
        feedback_obj = prompt_judge(request=request, response=response)
        metadata = feedback_obj.metadata or {{}}
        metadata['version'] = '{self.version}'
        feedback_obj.metadata = metadata
        return feedback_obj

    except Exception as e:
        from mlflow.entities import AssessmentError, Feedback, AssessmentSource

        return Feedback(
            name=judge_name,
            source=AssessmentSource(
                source_type='LLM_JUDGE',
                source_id='custom_prompt_judge'
            ),
            error=AssessmentError(
                error_code='EVALUATION_ERROR',
                error_message=f'Evaluation failed: {{str(e)}}',
                stack_trace=traceback.format_exc(),
            ),
        )
"""

            # Create a temporary file and load the function as a module
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(module_code)
                tmp.flush()

                # Load the module from the temporary file
                spec = util.spec_from_file_location('custom_prompt_scorer_module', tmp.name)
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Get the scorer function from the module
                scorer_func = module.custom_prompt_judge_scorer

                # Apply decorator and register
                decorated = scorer(scorer_func)
                # Only pass experiment_id if it's set to avoid using deleted experiments
                if self.experiment_id:
                    scorer_func = decorated.register(
                        name=scorer_name, experiment_id=self.experiment_id
                    )
                else:
                    # Register without experiment_id to let MLflow handle it
                    scorer_func = decorated.register(name=scorer_name)

                # Clean up the temporary file
                os.unlink(tmp.name)

            return scorer_func

        except Exception as e:
            logger.warning(f'Failed to register CustomPromptJudge scorer: {e}')
            return create_scorer_name(self.name, self.version)

    def optimize(self, traces: List[mlflow.entities.Trace]) -> bool:
        """Optimize the judge using labeled traces."""
        logger.info(
            f'Starting optimization for judge {self.name} with {len(traces)} traces'
        )

        if len(traces) < MIN_EXAMPLES_FOR_OPTIMIZATION:
            logger.warning(f'Need at least {MIN_EXAMPLES_FOR_OPTIMIZATION} traces for optimization')
            return False

        if not self.optimizer:
            logger.warning('No optimizer provided to judge')
            return False

        try:
            # Split traces into train/eval (50/50 split)
            split_idx = int(len(traces) * 0.5)
            train_data = traces[:split_idx]
            eval_data = traces[split_idx:] if split_idx < len(traces) else []

            # Run optimization
            result = self.optimizer.run_optimization(
                judge_name=self.name,
                prompt=self.prompt_template,
                train_data=train_data,
                eval_data=eval_data,
            )

            # Update the judge's prompt template with optimized version
            self.prompt_template = result.optimized_prompt
            logger.info(f'Successfully optimized judge {self.name} using {result.optimizer_name}')
            return True

        except Exception as e:
            logger.error(f'Optimization failed for judge {self.name}: {e}')
            return False
