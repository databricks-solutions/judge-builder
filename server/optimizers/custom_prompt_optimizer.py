"""Custom prompt optimizer using DSPy for judge optimization."""

import logging
from typing import Any, List, Optional

import dspy
import mlflow

from server.prompts import OPTIMIZED_JUDGE_PROMPT_TEMPLATE
from server.utils.naming_utils import sanitize_judge_name
from server.utils.parsing_utils import extract_request_from_trace, extract_response_from_trace

logger = logging.getLogger(__name__)


class OptimizerOutput:
    """Output from optimizer containing the optimized prompt and metadata."""

    def __init__(
        self,
        optimized_prompt: str,
        optimizer_name: str,
        optimized_instructions: Optional[str] = None,
    ):
        self.optimized_prompt = optimized_prompt
        self.optimizer_name = optimizer_name
        self.optimized_instructions = optimized_instructions or optimized_prompt


class DSPyPromptOptimizer:
    """Base class for DSPy-based prompt optimizers."""

    def run_optimization(
        self,
        prompt: str,
        train_data: List[Any],
        eval_data: List[Any] = None,
    ) -> OptimizerOutput:
        """Run optimization process.

        Args:
            prompt: Original prompt to optimize
            train_data: Training data
            eval_data: Evaluation data (optional)

        Returns:
            OptimizerOutput with optimized prompt and metadata
        """
        raise NotImplementedError('Subclasses must implement run_optimization')


class CustomPromptOptimizer(DSPyPromptOptimizer):
    """Custom optimizer supporting multiple DSPy algorithms for judge prompt optimization."""

    SUPPORTED_OPTIMIZERS = ['miprov2', 'simba']

    def __init__(self, optimizer_algorithm: str = 'miprov2'):
        """Initialize the optimizer with the specified algorithm.

        Args:
            optimizer_algorithm: The optimization algorithm to use.
        """
        if optimizer_algorithm.lower() not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer algorithm '{optimizer_algorithm}'. "
                f'Supported algorithms: {self.SUPPORTED_OPTIMIZERS}'
            )
        self.optimizer_algorithm = optimizer_algorithm.lower()

    def _extract_instructions(self, prompt: str) -> str:
        """Extract core instructions from the prompt template."""
        # Remove template variables and extract the core instruction
        lines = prompt.strip().split('\n')
        instructions = []

        for line in lines:
            # Skip template variable lines and format instructions
            if (
                not line.strip().startswith('<')
                and not line.strip().startswith('[[')
                and '{' not in line
                and line.strip()
                and 'You must choose' not in line
            ):
                instructions.append(line.strip())

        return ' '.join(instructions)

    def _trace_to_dspy_example(self, trace: mlflow.entities.Trace, judge_name: str) -> Optional[dspy.Example]:
        """Convert a trace to a DSPy example.

        Args:
            trace: MLflow trace object

        Returns:
            DSPy example with inputs (inputs, outputs) and outputs (result, rationale)
        """
        try:
            # Extract request and response from trace
            request = extract_request_from_trace(trace)
            response = extract_response_from_trace(trace)

            expected_result = next(
                (
                    assessment
                    for assessment in trace.info.assessments
                    if assessment.name == judge_name and assessment.source.source_type == 'HUMAN'
                ),
                None,
            )
            if not expected_result:
                return None

            if not expected_result.feedback:
                logger.warning(f'No feedback found in assessment for trace {trace.info.trace_id}')
                return None

            # Create DSPy example
            example = dspy.Example(
                inputs=request,
                outputs=response,
                result=expected_result.feedback.value.lower(),
                rationale=expected_result.rationale if expected_result.rationale else '',
            )

            # Set inputs (what the model should use as input)
            example = example.with_inputs('inputs', 'outputs')
            return example

        except Exception as e:
            import traceback
            logger.error(f'Failed to create DSPy example from trace: {e}')
            logger.error(f'Full stack trace:\n{traceback.format_exc()}')
            return None

    def _create_optimizer(self, agreement_metric):
        """Create the appropriate optimizer based on the selected algorithm."""
        if self.optimizer_algorithm == 'miprov2':
            return dspy.MIPROv2(
                metric=agreement_metric,
                init_temperature=1.0,
                auto='light',  # Use auto mode to avoid parameter conflicts
            )
        elif self.optimizer_algorithm == 'simba':
            return dspy.SIMBA(metric=agreement_metric, bsize=4)
        else:
            raise ValueError(f'Unsupported optimizer algorithm: {self.optimizer_algorithm}')

    def run_optimization(
        self,
        judge_name: str,
        prompt: str,
        train_data: List[mlflow.entities.Trace],
        eval_data: List[mlflow.entities.Trace] = None,
    ) -> OptimizerOutput:
        """Run DSPy MIPROv2 optimization.

        Args:
            judge_name: Name of the judge to optimize
            prompt: Original prompt to optimize
            train_data: Training data
            eval_data: Evaluation data (optional)

        Returns:
            OptimizerOutput with optimized prompt using MIPROv2
        """
        try:
            judge_name = sanitize_judge_name(judge_name)

            # Extract instructions from prompt
            instructions = self._extract_instructions(prompt)
            logger.info(f'Extracted instructions: {instructions}')

            # Create DSPy signature following MLflow pattern
            signature = dspy.make_signature(
                {
                    'inputs': (str, dspy.InputField(desc='Inputs to the model')),
                    'outputs': (str, dspy.InputField(desc='Outputs from the model')),
                    'result': (
                        str,
                        dspy.OutputField(desc='Pass or fail based on the inputs and outputs'),
                    ),
                    'rationale': (
                        str,
                        dspy.OutputField(desc='Rationale explaining the pass or fail result'),
                    ),
                },
                instructions,
            )

            # Create DSPy program
            program = dspy.ChainOfThought(signature)
            logger.info('Created DSPy program with signature')

            # Convert data to DSPy format
            dspy_train_data = [
                self._trace_to_dspy_example(trace, judge_name) for trace in train_data
            ]
            dspy_eval_data = [self._trace_to_dspy_example(trace, judge_name) for trace in eval_data]

            # Filter out None examples
            dspy_train_data = [ex for ex in dspy_train_data if ex is not None]
            dspy_eval_data = [ex for ex in dspy_eval_data if ex is not None]

            logger.info(f'Created {len(dspy_train_data)} training examples and {len(dspy_eval_data)} eval examples')

            if not dspy_train_data:
                raise ValueError('No valid training examples could be created from traces')
            if not dspy_eval_data:
                raise ValueError('No valid evaluation examples could be created from traces')

            # Create simple agreement metric for judge optimization
            def agreement_metric(example, pred, trace=None):
                """Simple agreement metric for judge optimization."""
                try:
                    # Normalize both example and pred to consistent format
                    expected_norm = str(example.result).lower().strip()
                    predicted_norm = str(pred.result).lower().strip()

                    agreement = 1.0 if expected_norm == predicted_norm else 0.0

                    return agreement
                except Exception as e:
                    logger.error(f'Metric evaluation failed: {e}')
                    return 0.0

            # Create the appropriate optimizer based on selected algorithm
            optimizer = self._create_optimizer(agreement_metric)
            optimizer_name = type(optimizer).__name__

            logger.info(f'Starting {optimizer_name} compilation...')

            # Compile the optimized program with algorithm-specific parameters
            if self.optimizer_algorithm == 'miprov2':
                optimized_program = optimizer.compile(
                    student=program,
                    trainset=dspy_train_data,
                    valset=dspy_eval_data,
                    requires_permission_to_run=False,
                )
            elif self.optimizer_algorithm == 'simba':
                # SIMBA only accepts student and trainset parameters
                optimized_program = optimizer.compile(
                    student=program,
                    trainset=dspy_train_data,
                    seed=42,  # Fixed seed for reproducibility
                )
            else:
                raise ValueError(f'Unsupported optimizer algorithm: {self.optimizer_algorithm}')

            logger.info(f'{optimizer_name} compilation completed')

            # Extract the optimized instructions and format using shared template
            optimized_instructions = optimized_program.predict.signature.instructions

            optimized_prompt = OPTIMIZED_JUDGE_PROMPT_TEMPLATE.format(
                system_instructions=optimized_instructions
            )

            logger.info(f'DSPy {optimizer_name} optimization completed successfully')

            return OptimizerOutput(
                optimized_prompt=optimized_prompt.strip(),
                optimizer_name=optimizer_name,
                optimized_instructions=optimized_instructions.strip(),
            )

        except ImportError as e:
            logger.error('DSPy not available for optimization: %s', e)
            raise ValueError(
                'DSPy is required for CustomPromptOptimizer but is not installed'
            ) from e
        except Exception as e:
            import traceback
            logger.error('DSPy optimization failed: %s', e)
            logger.error('Full stack trace:\n%s', traceback.format_exc())
            # No fallback - raise the actual error to see what's wrong
            raise RuntimeError(f'DSPy {optimizer_name} optimization failed: {str(e)}') from e
