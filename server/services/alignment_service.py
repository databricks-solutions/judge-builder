"""Alignment service for judge evaluation and alignment using DSPy."""

import logging
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.genai import evaluate, scorers
import dspy

from server.models import (
    AlignmentComparison,
    AlignmentMetrics,
    AlignmentResponse,
    ConfusionMatrix,
    EvaluationResult,
    JudgeResponse,
    SingleJudgeTestRequest,
    SingleJudgeTestResponse,
    TraceRequest,
)
from server.utils.constants import ALIGNED_SAMPLES_COUNT
from server.utils.naming_utils import create_scorer_name, sanitize_judge_name
from server.utils.parsing_utils import (
    assessment_has_error,
    get_human_feedback_from_trace,
    get_scorer_feedback_from_trace,
)
from server.utils import dspy_utils

from .base_service import BaseService
from .cache_service import cache_service

logger = logging.getLogger(__name__)


class AlignmentService(BaseService):
    """Handles judge evaluation and alignment using DSPy."""

    def __init__(self):
        super().__init__()
        # Set up DSPy language model
        lm = dspy_utils.AgentEvalLM()
        # Configure DSPy to use this language model
        dspy.configure(lm=lm)

    def _get_judge_scorer(self, judge: JudgeResponse) -> Optional[scorers.Scorer]:
        """Get the scorer for a judge."""
        scorers_list = scorers.list_scorers()
        if not scorers_list:
            logger.warning('No scorers found in list_scorers()')
            return None

        scorer_name = create_scorer_name(judge.name, judge.version)

        for scorer in scorers_list:
            if scorer.name == scorer_name:
                return scorer

        logger.warning(f'Scorer "{scorer_name}" not found among available scorers')
        return None

    # Judge evaluation and testing
    def evaluate_judge(self, judge_id: str, request: TraceRequest) -> EvaluationResult:
        """Run judge evaluation on traces and log to MLflow."""
        from server.services.judge_service import judge_service

        try:
            # Get judge details
            judge = judge_service.get_judge(judge_id)
            if not judge:
                raise ValueError(f'Judge {judge_id} not found')

            # Check if evaluation is cached
            cached_run_id = cache_service.get_evaluation_run_id(
                judge_id, judge.version, request.trace_ids, judge.experiment_id
            )
            if cached_run_id:
                logger.info(f'Using cached evaluation run {cached_run_id} for judge {judge_id}')
                return EvaluationResult(
                    judge_id=judge_id,
                    judge_version=judge.version,
                    mlflow_run_id=cached_run_id,
                    evaluation_results=[],  # TODO: Implement individual trace results
                    total_traces=len(request.trace_ids),
                )

            # Set experiment context
            mlflow.set_experiment(experiment_id=judge.experiment_id)

            # Find judge scorer
            judge_scorer = self._get_judge_scorer(judge)
            if not judge_scorer:
                raise ValueError(f'Scorer for judge {judge.name} not found')

            logger.info(f'Found scorer: {judge_scorer.name} for judge {judge.name} version {judge.version}')

            # Get traces using cache
            traces = []
            for trace_id in request.trace_ids:
                trace = cache_service.get_trace(trace_id)
                if trace:
                    traces.append(trace)
                else:
                    logger.warning(f'Could not fetch trace {trace_id}')

            if not traces:
                raise ValueError('No valid traces found')

            eval_data = [{'trace': trace} for trace in traces]

            # Run evaluation
            sanitized_name = sanitize_judge_name(judge.name)
            with mlflow.start_run(run_name=f'evaluation_{sanitized_name}_v{judge.version}') as run:
                mlflow.set_tag('judge_id', judge_id)
                mlflow.set_tag('judge_version', judge.version)
                mlflow.set_tag(
                    'dataset_version', cache_service.compute_dataset_version(request.trace_ids)
                )

                evaluate(data=eval_data, scorers=[judge_scorer])

                # Cache the evaluation result
                cache_service.cache_evaluation_run_id(
                    judge_id, judge.version, request.trace_ids, run.info.run_id
                )

                return EvaluationResult(
                    judge_id=judge_id,
                    judge_version=judge.version,
                    mlflow_run_id=run.info.run_id,
                    evaluation_results=[],  # TODO: Implement individual trace results
                    total_traces=len(eval_data),
                )

        except Exception as e:
            logger.error(f'Failed to evaluate judge {judge_id}: {e}')
            return EvaluationResult(
                judge_id=judge_id,
                judge_version=0,
                mlflow_run_id='',
                evaluation_results=[],
                total_traces=0,
            )

    def test_judge(self, judge_id: str, request: SingleJudgeTestRequest) -> SingleJudgeTestResponse:
        """Test judge on a single trace (for play buttons)."""
        from server.services.judge_service import judge_service

        try:
            # Get judge details
            judge = judge_service.get_judge(judge_id)
            if not judge:
                raise ValueError(f'Judge {judge_id} not found')

            # Get trace using cache
            trace = cache_service.get_trace(request.trace_id)
            if not trace:
                raise ValueError(f'Trace {request.trace_id} not found')

            # Find judge scorer
            judge_scorer = self._get_judge_scorer(judge)
            if not judge_scorer:
                raise ValueError(f'Scorer for judge {judge.name} not found')

            # Run scorer on trace
            feedback = judge_scorer(
                inputs=trace.data.request, outputs=trace.data.response, trace=trace
            )

            return SingleJudgeTestResponse(
                judge_id=judge_id,
                judge_version=judge.version,
                trace_id=request.trace_id,
                feedback=feedback,
            )

        except Exception as e:
            logger.error(f'Failed to test judge {judge_id}: {e}')
            raise

    # Alignment workflow
    def get_alignment_comparison(self, judge_id: str) -> Dict:
        """Get alignment comparison data including metrics and confusion matrix."""
        from server.services.judge_service import judge_service
        from server.services.labeling_service import labeling_service

        # Get current judge
        judge = judge_service.get_judge(judge_id)
        if not judge or judge.version < 2:
            raise ValueError(f'Judge {judge_id} must have version >= 2 for alignment comparison')

        # Get traces from labeling session
        examples = labeling_service.get_examples(judge_id)
        trace_ids = [ex.trace_id for ex in examples]
        if not trace_ids:
            raise ValueError('No traces found for alignment comparison')

        # Count examples with human feedback from assessments
        examples_with_feedback = []
        for ex in examples:
            # Get the actual trace object from cache to access assessments
            trace = cache_service.get_trace(ex.trace_id)
            if not trace:
                logger.warning(f'Trace {ex.trace_id} not found in cache for judge {judge_id}')
                continue

            human_feedback = get_human_feedback_from_trace(judge.name, trace)
            if human_feedback:
                examples_with_feedback.append((ex, human_feedback))

        # Get evaluation run IDs for both versions (cache will automatically search MLflow if needed)
        prev_run_id = cache_service.get_evaluation_run_id(judge_id, judge.version - 1, trace_ids, judge.experiment_id)
        curr_run_id = cache_service.get_evaluation_run_id(judge_id, judge.version, trace_ids, judge.experiment_id)
        
        if not prev_run_id or not curr_run_id:
            raise ValueError('Evaluation runs not found. Please run alignment first.')

        # Build per-row comparisons using trace_id matching
        comparisons = []
        human_labels = []

        for example, human_feedback in examples_with_feedback:
            trace = cache_service.get_trace(example.trace_id)
            if not trace:
                logger.warning(f'Skipping trace {example.trace_id}: trace not found in cache')
                continue

            # Get judge feedback for both versions using the utility functions
            prev_feedback = get_scorer_feedback_from_trace(judge.name, judge.version - 1, trace)
            curr_feedback = get_scorer_feedback_from_trace(judge.name, judge.version, trace)

            if not prev_feedback:
                logger.warning(f'Skipping trace {example.trace_id}: missing previous judge feedback (v{judge.version - 1})')
                continue
                
            if not curr_feedback:
                logger.warning(f'Skipping trace {example.trace_id}: missing current judge feedback (v{judge.version})')
                continue

            # Skip assessments with errors
            has_human_error = assessment_has_error(human_feedback)
            has_prev_error = assessment_has_error(prev_feedback)
            has_curr_error = assessment_has_error(curr_feedback)
            
            if has_human_error or has_prev_error or has_curr_error:
                logger.warning(f'Skipping trace {example.trace_id}: has errors (human={has_human_error}, prev={has_prev_error}, curr={has_curr_error})')
                continue
            
            human_labels.append(human_feedback.feedback.value)
            comparisons.append(AlignmentComparison(
                trace_id=example.trace_id,
                request=trace.data.request,
                response=trace.data.response,
                human_feedback=human_feedback,
                previous_judge_feedback=prev_feedback,
                new_judge_feedback=curr_feedback
            ))
        
        if not human_labels:
            raise ValueError('No valid examples with both human and judge feedback found')

        # Calculate metrics using only valid examples
        prev_judge_labels = [comp.previous_judge_feedback.feedback.value for comp in comparisons]
        curr_judge_labels = [comp.new_judge_feedback.feedback.value for comp in comparisons]

        metrics = AlignmentMetrics(
            total_samples=len(human_labels),
            previous_agreement_count=sum(1 for h, p in zip(human_labels, prev_judge_labels) if h.lower() == p.lower()),
            new_agreement_count=sum(1 for h, c in zip(human_labels, curr_judge_labels) if h.lower() == c.lower()),
            confusion_matrix_previous=self.calculate_confusion_matrix(human_labels, prev_judge_labels),
            confusion_matrix_new=self.calculate_confusion_matrix(human_labels, curr_judge_labels)
        )

        return {'metrics': metrics, 'comparisons': comparisons}

    def run_alignment(self, judge_id: str) -> AlignmentResponse:
        """Run DSPy-powered judge alignment and create new version."""
        from server.services.judge_service import judge_service

        # Get current judge
        current_judge = judge_service.get_judge(judge_id)
        if not current_judge:
            raise ValueError(f'Judge {judge_id} not found')

        # Get traces from the labeling service examples
        from server.services.labeling_service import labeling_service
        logger.info(f'Getting examples from judge {judge_id}')
        examples = labeling_service.get_examples(judge_id)

        # Get actual traces using trace_ids from examples
        traces = []
        for example in examples:
            trace = cache_service.get_trace(example.trace_id)
            if trace:
                traces.append(trace)
            else:
                logger.warning(f'Could not fetch trace {example.trace_id} from cache')

        if not traces:
            raise ValueError('No traces found in labeling session')

        # Check labeling progress to ensure we have enough examples for alignment
        from server.judges.custom_prompt_judge import MIN_EXAMPLES_FOR_OPTIMIZATION
        
        labeling_progress = labeling_service.get_labeling_progress(judge_id)
        
        if labeling_progress.labeled_examples < MIN_EXAMPLES_FOR_OPTIMIZATION:
            raise ValueError(f'Insufficient labeled examples for alignment. Found {labeling_progress.labeled_examples} labeled examples, but need at least {MIN_EXAMPLES_FOR_OPTIMIZATION}. Please complete more labeling tasks before running alignment.')

        # Extract trace IDs for evaluation
        trace_ids = [trace.info.trace_id for trace in traces]

        # Step 1: Run evaluation on current judge version (v_i)
        logger.info(f'Running evaluation on judge {judge_id} version {current_judge.version}')
        self.evaluate_judge(judge_id, TraceRequest(trace_ids=trace_ids))
        
        # Invalidate trace cache after evaluation to get fresh judge feedback
        logger.info(f'Invalidating trace cache for {len(trace_ids)} traces after evaluation')
        cache_service.invalidate_traces(trace_ids)

        # Get fresh traces with updated judge feedback for optimization
        fresh_traces = cache_service.get_traces(trace_ids)
        logger.info(f'Retrieved {len(fresh_traces)} fresh traces for optimization')

        # Step 2: Run optimization on the judge
        logger.info(f'Starting optimization for judge {judge_id}')
        judge_instance = judge_service._judges[judge_id]
        optimization_success = judge_instance.optimize(fresh_traces)
        
        # Check if optimization failed and fail early
        if not optimization_success:
            logger.error(f'Optimization failed for judge {judge_id}')
            raise RuntimeError(f'Judge optimization failed. Please check the app logs for details.')
        
        # Step 3: Create new judge version (v_i+1) with optimized prompt
        optimized_instructions = judge_instance.prompt_template.replace(
            'Evaluation criteria: ', ''
        ).split('<request>')[0].strip()
        
        logger.info(f'Creating new version for judge {judge_id} with optimized instructions: {optimized_instructions}')
        new_judge = judge_service.create_new_version(judge_id, optimized_instructions)

        # Step 4: Run evaluation on new judge version (v_i+1)
        logger.info(f'Running evaluation on new judge version {new_judge.version}')
        new_eval_result = self.evaluate_judge(new_judge.id, TraceRequest(trace_ids=trace_ids))
        
        # Invalidate trace cache after second evaluation to get fresh judge feedback
        logger.info(f'Invalidating trace cache for {len(trace_ids)} traces after new version evaluation')
        cache_service.invalidate_traces(trace_ids)

        # Step 5: Use labeling service count for aligned samples count
        from server.services.labeling_service import labeling_service
        
        labeling_progress = labeling_service.get_labeling_progress(judge_id)
        aligned_samples_count = labeling_progress.labeled_examples
        
        logger.info(f'Found {aligned_samples_count} traces with valid human feedback out of {len(traces)} total traces')

        # Step 6: Tag the existing labeling run with alignment info
        client = MlflowClient()
        client.set_tag(current_judge.labeling_run_id, ALIGNED_SAMPLES_COUNT, str(aligned_samples_count))
        logger.info(f'Tagged labeling run {current_judge.labeling_run_id} with aligned samples count: {aligned_samples_count}')

        return AlignmentResponse(
            judge_id=new_judge.id,
            success=True,
            message=f'Successfully optimized judge from version {current_judge.version} to {new_judge.version} using {aligned_samples_count} aligned samples',
            new_version=new_judge.version,
            improvement_metrics=None,
        )

    def calculate_confusion_matrix(
        self, human_labels: list, judge_results: list
    ) -> ConfusionMatrix:
        """Calculate confusion matrix from human labels and judge results.

        Args:
            human_labels: List of human labels ('pass'/'fail')
            judge_results: List of judge results ('pass'/'fail')

        Returns:
            ConfusionMatrix object with calculated metrics
        """
        if len(human_labels) != len(judge_results):
            raise ValueError('Human labels and judge results must have the same length')

        true_positive = 0  # Judge Pass & Human Pass
        false_negative = 0  # Judge Fail & Human Pass
        false_positive = 0  # Judge Pass & Human Fail
        true_negative = 0  # Judge Fail & Human Fail

        for human, judge in zip(human_labels, judge_results):
            # Normalize to pass/fail
            human_pass = str(human).lower() == 'pass'
            judge_pass = str(judge).lower() == 'pass'

            if human_pass and judge_pass:
                true_positive += 1
            elif human_pass and not judge_pass:
                false_negative += 1
            elif not human_pass and judge_pass:
                false_positive += 1
            elif not human_pass and not judge_pass:
                true_negative += 1

        return ConfusionMatrix(
            true_positive=true_positive,
            false_negative=false_negative,
            false_positive=false_positive,
            true_negative=true_negative,
        )


# Global service instance
alignment_service = AlignmentService()
