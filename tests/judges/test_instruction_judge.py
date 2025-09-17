"""Unit tests for InstructionJudge implementation."""

import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from server.judges.instruction_judge import InstructionJudge


class TestInstructionJudge(TestCase):
    """Test cases for InstructionJudge class."""

    @patch('server.judges.instruction_judge.make_judge')
    def test_instruction_judge_initialization_basic(self, mock_make_judge):
        """Test InstructionJudge initialization with basic parameters."""
        mock_judge = Mock()
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(
            name='Quality Judge', user_instructions='Check if response is helpful and accurate'
        )

        self.assertEqual(judge.name, 'Quality Judge')
        self.assertEqual(judge.user_instructions, 'Check if response is helpful and accurate')
        self.assertEqual(judge.system_instructions, 'Check if response is helpful and accurate')
        self.assertEqual(judge.version, 1)
        self.assertIsNone(judge.experiment_id)
        self.assertIsNotNone(judge.id)
        self.assertIsNotNone(judge.scorer_func)
        self.assertEqual(judge.scorer_func, mock_judge)
        
        # Verify make_judge was called with correct parameters
        mock_make_judge.assert_called_once()
        call_args = mock_make_judge.call_args
        self.assertIn('name', call_args.kwargs)
        self.assertIn('instructions', call_args.kwargs)
        self.assertIn('model', call_args.kwargs)

    @patch('server.judges.instruction_judge.make_judge')
    def test_instruction_judge_initialization_with_experiment(self, mock_make_judge):
        """Test InstructionJudge initialization with experiment ID."""
        mock_judge = Mock()
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(
            name='Experiment Quality Judge',
            user_instructions='Evaluate response quality',
            experiment_id='exp456',
        )

        self.assertEqual(judge.name, 'Experiment Quality Judge')
        self.assertEqual(judge.experiment_id, 'exp456')

    @patch('server.judges.instruction_judge.make_judge')
    def test_instruction_judge_with_custom_model(self, mock_make_judge):
        """Test InstructionJudge with custom model parameter."""
        mock_judge = Mock()
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(
            name='Custom Model Judge',
            user_instructions='Test custom model',
        )

        mock_make_judge.assert_called_once()
        call_args = mock_make_judge.call_args
        self.assertEqual(call_args.kwargs['model'], 'anthropic:/claude-3-sonnet')

    @patch('server.judges.instruction_judge.make_judge')
    def test_evaluate_method_with_trace_preferred(self, mock_make_judge):
        """Test evaluate method prefers trace when provided."""
        mock_judge = Mock()
        mock_feedback = Mock()
        mock_feedback.metadata = {'existing': 'data'}
        mock_judge.return_value = mock_feedback
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Evaluate Test Judge', user_instructions='Test evaluation')

        inputs = {'request': 'Test question'}
        outputs = {'response': 'Test answer'}
        mock_trace = Mock()

        result = judge.evaluate(inputs, outputs, mock_trace)

        # Verify MLflow judge was called with trace (preferred method)
        mock_judge.assert_called_once_with(trace=mock_trace)
        
        # Verify version metadata was added
        self.assertEqual(result.metadata['version'], '1')
        self.assertEqual(result.metadata['existing'], 'data')

    @patch('server.judges.instruction_judge.make_judge')
    @patch('server.judges.instruction_judge.parse_inputs_to_str')
    @patch('server.judges.instruction_judge.parse_outputs_to_str')
    def test_evaluate_with_trace(self, mock_parse_outputs, mock_parse_inputs, mock_make_judge):
        """Test evaluate method with trace-based evaluation."""
        mock_judge = Mock()
        mock_feedback = Mock()
        mock_feedback.metadata = {'existing': 'data'}
        mock_judge.return_value = mock_feedback
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Trace Test Judge', user_instructions='Test trace evaluation')

        mock_trace = Mock()
        result = judge.evaluate({}, {}, trace=mock_trace)

        # Verify MLflow judge was called with trace (canonical parsing not used)
        mock_judge.assert_called_once_with(trace=mock_trace)
        mock_parse_inputs.assert_not_called()
        mock_parse_outputs.assert_not_called()
        
        # Verify version metadata was added
        self.assertEqual(result.metadata['version'], '1')
        self.assertEqual(result.metadata['existing'], 'data')

    @patch('server.judges.instruction_judge.make_judge')
    @patch('server.judges.instruction_judge.parse_inputs_to_str')
    @patch('server.judges.instruction_judge.parse_outputs_to_str')
    def test_evaluate_fallback_to_inputs_outputs(self, mock_parse_outputs, mock_parse_inputs, mock_make_judge):
        """Test evaluate method fallback when no trace provided."""
        mock_judge = Mock()
        mock_feedback = Mock()
        mock_feedback.metadata = None
        mock_judge.return_value = mock_feedback
        mock_make_judge.return_value = mock_judge
        mock_parse_inputs.return_value = "parsed request"
        mock_parse_outputs.return_value = "parsed response"
        
        judge = InstructionJudge(name='Fallback Test Judge', user_instructions='Test fallback')

        inputs = {'request': 'test question'}
        outputs = {'response': 'test answer'}
        result = judge.evaluate(inputs, outputs)

        # Verify canonical parsing was used
        mock_parse_inputs.assert_called_once_with(inputs)
        mock_parse_outputs.assert_called_once_with(outputs)
        mock_judge.assert_called_once_with(
            inputs={'request': 'parsed request'}, 
            outputs={'response': 'parsed response'}
        )
        
        # Verify version metadata was added
        self.assertEqual(result.metadata, {'version': '1'})

    @patch('server.judges.instruction_judge.make_judge')
    def test_register_scorer_with_experiment(self, mock_make_judge):
        """Test register_scorer with experiment ID."""
        mock_judge = Mock()
        mock_registered = Mock()
        mock_judge.register.return_value = mock_registered
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(
            name='Register Test Judge',
            user_instructions='Test registration',
            experiment_id='exp123',
        )

        result = judge.register_scorer()

        # Verify register was called with proper naming and experiment_id
        mock_judge.register.assert_called_once()
        call_args = mock_judge.register.call_args
        self.assertIn('name', call_args.kwargs)
        self.assertEqual(call_args.kwargs['experiment_id'], 'exp123')
        self.assertEqual(result, mock_registered)

    @patch('server.judges.instruction_judge.make_judge')
    def test_register_scorer_without_experiment(self, mock_make_judge):
        """Test register_scorer without experiment ID."""
        mock_judge = Mock()
        mock_registered = Mock()
        mock_judge.register.return_value = mock_registered
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='No Exp Judge', user_instructions='Test without experiment')

        result = judge.register_scorer()

        # Verify register was called without experiment_id
        mock_judge.register.assert_called_once()
        call_args = mock_judge.register.call_args
        self.assertIn('name', call_args.kwargs)
        self.assertNotIn('experiment_id', call_args.kwargs)

    @patch('server.judges.instruction_judge.make_judge')
    @patch('server.judges.instruction_judge.logger')
    def test_register_scorer_error_handling(self, mock_logger, mock_make_judge):
        """Test register_scorer handles errors gracefully."""
        mock_judge = Mock()
        mock_judge.register.side_effect = Exception("Registration failed")
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(
            name='Error Register Judge', user_instructions='This registration will fail'
        )

        result = judge.register_scorer()

        # Should return None and log warning
        self.assertIsNone(result)
        mock_logger.warning.assert_called_once()

    @patch('server.judges.instruction_judge.make_judge')
    def test_optimize_with_sufficient_data(self, mock_make_judge):
        """Test optimize method with sufficient traces."""
        mock_judge = Mock()
        mock_aligned_judge = Mock()
        mock_judge.align.return_value = mock_aligned_judge
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Optimize Test Judge', user_instructions='Test optimization')

        # Create mock traces (need at least 10 for MIN_EXAMPLES_FOR_OPTIMIZATION)
        traces = [Mock() for _ in range(12)]

        result = judge.optimize(traces)

        # Verify alignment was called and judge was updated
        mock_judge.align.assert_called_once_with(traces=traces)
        self.assertEqual(judge.scorer_func, mock_aligned_judge)
        self.assertTrue(result)

    @patch('server.judges.instruction_judge.make_judge')
    @patch('server.judges.instruction_judge.logger')
    def test_register_scorer_failure_returns_none(self, mock_logger, mock_make_judge):
        """Test register_scorer returns None on failure."""
        mock_judge = Mock()
        mock_judge.register.side_effect = Exception("Registration failed")
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(
            name='Registration Fail Judge', user_instructions='Test registration failure'
        )

        result = judge.register_scorer()

        # Should return None and log warning
        self.assertIsNone(result)
        mock_logger.warning.assert_called_once()

    @patch('server.judges.instruction_judge.make_judge')
    @patch('server.judges.instruction_judge.logger')
    def test_optimize_alignment_failure(self, mock_logger, mock_make_judge):
        """Test optimize method when alignment fails."""
        mock_judge = Mock()
        mock_judge.align.side_effect = Exception("Alignment failed")
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Alignment Fail Judge', user_instructions='Test alignment failure')

        traces = [Mock() for _ in range(12)]  # Sufficient traces
        result = judge.optimize(traces)

        # Should return False and log error
        self.assertFalse(result)
        mock_logger.error.assert_called_once()
        self.assertIn("Alignment failed", mock_logger.error.call_args[0][0])

    @patch('server.judges.instruction_judge.make_judge')
    def test_scorer_function_creation(self, mock_make_judge):
        """Test that _create_scorer returns the MLflow judge directly."""
        mock_judge = Mock()
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Scorer Test Judge', user_instructions='Test scorer creation')

        # Test that scorer function is the MLflow judge and is callable
        self.assertEqual(judge.scorer_func, mock_judge)
        self.assertTrue(callable(judge.scorer_func))

    @patch('server.judges.instruction_judge.make_judge')
    def test_instruction_judge_inherits_base_properties(self, mock_make_judge):
        """Test that InstructionJudge properly inherits BaseJudge properties."""
        mock_judge = Mock()
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(
            name='Inheritance Test Judge', user_instructions='Test inheritance'
        )

        # Should have all BaseJudge properties
        self.assertTrue(hasattr(judge, 'id'))
        self.assertTrue(hasattr(judge, 'name'))
        self.assertTrue(hasattr(judge, 'user_instructions'))
        self.assertTrue(hasattr(judge, 'system_instructions'))
        self.assertTrue(hasattr(judge, 'version'))
        self.assertTrue(hasattr(judge, 'scorer_func'))

        # Should also have InstructionJudge-specific properties  
        self.assertTrue(hasattr(judge, 'scorer_func'))


if __name__ == '__main__':
    unittest.main()