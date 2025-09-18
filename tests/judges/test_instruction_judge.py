"""Unit tests for InstructionJudge implementation."""

import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from server.judges.instruction_judge import InstructionJudge


class TestInstructionJudge(TestCase):
    """Test cases for InstructionJudge class."""

    @patch('server.judges.instruction_judge.make_judge')
    def test_judge_creation_and_basic_properties(self, mock_make_judge):
        """Test that judge can be created with proper configuration."""
        mock_judge = Mock()
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(
            name='Quality Judge', 
            user_instructions='Check if response is helpful and accurate',
            experiment_id='exp456'
        )

        # Test basic judge properties are set correctly
        self.assertEqual(judge.name, 'Quality Judge')
        self.assertEqual(judge.user_instructions, 'Check if response is helpful and accurate')
        self.assertEqual(judge.experiment_id, 'exp456')
        self.assertEqual(judge.version, 1)
        self.assertIsNotNone(judge.id)
        
        # Verify MLflow integration is properly initialized
        mock_make_judge.assert_called_once()
        self.assertIsNotNone(judge.scorer_func)

    @patch('server.judges.instruction_judge.make_judge')
    def test_evaluate_returns_feedback_with_version(self, mock_make_judge):
        """Test that evaluate method returns feedback with version metadata."""
        mock_judge = Mock()
        mock_feedback = Mock()
        mock_feedback.metadata = {'existing': 'data'}
        mock_judge.return_value = mock_feedback
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Test Judge', user_instructions='Test evaluation')

        inputs = {'request': 'Test question'}
        outputs = {'response': 'Test answer'}
        trace = Mock()

        result = judge.evaluate(inputs, outputs, trace)

        # Test that evaluation returns feedback with correct version
        self.assertIsNotNone(result)
        self.assertEqual(result.metadata['version'], '1')
        
    @patch('server.judges.instruction_judge.make_judge')  
    def test_evaluate_handles_missing_metadata(self, mock_make_judge):
        """Test evaluate gracefully handles feedback without metadata."""
        mock_judge = Mock()
        mock_feedback = Mock()
        mock_feedback.metadata = None
        mock_judge.return_value = mock_feedback
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Test Judge', user_instructions='Test evaluation')

        result = judge.evaluate({'input': 'test'}, {'output': 'test'})

        # Should add version metadata even when none exists
        self.assertEqual(result.metadata, {'version': '1'})

    @patch('server.judges.instruction_judge.make_judge')
    def test_scorer_registration_success(self, mock_make_judge):
        """Test successful scorer registration."""
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

        # Should return registered scorer
        self.assertEqual(result, mock_registered)

    @patch('server.judges.instruction_judge.make_judge')
    def test_scorer_registration_handles_failure(self, mock_make_judge):
        """Test scorer registration gracefully handles failures."""
        mock_judge = Mock()
        mock_judge.register.side_effect = Exception("Registration failed")
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Test Judge', user_instructions='Test registration')

        result = judge.register_scorer()

        # Should handle error gracefully and return None
        self.assertIsNone(result)

    @patch('server.judges.instruction_judge.make_judge')
    def test_judge_optimization_success(self, mock_make_judge):
        """Test successful judge optimization with training data."""
        mock_judge = Mock()
        mock_aligned_judge = Mock()
        mock_judge.align.return_value = mock_aligned_judge
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Test Judge', user_instructions='Test optimization')

        # Provide sufficient training traces
        traces = [Mock() for _ in range(12)]
        result = judge.optimize(traces)

        # Should return success and update judge
        self.assertTrue(result)
        self.assertEqual(judge.scorer_func, mock_aligned_judge)

    @patch('server.judges.instruction_judge.make_judge')
    def test_judge_optimization_handles_failure(self, mock_make_judge):
        """Test judge optimization handles alignment failures."""
        mock_judge = Mock()
        mock_judge.align.side_effect = Exception("Alignment failed")
        mock_make_judge.return_value = mock_judge
        
        judge = InstructionJudge(name='Test Judge', user_instructions='Test optimization')

        traces = [Mock() for _ in range(12)]
        result = judge.optimize(traces)

        # Should handle failure gracefully
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()