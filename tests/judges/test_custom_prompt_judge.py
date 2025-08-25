"""Unit tests for CustomPromptJudge implementation."""

import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from server.judges.custom_prompt_judge import PROMPT_TEMPLATE, CustomPromptJudge


class TestCustomPromptJudge(TestCase):
    """Test cases for CustomPromptJudge class."""

    def test_custom_prompt_judge_initialization_basic(self):
        """Test CustomPromptJudge initialization with basic parameters."""
        judge = CustomPromptJudge(
            name='Quality Judge', user_instructions='Check if response is helpful and accurate'
        )

        self.assertEqual(judge.name, 'Quality Judge')
        self.assertEqual(judge.user_instructions, 'Check if response is helpful and accurate')
        self.assertEqual(judge.system_instructions, 'Check if response is helpful and accurate')
        self.assertEqual(judge.version, 1)
        self.assertIsNone(judge.experiment_id)
        self.assertIsNotNone(judge.id)
        self.assertIsNotNone(judge.scorer_func)

    def test_custom_prompt_judge_initialization_with_experiment(self):
        """Test CustomPromptJudge initialization with experiment ID."""
        judge = CustomPromptJudge(
            name='Experiment Quality Judge',
            user_instructions='Evaluate response quality',
            experiment_id='exp456',
        )

        self.assertEqual(judge.name, 'Experiment Quality Judge')
        self.assertEqual(judge.experiment_id, 'exp456')

    def test_custom_prompt_judge_prompt_template_formatting(self):
        """Test that CustomPromptJudge properly formats the prompt template."""
        instructions = 'Check if the response is factually correct'
        judge = CustomPromptJudge(name='Fact Check Judge', user_instructions=instructions)

        expected_prompt = PROMPT_TEMPLATE.format(system_instructions=instructions)
        self.assertEqual(judge.prompt_template, expected_prompt)

        # Verify the template contains the instructions
        self.assertIn(instructions, judge.prompt_template)
        self.assertIn('{{request}}', judge.prompt_template)
        self.assertIn('{{response}}', judge.prompt_template)
        self.assertIn('[[pass]]', judge.prompt_template)
        self.assertIn('[[fail]]', judge.prompt_template)

    def test_global_prompt_template_constants(self):
        """Test that the global PROMPT_TEMPLATE has expected structure."""
        self.assertIn('{system_instructions}', PROMPT_TEMPLATE)
        self.assertIn('{{request}}', PROMPT_TEMPLATE)
        self.assertIn('{{response}}', PROMPT_TEMPLATE)
        self.assertIn('[[pass]]', PROMPT_TEMPLATE)
        self.assertIn('[[fail]]', PROMPT_TEMPLATE)
        self.assertIn('evaluation criteria', PROMPT_TEMPLATE.lower())

    def test_scorer_function_creation(self):
        """Test that _get_scorer_function creates a callable scorer."""
        judge = CustomPromptJudge(name='Quality Judge', user_instructions='Check quality')

        # Test that scorer function is created and is callable
        self.assertIsNotNone(judge.scorer_func)
        self.assertTrue(callable(judge.scorer_func))

    @patch('server.judges.custom_prompt_judge.logger')
    def test_scorer_function_error_handling(self, mock_logger):
        """Test scorer function handles errors gracefully."""
        judge = CustomPromptJudge(name='Error Test Judge', user_instructions='This will fail')

        # Since the scorer function uses try/catch internally, we can test
        # that it doesn't raise exceptions even if MLflow components fail
        inputs = {'request': 'test'}
        outputs = {'response': 'test'}

        # This should not raise an exception
        try:
            result = judge.scorer_func(inputs, outputs)
            # The function should return something (either success or error feedback)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f'Scorer function raised an exception: {e}')

    def test_evaluate_method(self):
        """Test evaluate method calls scorer_func with correct parameters."""
        judge = CustomPromptJudge(name='Evaluate Test Judge', user_instructions='Test evaluation')

        # Mock the scorer function
        judge.scorer_func = Mock(return_value='mock_evaluation_result')

        inputs = {'request': 'Test question'}
        outputs = {'response': 'Test answer'}
        mock_trace = Mock()

        result = judge.evaluate(inputs, outputs, mock_trace)

        self.assertEqual(result, 'mock_evaluation_result')
        judge.scorer_func.assert_called_once_with(inputs, outputs, mock_trace)

    def test_evaluate_method_without_trace(self):
        """Test evaluate method without trace parameter."""
        judge = CustomPromptJudge(
            name='No Trace Test Judge', user_instructions='Test without trace'
        )

        judge.scorer_func = Mock(return_value='no_trace_result')

        inputs = {'request': 'No trace question'}
        outputs = {'response': 'No trace answer'}

        result = judge.evaluate(inputs, outputs)

        self.assertEqual(result, 'no_trace_result')
        judge.scorer_func.assert_called_once_with(inputs, outputs, None)

    def test_register_scorer_returns_string(self):
        """Test that register_scorer returns a string (either registered scorer or fallback name)."""
        judge = CustomPromptJudge(
            name='Register Test Judge',
            user_instructions='Test registration',
            experiment_id='exp123',
        )

        result = judge.register_scorer()

        # Should return a string (either successful registration or fallback scorer name)
        self.assertIsInstance(result, str)

    def test_register_scorer_without_experiment_id(self):
        """Test scorer registration without experiment ID."""
        judge = CustomPromptJudge(name='No Exp Judge', user_instructions='Test without experiment')

        result = judge.register_scorer()

        # Should return a string (either successful registration or fallback scorer name)
        self.assertIsInstance(result, str)

    @patch('server.judges.custom_prompt_judge.logger')
    def test_register_scorer_error_handling(self, mock_logger):
        """Test register_scorer handles errors gracefully."""
        judge = CustomPromptJudge(
            name='Error Register Judge', user_instructions='This registration will fail'
        )

        # Since the method has try/catch, it should return a string without raising
        result = judge.register_scorer()

        # Should return a string (fallback scorer name)
        self.assertIsInstance(result, str)

    def test_optimize_with_sufficient_data(self):
        """Test optimize method with sufficient labeled records."""
        judge = CustomPromptJudge(name='Optimize Test Judge', user_instructions='Test optimization')

        labeled_records = [
            {'trace_id': 't1', 'label': 'pass', 'rationale': 'Good response'},
            {'trace_id': 't2', 'label': 'fail', 'rationale': 'Poor response'},
            {'trace_id': 't3', 'label': 'pass', 'rationale': 'Excellent response'},
        ]

        # Currently optimization is not implemented, so should return False
        result = judge.optimize(labeled_records)
        self.assertFalse(result)

    def test_optimize_with_insufficient_data(self):
        """Test optimize method with insufficient labeled records."""
        judge = CustomPromptJudge(
            name='Insufficient Data Judge', user_instructions='Test with insufficient data'
        )

        # Test with 0 records
        result = judge.optimize([])
        self.assertFalse(result)

        # Test with 1 record
        single_record = [{'trace_id': 't1', 'label': 'pass'}]
        result = judge.optimize(single_record)
        self.assertFalse(result)

    @patch('server.judges.custom_prompt_judge.logger')
    def test_optimize_logging(self, mock_logger):
        """Test that optimize method logs appropriately."""
        judge = CustomPromptJudge(
            name='Logging Test Judge', user_instructions='Test logging behavior'
        )

        # Test with insufficient data
        judge.optimize([{'trace_id': 't1'}])

        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn('Need at least 2', warning_call)

        # Test with sufficient data
        labeled_records = [{'trace_id': 't1', 'label': 'pass'}, {'trace_id': 't2', 'label': 'fail'}]

        judge.optimize(labeled_records)

        # Should log start of optimization
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any('Starting optimization' in call for call in info_calls))
        self.assertTrue(any('not yet implemented' in call for call in info_calls))

    def test_prompt_template_with_special_characters(self):
        """Test prompt template formatting with special characters in instructions."""
        special_instructions = "Check if response contains 'quotes' and {braces} and [brackets]"
        judge = CustomPromptJudge(name='Special Char Judge', user_instructions=special_instructions)

        # Should handle special characters without breaking
        self.assertIn(special_instructions, judge.prompt_template)
        self.assertIn("'quotes'", judge.prompt_template)
        self.assertIn('{braces}', judge.prompt_template)
        self.assertIn('[brackets]', judge.prompt_template)

    def test_custom_prompt_judge_inherits_base_properties(self):
        """Test that CustomPromptJudge properly inherits BaseJudge properties."""
        judge = CustomPromptJudge(
            name='Inheritance Test Judge', user_instructions='Test inheritance'
        )

        # Should have all BaseJudge properties
        self.assertTrue(hasattr(judge, 'id'))
        self.assertTrue(hasattr(judge, 'name'))
        self.assertTrue(hasattr(judge, 'user_instructions'))
        self.assertTrue(hasattr(judge, 'system_instructions'))
        self.assertTrue(hasattr(judge, 'version'))
        self.assertTrue(hasattr(judge, 'scorer_func'))

        # Should also have CustomPromptJudge-specific properties
        self.assertTrue(hasattr(judge, 'prompt_template'))


if __name__ == '__main__':
    unittest.main()
