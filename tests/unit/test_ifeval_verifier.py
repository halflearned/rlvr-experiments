"""Unit tests for IFEval verifier."""

import json
import pytest
from rlvr_experiments.verifiers.ifeval import (
    IFEvalVerifier,
    validate_lowercase,
    validate_uppercase,
    validate_no_commas,
    validate_quotation,
    verify_keyword_frequency,
    verify_keyword_existence,
    verify_forbidden_words,
    verify_letter_frequency,
    verify_paragraph_count,
    verify_sentence_count,
    verify_word_count,
    verify_bullet_points,
    verify_end_phrase,
    verify_postscript,
    verify_json_format,
    verify_title_format,
    verify_placeholder,
    verify_two_responses,
    verify_highlighted_sections,
    verify_capitalized_words,
)


class TestCaseConstraints:
    """Test case-related constraints."""

    def test_validate_lowercase_pass(self):
        assert validate_lowercase("hello world") is True

    def test_validate_lowercase_fail(self):
        assert validate_lowercase("Hello World") is False

    def test_validate_lowercase_with_numbers(self):
        assert validate_lowercase("hello 123 world") is True

    def test_validate_uppercase_pass(self):
        assert validate_uppercase("HELLO WORLD") is True

    def test_validate_uppercase_fail(self):
        assert validate_uppercase("Hello World") is False

    def test_validate_uppercase_with_numbers(self):
        assert validate_uppercase("HELLO 123 WORLD") is True


class TestPunctuationConstraints:
    """Test punctuation and format constraints."""

    def test_validate_no_commas_pass(self):
        assert validate_no_commas("Hello world. This is a test.") is True

    def test_validate_no_commas_fail(self):
        assert validate_no_commas("Hello, world") is False

    def test_validate_quotation_pass(self):
        assert validate_quotation('"This is quoted"') is True

    def test_validate_quotation_with_whitespace(self):
        assert validate_quotation('  "This is quoted"  ') is True

    def test_validate_quotation_fail(self):
        assert validate_quotation("This is not quoted") is False

    def test_validate_quotation_single_quotes(self):
        assert validate_quotation("'This uses single quotes'") is False


class TestKeywordConstraints:
    """Test keyword-related constraints."""

    def test_verify_keyword_frequency_pass(self):
        assert verify_keyword_frequency("test test test", keyword="test", N=3) is True

    def test_verify_keyword_frequency_fail(self):
        assert verify_keyword_frequency("test test", keyword="test", N=3) is False

    def test_verify_keyword_frequency_case_insensitive(self):
        assert verify_keyword_frequency("Test TEST test", keyword="test", N=3) is True

    def test_verify_keyword_frequency_none_params(self):
        assert verify_keyword_frequency("test") is False

    def test_verify_keyword_existence_pass(self):
        assert verify_keyword_existence(
            "Hello world test", keyword_list=["hello", "test"]
        ) is True

    def test_verify_keyword_existence_fail(self):
        assert verify_keyword_existence(
            "Hello world", keyword_list=["hello", "test"]
        ) is False

    def test_verify_keyword_existence_case_insensitive(self):
        assert verify_keyword_existence(
            "HELLO WORLD", keyword_list=["hello", "world"]
        ) is True

    def test_verify_keyword_existence_empty_list(self):
        assert verify_keyword_existence("Hello world", keyword_list=[]) is True

    def test_verify_keyword_existence_none(self):
        assert verify_keyword_existence("Hello world", keyword_list=None) is True

    def test_verify_forbidden_words_pass(self):
        assert verify_forbidden_words(
            "This is good", forbidden_words=["bad", "wrong"]
        ) is True

    def test_verify_forbidden_words_fail(self):
        assert verify_forbidden_words(
            "This is bad", forbidden_words=["bad", "wrong"]
        ) is False

    def test_verify_forbidden_words_case_insensitive(self):
        assert verify_forbidden_words(
            "This is BAD", forbidden_words=["bad"]
        ) is False

    def test_verify_letter_frequency_pass(self):
        assert verify_letter_frequency("aaa bbb", letter="a", N=3) is True

    def test_verify_letter_frequency_fail(self):
        assert verify_letter_frequency("aa bbb", letter="a", N=3) is False

    def test_verify_letter_frequency_case_insensitive(self):
        assert verify_letter_frequency("AaA", letter="a", N=3) is True


class TestStructureConstraints:
    """Test structure-related constraints."""

    def test_verify_paragraph_count_pass(self):
        response = "Para 1\n***\nPara 2\n***\nPara 3"
        assert verify_paragraph_count(response, N=3, section_splitter="***") is True

    def test_verify_paragraph_count_fail(self):
        response = "Para 1\n***\nPara 2"
        assert verify_paragraph_count(response, N=3, section_splitter="***") is False

    def test_verify_paragraph_count_default_splitter(self):
        response = "Para 1\n***\nPara 2"
        assert verify_paragraph_count(response, N=2) is True

    def test_verify_sentence_count_pass(self):
        response = "First sentence. Second sentence. Third sentence!"
        assert verify_sentence_count(response, N=3) is True

    def test_verify_sentence_count_fail(self):
        response = "First sentence. Second sentence."
        assert verify_sentence_count(response, N=3) is False

    def test_verify_word_count_at_least(self):
        response = "one two three four five"
        assert verify_word_count(response, N=5, quantifier="at_least") is True
        assert verify_word_count(response, N=6, quantifier="at_least") is False

    def test_verify_word_count_at_most(self):
        response = "one two three"
        assert verify_word_count(response, N=3, quantifier="at_most") is True
        assert verify_word_count(response, N=2, quantifier="at_most") is False

    def test_verify_word_count_exactly(self):
        response = "one two three"
        assert verify_word_count(response, N=3, quantifier="exactly") is True
        assert verify_word_count(response, N=2, quantifier="exactly") is False

    def test_verify_bullet_points_pass(self):
        response = "- Item 1\n- Item 2\n- Item 3"
        assert verify_bullet_points(response, N=3) is True

    def test_verify_bullet_points_asterisks(self):
        response = "* Item 1\n* Item 2"
        assert verify_bullet_points(response, N=2) is True

    def test_verify_bullet_points_fail(self):
        response = "- Item 1\n- Item 2"
        assert verify_bullet_points(response, N=3) is False


class TestContentConstraints:
    """Test content-related constraints."""

    def test_verify_end_phrase_pass(self):
        assert verify_end_phrase(
            "Some text ending with: The End", end_phrase="The End"
        ) is True

    def test_verify_end_phrase_fail(self):
        assert verify_end_phrase(
            "Some text not ending correctly", end_phrase="The End"
        ) is False

    def test_verify_end_phrase_with_whitespace(self):
        assert verify_end_phrase(
            "Some text ending with: The End  ", end_phrase="The End"
        ) is True

    def test_verify_postscript_pass(self):
        assert verify_postscript("Main content\n\nP.S. Don't forget!") is True

    def test_verify_postscript_custom_marker(self):
        assert verify_postscript(
            "Main content\n\nPS: Note this", postscript_marker="PS:"
        ) is True

    def test_verify_postscript_fail(self):
        assert verify_postscript("Main content without postscript") is False


class TestFormatConstraints:
    """Test format-related constraints."""

    def test_verify_json_format_pass(self):
        assert verify_json_format('{"key": "value"}') is True

    def test_verify_json_format_array(self):
        assert verify_json_format('[1, 2, 3]') is True

    def test_verify_json_format_in_code_block(self):
        response = 'Here is JSON:\n```json\n{"key": "value"}\n```'
        assert verify_json_format(response) is True

    def test_verify_json_format_fail(self):
        assert verify_json_format("Not valid JSON") is False

    def test_verify_title_format_pass(self):
        assert verify_title_format("<<My Title>>\nContent here") is True

    def test_verify_title_format_fail(self):
        assert verify_title_format("# My Title\nContent here") is False

    def test_verify_placeholder_pass(self):
        assert verify_placeholder("Hello [NAME], welcome to [PLACE]!", N=2) is True

    def test_verify_placeholder_fail(self):
        assert verify_placeholder("Hello [NAME]!", N=2) is False

    def test_verify_two_responses_pass(self):
        assert verify_two_responses("Response 1\n***\nResponse 2") is True

    def test_verify_two_responses_dashes(self):
        assert verify_two_responses("Response 1\n---\nResponse 2") is True

    def test_verify_two_responses_fail(self):
        assert verify_two_responses("Single response only") is False

    def test_verify_highlighted_sections_pass(self):
        response = "This is *important* and **very important**"
        assert verify_highlighted_sections(response, N=2) is True

    def test_verify_highlighted_sections_fail(self):
        response = "This is *important*"
        assert verify_highlighted_sections(response, N=2) is False

    def test_verify_capitalized_words_pass(self):
        response = "This is VERY IMPORTANT"
        assert verify_capitalized_words(response, N=2) is True

    def test_verify_capitalized_words_fail(self):
        response = "This is IMPORTANT"
        assert verify_capitalized_words(response, N=2) is False


class TestIFEvalVerifier:
    """Test the IFEvalVerifier class."""

    @pytest.fixture
    def verifier(self):
        return IFEvalVerifier()

    def test_verify_lowercase(self, verifier):
        gt = json.dumps({"func_name": "validate_lowercase"})
        assert verifier.verify("hello world", gt) == 1.0
        assert verifier.verify("Hello World", gt) == 0.0

    def test_verify_uppercase(self, verifier):
        gt = json.dumps({"func_name": "validate_uppercase"})
        assert verifier.verify("HELLO WORLD", gt) == 1.0
        assert verifier.verify("Hello World", gt) == 0.0

    def test_verify_no_commas(self, verifier):
        gt = json.dumps({"func_name": "validate_no_commas"})
        assert verifier.verify("Hello world", gt) == 1.0
        assert verifier.verify("Hello, world", gt) == 0.0

    def test_verify_keyword_frequency(self, verifier):
        gt = json.dumps({
            "func_name": "verify_keyword_frequency",
            "keyword": "test",
            "N": 2
        })
        assert verifier.verify("test test test", gt) == 1.0
        assert verifier.verify("test once", gt) == 0.0

    def test_verify_forbidden_words(self, verifier):
        gt = json.dumps({
            "func_name": "verify_forbidden_words",
            "forbidden_words": ["bad", "wrong"]
        })
        assert verifier.verify("this is good", gt) == 1.0
        assert verifier.verify("this is bad", gt) == 0.0

    def test_verify_json_format(self, verifier):
        gt = json.dumps({"func_name": "verify_json_format"})
        assert verifier.verify('{"key": "value"}', gt) == 1.0
        assert verifier.verify("not json", gt) == 0.0

    def test_verify_with_dict_ground_truth(self, verifier):
        gt = {"func_name": "validate_lowercase"}
        assert verifier.verify("hello world", gt) == 1.0

    def test_unknown_function(self, verifier):
        gt = json.dumps({"func_name": "unknown_function"})
        assert verifier.verify("any response", gt) == 0.0

    def test_missing_func_name(self, verifier):
        gt = json.dumps({"N": 5})
        assert verifier.verify("any response", gt) == 0.0

    def test_invalid_json(self, verifier):
        gt = "not valid json"
        assert verifier.verify("any response", gt) == 0.0

    def test_null_params_filtered(self, verifier):
        # Real format from RLVR-IFeval with null values
        gt = json.dumps({
            "func_name": "validate_lowercase",
            "N": None,
            "quantifier": None,
            "keyword_list": None
        })
        assert verifier.verify("hello world", gt) == 1.0


class TestIFEvalVerifierAsync:
    """Test async methods of IFEvalVerifier."""

    @pytest.fixture
    def verifier(self):
        return IFEvalVerifier()

    @pytest.mark.asyncio
    async def test_verify_completions(self, verifier):
        problem = {
            "ground_truth": json.dumps({"func_name": "validate_lowercase"})
        }
        completions = [
            "hello world",
            "Hello World",
            "HELLO WORLD",
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert scores == [1.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        problems = [
            {"ground_truth": json.dumps({"func_name": "validate_lowercase"})},
            {"ground_truth": json.dumps({"func_name": "validate_uppercase"})},
            {"ground_truth": json.dumps({"func_name": "validate_no_commas"})},
        ]
        completions = [
            "hello world",  # pass lowercase
            "HELLO WORLD",  # pass uppercase
            "hello, world", # fail no commas
        ]
        scores, durations = await verifier.verify_batch(problems, completions)
        assert scores == [1.0, 1.0, 0.0]
        assert len(durations) == 3
        assert all(d >= 0 for d in durations)

    @pytest.mark.asyncio
    async def test_verify_batch_with_timing(self, verifier):
        problems = [
            {"ground_truth": json.dumps({"func_name": "validate_lowercase"})}
        ]
        completions = ["hello"]
        scores, durations, timing_spans = await verifier.verify_batch_with_timing(
            problems, completions
        )
        assert scores == [1.0]
        assert len(durations) == 1
        assert len(timing_spans) == 1
        assert timing_spans[0][0] == 0.0  # First span starts at 0
