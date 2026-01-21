"""Unit tests for IFEval verifier.

Tests the AllenAI IFEval constraint verification functions from:
https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
"""

import json
import pytest
from rlvr_experiments.verifiers.ifeval import (
    IFEvalVerifier,
    IF_FUNCTIONS_MAP,
    # Case constraints
    validate_lowercase,
    validate_uppercase,
    # Punctuation/format constraints
    validate_no_commas,
    validate_quotation,
    validate_json_format,
    validate_title,
    # Keyword constraints
    verify_keywords,
    verify_keyword_frequency,
    validate_forbidden_words,
    verify_letter_frequency,
    # Structure constraints
    verify_paragraph_count,
    validate_word_constraint,
    verify_sentence_constraint,
    validate_paragraphs,
    verify_bullet_points,
    validate_sections,
    validate_highlighted_sections,
    # Content constraints
    validate_end,
    verify_postscript,
    validate_placeholders,
    validate_two_responses,
    validate_repeat_prompt,
    validate_frequency_capital_words,
    validate_choice,
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

    def test_validate_quotation_fail(self):
        assert validate_quotation("This is not quoted") is False

    def test_validate_quotation_single_quotes(self):
        assert validate_quotation("'This uses single quotes'") is False


class TestKeywordConstraints:
    """Test keyword-related constraints."""

    def test_verify_keywords_pass(self):
        assert verify_keywords(
            "Hello world test", keyword_list=["hello", "test"]
        ) is True

    def test_verify_keywords_fail(self):
        assert verify_keywords(
            "Hello world", keyword_list=["hello", "test"]
        ) is False

    def test_verify_keywords_case_insensitive(self):
        assert verify_keywords(
            "HELLO WORLD", keyword_list=["hello", "world"]
        ) is True

    def test_verify_keyword_frequency_exact(self):
        # AllenAI version checks exact count
        assert verify_keyword_frequency("test test test", word="test", N=3) is True

    def test_verify_keyword_frequency_wrong_count(self):
        assert verify_keyword_frequency("test test", word="test", N=3) is False

    def test_verify_keyword_frequency_case_insensitive(self):
        assert verify_keyword_frequency("Test TEST test", word="test", N=3) is True

    def test_validate_forbidden_words_pass(self):
        assert validate_forbidden_words(
            "This is good", forbidden_words=["bad", "wrong"]
        ) is True

    def test_validate_forbidden_words_fail(self):
        assert validate_forbidden_words(
            "This is bad", forbidden_words=["bad", "wrong"]
        ) is False

    def test_validate_forbidden_words_case_insensitive(self):
        assert validate_forbidden_words(
            "This is BAD", forbidden_words=["bad"]
        ) is False

    def test_verify_letter_frequency_exact(self):
        # AllenAI version checks exact count (case-sensitive)
        assert verify_letter_frequency("aaa", letter="a", N=3) is True

    def test_verify_letter_frequency_wrong_count(self):
        assert verify_letter_frequency("aa", letter="a", N=3) is False


class TestStructureConstraints:
    """Test structure-related constraints."""

    def test_verify_paragraph_count_pass(self):
        # AllenAI uses "* * *" as separator
        response = "Para 1\n* * *\nPara 2\n* * *\nPara 3"
        assert verify_paragraph_count(response, N=3) is True

    def test_verify_paragraph_count_fail(self):
        response = "Para 1\n* * *\nPara 2"
        assert verify_paragraph_count(response, N=3) is False

    def test_validate_word_constraint_at_least(self):
        response = "one two three four five"
        assert validate_word_constraint(response, N=5, quantifier="at least") is True
        assert validate_word_constraint(response, N=6, quantifier="at least") is False

    def test_validate_word_constraint_at_most(self):
        response = "one two three"
        assert validate_word_constraint(response, N=3, quantifier="at most") is True
        assert validate_word_constraint(response, N=2, quantifier="at most") is False

    def test_validate_word_constraint_around(self):
        # "around" allows 10% tolerance
        response = "one two three four five six seven eight nine ten"  # 10 words
        assert validate_word_constraint(response, N=10, quantifier="around") is True
        assert validate_word_constraint(response, N=11, quantifier="around") is True  # Within 10%
        assert validate_word_constraint(response, N=20, quantifier="around") is False

    def test_verify_sentence_constraint_at_least(self):
        response = "First sentence. Second sentence. Third sentence."
        assert verify_sentence_constraint(response, N=3, quantifier="at least") is True

    def test_verify_sentence_constraint_around(self):
        response = "First sentence. Second sentence."
        assert verify_sentence_constraint(response, N=2, quantifier="around") is True
        assert verify_sentence_constraint(response, N=3, quantifier="around") is True  # Within 1

    def test_validate_paragraphs_with_first_word(self):
        response = "First paragraph.\n\nSecond paragraph here.\n\nThird paragraph."
        assert validate_paragraphs(response, N=3, i=2, first_word="Second") is True
        assert validate_paragraphs(response, N=3, i=2, first_word="Wrong") is False

    def test_validate_paragraphs_wrong_count(self):
        response = "First paragraph.\n\nSecond paragraph."
        assert validate_paragraphs(response, N=3, i=1, first_word="First") is False

    def test_verify_bullet_points_exact(self):
        # AllenAI version checks exact count
        response = "- Item 1\n- Item 2\n- Item 3"
        assert verify_bullet_points(response, N=3) is True

    def test_verify_bullet_points_asterisks(self):
        response = "* Item 1\n* Item 2"
        assert verify_bullet_points(response, N=2) is True

    def test_verify_bullet_points_wrong_count(self):
        response = "- Item 1\n- Item 2"
        assert verify_bullet_points(response, N=3) is False

    def test_validate_sections_pass(self):
        response = "Section 1###Section 2###Section 3"
        assert validate_sections(response, N=3, section_splitter="###") is True

    def test_validate_sections_leading_separator(self):
        # AllenAI removes leading empty section
        response = "###Section 1###Section 2"
        assert validate_sections(response, N=2, section_splitter="###") is True


class TestContentConstraints:
    """Test content-related constraints."""

    def test_validate_end_pass(self):
        assert validate_end("Some text The End", end_phrase="The End") is True

    def test_validate_end_fail(self):
        assert validate_end("Some text not ending correctly", end_phrase="The End") is False

    def test_verify_postscript_pass(self):
        assert verify_postscript(
            "Main content\n\nP.S. Don't forget!",
            postscript_marker="P.S."
        ) is True

    def test_verify_postscript_needs_content(self):
        # Postscript marker must have content after it
        assert verify_postscript("Main P.S.", postscript_marker="P.S.") is False

    def test_verify_postscript_fail(self):
        assert verify_postscript(
            "Main content without postscript",
            postscript_marker="P.S."
        ) is False

    def test_validate_placeholders_pass(self):
        assert validate_placeholders("Hello [NAME], welcome to [PLACE]!", N=2) is True

    def test_validate_placeholders_fail(self):
        assert validate_placeholders("Hello [NAME]!", N=2) is False

    def test_validate_two_responses_pass(self):
        response = "Response 1\n******\nResponse 2"
        assert validate_two_responses(response) is True

    def test_validate_two_responses_same(self):
        # Must be different responses
        response = "Same\n******\nSame"
        assert validate_two_responses(response) is False

    def test_validate_two_responses_wrong_separator(self):
        response = "Response 1\n---\nResponse 2"
        assert validate_two_responses(response) is False  # AllenAI uses ******

    def test_validate_repeat_prompt_pass(self):
        assert validate_repeat_prompt(
            "Original prompt: here is my response",
            original_prompt="Original prompt:"
        ) is True

    def test_validate_repeat_prompt_fail(self):
        assert validate_repeat_prompt(
            "Different start",
            original_prompt="Original prompt:"
        ) is False


class TestFormatConstraints:
    """Test format-related constraints."""

    def test_validate_json_format_pass(self):
        assert validate_json_format('{"key": "value"}') is True

    def test_validate_json_format_array(self):
        assert validate_json_format('[1, 2, 3]') is True

    def test_validate_json_format_fail(self):
        assert validate_json_format("Not valid JSON") is False

    def test_validate_title_pass(self):
        assert validate_title("<<My Title>>\nContent here") is True

    def test_validate_title_fail(self):
        assert validate_title("# My Title\nContent here") is False

    def test_validate_highlighted_sections_pass(self):
        response = "This is *important* and *very important*"
        assert validate_highlighted_sections(response, N=2) is True

    def test_validate_highlighted_sections_fail(self):
        response = "This is *important*"
        assert validate_highlighted_sections(response, N=2) is False

    def test_validate_frequency_capital_words_at_least(self):
        response = "This is VERY IMPORTANT INFO"
        assert validate_frequency_capital_words(response, N=3, quantifier="at least") is True

    def test_validate_frequency_capital_words_at_most(self):
        response = "This is IMPORTANT"
        assert validate_frequency_capital_words(response, N=2, quantifier="at most") is True

    def test_validate_choice_pass(self):
        # AllenAI checks if text is in any option
        assert validate_choice("a)", options=["a)", "b)", "c)"]) is True

    def test_validate_choice_fail(self):
        assert validate_choice("d)", options=["a)", "b)", "c)"]) is False


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
            "word": "test",
            "N": 2
        })
        assert verifier.verify("test test", gt) == 1.0
        assert verifier.verify("test once", gt) == 0.0

    def test_verify_keywords(self, verifier):
        gt = json.dumps({
            "func_name": "verify_keywords",
            "keyword_list": ["hello", "world"]
        })
        assert verifier.verify("hello world", gt) == 1.0
        assert verifier.verify("hello there", gt) == 0.0

    def test_validate_forbidden_words(self, verifier):
        gt = json.dumps({
            "func_name": "validate_forbidden_words",
            "forbidden_words": ["bad", "wrong"]
        })
        assert verifier.verify("this is good", gt) == 1.0
        assert verifier.verify("this is bad", gt) == 0.0

    def test_validate_json_format(self, verifier):
        gt = json.dumps({"func_name": "validate_json_format"})
        assert verifier.verify('{"key": "value"}', gt) == 1.0
        assert verifier.verify("not json", gt) == 0.0

    def test_validate_choice(self, verifier):
        gt = json.dumps({
            "func_name": "validate_choice",
            "options": ["a)", "b)", "c)", "d)"]
        })
        assert verifier.verify("a)", gt) == 1.0
        assert verifier.verify("e)", gt) == 0.0

    def test_validate_paragraphs(self, verifier):
        gt = json.dumps({
            "func_name": "validate_paragraphs",
            "N": 3,
            "i": 2,
            "first_word": "Second"
        })
        response = "First para.\n\nSecond para here.\n\nThird para."
        assert verifier.verify(response, gt) == 1.0

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
            "hello, world",  # fail no commas
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


class TestIFFunctionsMap:
    """Test that all expected functions are in the registry."""

    def test_all_functions_registered(self):
        expected_functions = [
            "verify_keywords",
            "verify_keyword_frequency",
            "validate_forbidden_words",
            "verify_letter_frequency",
            "validate_response_language",
            "verify_paragraph_count",
            "validate_word_constraint",
            "verify_sentence_constraint",
            "validate_paragraphs",
            "verify_postscript",
            "validate_placeholders",
            "verify_bullet_points",
            "validate_title",
            "validate_choice",
            "validate_highlighted_sections",
            "validate_sections",
            "validate_json_format",
            "validate_repeat_prompt",
            "validate_two_responses",
            "validate_uppercase",
            "validate_lowercase",
            "validate_frequency_capital_words",
            "validate_end",
            "validate_quotation",
            "validate_no_commas",
        ]
        for func_name in expected_functions:
            assert func_name in IF_FUNCTIONS_MAP, f"Missing function: {func_name}"

    def test_function_count(self):
        # Should have exactly 25 functions (IFEval taxonomy)
        assert len(IF_FUNCTIONS_MAP) == 25
