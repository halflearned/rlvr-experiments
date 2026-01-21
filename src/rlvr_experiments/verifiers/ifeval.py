"""IFEval verifier for instruction-following constraints.

Implements verification functions from AllenAI's open-instruct:
https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py

Covers all 25 constraints from the IFEval taxonomy.
"""

import json
import re
import time


# =============================================================================
# AllenAI IFEval constraint verification functions
# From: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
# =============================================================================


def verify_keywords(text, keyword_list, **kwargs):
    """
    Verify if the response contains all the specified keywords.
    """
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


def verify_keyword_frequency(text, word, N, **kwargs):
    """
    Verifies if a keyword appears exactly N times in the given text.
    """
    text = text.lower()
    keyword = word.lower()
    words = re.findall(r"\b\w+\b", text)
    actual_count = sum(1 for w in words if w == keyword)
    return actual_count == N


def validate_forbidden_words(text, forbidden_words, **kwargs):
    """
    Validates that the text does not contain any of the specified forbidden words.
    """
    text_lower = text.lower()
    found_words = [word for word in forbidden_words if word.lower() in text_lower]
    return len(found_words) == 0


def verify_letter_frequency(text, letter, N, **kwargs):
    """
    Verifies if a given letter appears exactly the specified number of times in the text.
    """
    if len(letter) != 1:
        return False
    actual_count = text.count(letter)
    return actual_count == N


def validate_response_language(text, language, **kwargs):
    """
    Validates that the entire response is in the specified language.
    """
    try:
        from langdetect import detect
        detected_language = detect(text)
        return detected_language == language
    except Exception:
        return False


def verify_paragraph_count(text, N, **kwargs):
    """
    Verifies that a text contains the expected number of paragraphs,
    where paragraphs are separated by markdown dividers '* * *'
    """
    def clean_text(text):
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    text = clean_text(text)
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False
    return actual_count == N


def validate_word_constraint(text, N, quantifier, **kwargs):
    """
    Validates if a text meets specified word count constraints.
    """
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    else:
        return False


def verify_sentence_constraint(text, N, quantifier, **kwargs):
    """
    Verifies if a text contains the expected number of sentences.
    """
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    actual_count = len(sentences)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    else:
        return False


def validate_paragraphs(text, N, first_word, i, **kwargs):
    """
    Validates that a text contains the expected number of paragraphs and that
    the i-th paragraph starts with a specific word.
    """
    paragraphs = text.split("\n\n")

    if len(paragraphs) != N:
        return False

    return bool(paragraphs[i - 1].strip().startswith(first_word))


def verify_postscript(text, postscript_marker, **kwargs):
    """
    Verifies if a text contains a postscript starting with the marker.
    """
    if postscript_marker in text:
        marker_index = text.find(postscript_marker)
        remaining_text = text[marker_index:].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


def validate_placeholders(text, N, **kwargs):
    """
    Validates if a text contains at least the specified number of placeholders
    in square brackets.
    """
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)
    return len(placeholders) >= N


def verify_bullet_points(text, N, **kwargs):
    """
    Verifies if a text contains exactly N bullet points in markdown format.
    """
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    actual_count = len(bullet_points)
    return actual_count == N


def validate_title(text, **kwargs):
    """
    Validates if text contains a title wrapped in <<brackets>>.
    """
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)
    return len(matches) > 0


def validate_choice(text, options, **kwargs):
    """
    Validates if text matches one of the provided options.
    """
    # Handle both list and string formats for options
    if isinstance(options, str):
        options = [opt.strip() for opt in options.split(",")]
    return any(text in option for option in options)


def validate_highlighted_sections(text, N, **kwargs):
    """
    Validates if text contains at least N highlighted sections (*text*).
    """
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)
    return len(matches) >= N


def validate_sections(text, N, section_splitter, **kwargs):
    """
    Validates if text has exactly N sections when split by delimiter.
    """
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    return len(sections) == N


def validate_json_format(text, **kwargs):
    """
    Validates if text is valid JSON.
    """
    try:
        json.loads(text)
        return True
    except (ValueError, json.JSONDecodeError):
        return False


def validate_repeat_prompt(text, original_prompt, **kwargs):
    """
    Validates if text starts with the original prompt.
    """
    return bool(text.startswith(original_prompt))


def validate_two_responses(text, **kwargs):
    """
    Validates if text contains exactly two distinct responses separated by '******'.
    """
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


def validate_uppercase(text, **kwargs):
    """
    Validates if entire text is uppercase.
    """
    return text == text.upper()


def validate_lowercase(text, **kwargs):
    """
    Validates if entire text is lowercase.
    """
    return text == text.lower()


def validate_frequency_capital_words(text, N, quantifier, **kwargs):
    """
    Validates capital word frequency with quantifier support.
    """
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return len(words) == N
    elif quantifier == "at most":
        return len(words) <= N
    else:
        return False


def validate_end(text, end_phrase, **kwargs):
    """
    Validates if text ends with the specified phrase.
    """
    return bool(text.endswith(end_phrase))


def validate_quotation(text, **kwargs):
    """
    Validates if text is wrapped in double quotes.
    """
    return bool(text.startswith('"') and text.endswith('"'))


def validate_no_commas(text, **kwargs):
    """
    Validates if text contains no commas.
    """
    return "," not in text


# =============================================================================
# Function registry - maps func_name to validation function
# =============================================================================

IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}


class IFEvalVerifier:
    """
    Verifier for instruction-following constraints from IFEval datasets.

    Supports both:
    - allenai/RLVR-GSM-MATH-IF-Mixed-Constraints (func_name format)
    - allenai/RLVR-IFeval (func_name format)

    Each problem has a ground_truth JSON with func_name and parameters.
    Returns 1.0 if constraint is satisfied, 0.0 otherwise.
    """

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout

    def _parse_ground_truth(self, ground_truth: str | dict) -> dict:
        """Parse ground_truth JSON string."""
        if isinstance(ground_truth, dict):
            return ground_truth
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return {}

    def verify(self, response: str, ground_truth: str | dict) -> float:
        """Verify a single response against ground_truth constraints."""
        gt = self._parse_ground_truth(ground_truth)

        func_name = gt.get("func_name")
        if not func_name:
            return 0.0

        func = IF_FUNCTIONS_MAP.get(func_name)
        if func is None:
            # Unknown function - return 0.0
            return 0.0

        # Build kwargs from ground_truth, filtering out None values and func_name
        kwargs = {k: v for k, v in gt.items() if k != "func_name" and v is not None}

        try:
            result = func(response, **kwargs)
            return 1.0 if result else 0.0
        except Exception:
            return 0.0

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem. Returns list of scores."""
        ground_truth = problem.get("ground_truth", "")
        return [self.verify(c, ground_truth) for c in completions]

    async def verify_batch(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float]]:
        """Verify a batch. Returns (scores, durations_ms)."""
        scores = []
        durations = []
        for p, c in zip(problems, completions):
            t0 = time.perf_counter()
            score = self.verify(c, p.get("ground_truth", ""))
            dur = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur)
        return scores, durations

    async def verify_batch_with_timing(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Verify a batch with timing spans. Returns (scores, durations_ms, timing_spans)."""
        scores = []
        durations = []
        timing_spans = []
        offset = 0.0
        for p, c in zip(problems, completions):
            t0 = time.perf_counter()
            score = self.verify(c, p.get("ground_truth", ""))
            dur_ms = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur_ms)
            timing_spans.append((offset, dur_ms))
            offset += dur_ms
        return scores, durations, timing_spans
