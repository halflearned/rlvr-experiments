"""IFEval verifier for instruction-following constraints.

Implements verification functions for the RLVR-IFeval dataset constraints.
Based on the IFEval benchmark: https://arxiv.org/abs/2311.07911

The ground_truth field contains a JSON object with:
- func_name: Name of the validation function
- Various parameters (N, keyword_list, forbidden_words, etc.)
"""

import json
import re
import time


# --- Validation functions ---
# Each takes (response, **kwargs) and returns bool


def validate_lowercase(response: str, **kwargs) -> bool:
    """Check if response is entirely lowercase."""
    return response == response.lower()


def validate_uppercase(response: str, **kwargs) -> bool:
    """Check if response is entirely uppercase."""
    return response == response.upper()


def validate_no_commas(response: str, **kwargs) -> bool:
    """Check if response contains no commas."""
    return "," not in response


def validate_quotation(response: str, **kwargs) -> bool:
    """Check if response is wrapped in double quotes."""
    response = response.strip()
    return response.startswith('"') and response.endswith('"')


def verify_keyword_frequency(response: str, keyword: str = None, N: int = None, **kwargs) -> bool:
    """Check if keyword appears exactly N times."""
    if keyword is None or N is None:
        return False
    # Case-insensitive count
    count = len(re.findall(re.escape(keyword), response, re.IGNORECASE))
    return count >= N


def verify_keyword_existence(response: str, keyword_list: list = None, **kwargs) -> bool:
    """Check if all keywords in the list exist in the response."""
    if keyword_list is None:
        return True
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keyword_list)


def verify_forbidden_words(response: str, forbidden_words: list = None, **kwargs) -> bool:
    """Check that none of the forbidden words appear in the response."""
    if forbidden_words is None:
        return True
    response_lower = response.lower()
    return not any(word.lower() in response_lower for word in forbidden_words)


def verify_letter_frequency(response: str, letter: str = None, N: int = None, **kwargs) -> bool:
    """Check if a specific letter appears at least N times."""
    if letter is None or N is None:
        return False
    count = response.lower().count(letter.lower())
    return count >= N


def verify_paragraph_count(response: str, N: int = None, section_splitter: str = None, **kwargs) -> bool:
    """Check if response has exactly N paragraphs separated by the splitter."""
    if N is None:
        return False
    splitter = section_splitter or "***"
    # Count paragraphs (number of parts when split)
    parts = [p.strip() for p in response.split(splitter) if p.strip()]
    return len(parts) == N


def verify_sentence_count(response: str, N: int = None, **kwargs) -> bool:
    """Check if response has at least N sentences."""
    if N is None:
        return False
    # Simple sentence counting via punctuation
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) >= N


def verify_word_count(response: str, N: int = None, quantifier: str = None, **kwargs) -> bool:
    """Check word count constraints. quantifier can be 'at_least', 'at_most', 'exactly'."""
    if N is None:
        return False
    words = response.split()
    word_count = len(words)

    if quantifier == "at_least":
        return word_count >= N
    elif quantifier == "at_most":
        return word_count <= N
    elif quantifier == "exactly":
        return word_count == N
    else:
        # Default: at least N words
        return word_count >= N


def verify_bullet_points(response: str, N: int = None, **kwargs) -> bool:
    """Check if response has at least N bullet points."""
    if N is None:
        return False
    # Match lines starting with *, -, or numbered bullets
    bullet_pattern = r'^[\s]*[-*•]|\d+[.)]\s'
    bullets = re.findall(bullet_pattern, response, re.MULTILINE)
    return len(bullets) >= N


def verify_sections(response: str, N: int = None, section_splitter: str = None, **kwargs) -> bool:
    """Check if response has exactly N sections."""
    return verify_paragraph_count(response, N=N, section_splitter=section_splitter, **kwargs)


def verify_end_phrase(response: str, end_phrase: str = None, **kwargs) -> bool:
    """Check if response ends with a specific phrase."""
    if end_phrase is None:
        return False
    return response.strip().endswith(end_phrase)


def verify_postscript(response: str, postscript_marker: str = None, **kwargs) -> bool:
    """Check if response contains a postscript with the given marker."""
    if postscript_marker is None:
        postscript_marker = "P.S."
    return postscript_marker in response


def verify_json_format(response: str, **kwargs) -> bool:
    """Check if response is valid JSON or contains a JSON block."""
    # Try to parse entire response as JSON
    try:
        json.loads(response.strip())
        return True
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, response)
    for match in matches:
        try:
            json.loads(match.strip())
            return True
        except json.JSONDecodeError:
            continue

    return False


def verify_title_format(response: str, **kwargs) -> bool:
    """Check if response contains a title wrapped in <<brackets>>."""
    return bool(re.search(r'<<[^>]+>>', response))


def verify_placeholder(response: str, N: int = None, **kwargs) -> bool:
    """Check if response contains at least N placeholders in [brackets]."""
    if N is None:
        N = 1
    placeholders = re.findall(r'\[[^\]]+\]', response)
    return len(placeholders) >= N


def verify_two_responses(response: str, **kwargs) -> bool:
    """Check if response contains two distinct responses separated by markers."""
    # Look for common separators
    separators = ['***', '---', '===', '\n\n\n']
    for sep in separators:
        parts = [p.strip() for p in response.split(sep) if p.strip()]
        if len(parts) >= 2:
            return True
    return False


def verify_repeat_prompt(response: str, original_prompt: str = None, **kwargs) -> bool:
    """Check if response repeats the original prompt."""
    if original_prompt is None:
        return False
    # Check if original prompt appears in response (case-insensitive)
    return original_prompt.lower() in response.lower()


def verify_highlighted_sections(response: str, N: int = None, **kwargs) -> bool:
    """Check if response has at least N highlighted sections (*text* or **text**)."""
    if N is None:
        N = 1
    highlights = re.findall(r'\*+[^*]+\*+', response)
    return len(highlights) >= N


def verify_capitalized_words(response: str, N: int = None, **kwargs) -> bool:
    """Check if response has at least N fully capitalized words."""
    if N is None:
        return False
    # Match words that are all caps (at least 2 chars)
    caps_words = re.findall(r'\b[A-Z]{2,}\b', response)
    return len(caps_words) >= N


# --- Function registry ---

VALIDATION_FUNCTIONS = {
    # Case constraints
    "validate_lowercase": validate_lowercase,
    "validate_uppercase": validate_uppercase,

    # Punctuation/format constraints
    "validate_no_commas": validate_no_commas,
    "validate_quotation": validate_quotation,
    "verify_json_format": verify_json_format,
    "verify_title_format": verify_title_format,

    # Keyword constraints
    "verify_keyword_frequency": verify_keyword_frequency,
    "verify_keyword_existence": verify_keyword_existence,
    "verify_forbidden_words": verify_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,

    # Structure constraints
    "verify_paragraph_count": verify_paragraph_count,
    "verify_sentence_count": verify_sentence_count,
    "verify_word_count": verify_word_count,
    "verify_bullet_points": verify_bullet_points,
    "verify_sections": verify_sections,
    "verify_highlighted_sections": verify_highlighted_sections,

    # Content constraints
    "verify_end_phrase": verify_end_phrase,
    "verify_postscript": verify_postscript,
    "verify_placeholder": verify_placeholder,
    "verify_two_responses": verify_two_responses,
    "verify_repeat_prompt": verify_repeat_prompt,
    "verify_capitalized_words": verify_capitalized_words,
}


class IFEvalVerifier:
    """
    Verifier for instruction-following constraints from RLVR-IFeval dataset.

    Each problem has a ground_truth JSON with func_name and parameters.
    Returns 1.0 if constraint is satisfied, 0.0 otherwise.
    """

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout

    def _parse_ground_truth(self, ground_truth: str) -> dict:
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

        func = VALIDATION_FUNCTIONS.get(func_name)
        if func is None:
            # Unknown function - return 0.0
            return 0.0

        # Build kwargs from ground_truth, filtering out None values
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
