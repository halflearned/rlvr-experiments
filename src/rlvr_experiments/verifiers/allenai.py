"""
Verifiers for AllenAI RLVR dataset.

These verifiers match the extraction and verification logic from the open-instruct codebase:
https://github.com/allenai/open-instruct/blob/main/open_instruct/ground_truth_utils.py
https://github.com/allenai/open-instruct/blob/main/open_instruct/math_utils.py

Key differences from our other verifiers:
- GSM8K: Extracts last number from response
- MATH: Uses flexible extraction (boxed, minerva format, dollar signs) + hendrycks equivalence
"""

import re
import time
from typing import Optional


# =============================================================================
# AllenAI GSM8K Verifier
# =============================================================================

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the last number from the response.

    AllenAI's GSM8K verifier extracts the last number in the response,
    which handles both "So the answer is X." and other formats.

    Based on ground_truth_utils.py but with fix for negative integers.
    """
    # Remove commas from numbers
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Match integers and decimals, with optional sign
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", text)
    if numbers:
        return numbers[-1]
    return None


def normalize_gsm8k(s: str) -> Optional[str]:
    """Normalize GSM8K answer for comparison."""
    if s is None:
        return None
    s = str(s).strip()
    # Remove commas
    s = s.replace(",", "")
    return s


class AllenAIGSM8KVerifier:
    """
    Verifier for AllenAI's GSM8K dataset.

    Extracts the last number from the response and compares to ground truth.
    This matches the AllenAI open-instruct verification logic.
    """

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if extracted answer matches target, 0.0 otherwise."""
        try:
            pred = extract_gsm8k_answer(response)
            pred_norm = normalize_gsm8k(pred)
            gold_norm = normalize_gsm8k(target)

            if pred_norm is None or gold_norm is None:
                return 0.0

            # Direct string match
            if pred_norm == gold_norm:
                return 1.0

            # Try numeric comparison
            try:
                pred_num = float(pred_norm)
                gold_num = float(gold_norm)
                # Compare as integers if both are whole numbers
                if pred_num == int(pred_num) and gold_num == int(gold_num):
                    return 1.0 if int(pred_num) == int(gold_num) else 0.0
                return 1.0 if pred_num == gold_num else 0.0
            except (ValueError, TypeError):
                pass

            return 0.0
        except Exception:
            return 0.0

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem."""
        target = problem.get("ground_truth", "")
        return [self.verify(c, target) for c in completions]

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
        """Verify a batch with timing spans."""
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


# =============================================================================
# AllenAI MATH Verifier
# From: https://github.com/allenai/open-instruct/blob/main/open_instruct/math_utils.py
# =============================================================================

# Substitutions for normalization (from math_utils.py)
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{googol)}$, and target", ""),
    ("\\text{googol}", "10^{100}"),
    ("\\text{, }", ","),
    ("\\text{,}", ","),
    ("}", ""),
    ("{", ""),
    ("'", ""),
    ('"', ""),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "hours", "km",
    "units", "\\ldots", ", ",
    "\\%", "\\## feet", "\\## meters", "\\## miles", "\\## gallons",
    "\\text{s}", "\\text{.}", "\\text{\ns]}am am am",
    "\\text{}^am am am",
    "-", "^{\\circ}", "^\\circ",
    "\\text{degrees}", "\\text{googol}",
]


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} content from the string."""
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper from answer."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]


def get_unnormalized_answer(text: str) -> str:
    """Extract answer using Minerva format."""
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


def normalize_final_answer(final_answer: str) -> str:
    """Normalize MATH answer for comparison."""
    if final_answer is None:
        return ""

    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Remove $...$ wrappers
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"\3", final_answer)
    # Remove text wrappers
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
    # Fix fractions and sqrt
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Remove commas from pure numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def remove_right_units(string: str) -> str:
    """Remove units from the right side of the string."""
    if "\\text{" in string:
        splits = string.split("\\text{")
        return splits[0]
    return string


def fix_sqrt(string: str) -> str:
    """Fix sqrt formatting."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def fix_fracs(string: str) -> str:
    """Fix fraction formatting."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string: str) -> str:
    """Convert a/b to \\frac{a}{b}."""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except (ValueError, AssertionError):
        return string


def strip_string(string: str) -> str:
    """Strip and normalize a string for comparison (Hendrycks style)."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # Handle x = answer format
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)

    # Handle 0.5 -> 1/2 conversion
    if string == "0.5":
        string = "\\frac{1}{2}"

    string = fix_a_slash_b(string)
    return string


def hendrycks_is_equiv(str1: Optional[str], str2: Optional[str]) -> bool:
    """Check equivalence using Hendrycks MATH style normalization."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def is_equiv_sympy(x1: str, x2: str, timeout_seconds: float = 5.0) -> bool:
    """Check equivalence using sympy (with timeout)."""
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify

        try:
            parsed_x1 = parse_latex(x1)
            parsed_x2 = parse_latex(x2)
        except Exception:
            return False

        try:
            diff = parsed_x1 - parsed_x2
        except TypeError:
            return False

        try:
            return simplify(diff) == 0
        except Exception:
            return False
    except Exception:
        return False


class AllenAIMathVerifier:
    """
    Verifier for AllenAI's MATH dataset with flexible extraction.

    Uses the same multi-strategy approach as AllenAI's MathVerifier:
    1. Extract from \\boxed{} content
    2. Extract via Minerva format ("Final Answer: The final answer is X.")
    3. Extract content between last two $ signs
    4. Fall back to normalized full output

    Compares using both is_equiv (sympy) and hendrycks_is_equiv.
    """

    def __init__(self, use_sympy: bool = True):
        """
        Args:
            use_sympy: If True, use sympy for symbolic equivalence checking.
                       If False, use only string comparison (faster but less accurate).
        """
        self.use_sympy = use_sympy

    def _extract_all_answers(self, raw_answer: str) -> list[str]:
        """Extract candidate answers using multiple strategies."""
        all_answers = []

        # Strategy 1: Boxed answer
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
                all_answers.append(boxed_answer)
            except AssertionError:
                pass

        # Strategy 2: Minerva format
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)

        # Strategy 3: Between last two $ signs
        if not all_answers:
            dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
            if len(dollars) > 1:
                answer = normalize_final_answer(raw_answer[dollars[-2] + 1:dollars[-1]])
                all_answers.append(answer)

        # Strategy 4: Full normalized output (fallback)
        if not all_answers:
            all_answers.append(normalize_final_answer(raw_answer))
            all_answers.append(raw_answer)

        return all_answers

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if any extracted answer matches target, 0.0 otherwise."""
        try:
            candidates = self._extract_all_answers(response)
            label = target

            for answer in candidates:
                # Try hendrycks equivalence first (string-based, fast)
                if hendrycks_is_equiv(answer, label):
                    return 1.0

                # Try sympy equivalence if enabled
                if self.use_sympy and is_equiv_sympy(answer, label):
                    return 1.0

            return 0.0
        except Exception:
            return 0.0

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem."""
        target = problem.get("ground_truth", "")
        return [self.verify(c, target) for c in completions]

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
        """Verify a batch with timing spans."""
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
