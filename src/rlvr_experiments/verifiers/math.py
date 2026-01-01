import re


def _extract_boxed(text: str) -> str | None:
    """Extract contents of \\boxed{...} with balanced braces."""
    match = re.search(r"\\boxed\{", text)
    if not match:
        return None
    start = match.end()
    depth = 1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def _extract_answer(text: str) -> str | None:
    """Extract answer from <answer> tags or \\boxed{}."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return _extract_boxed(text)


def _parse_number(s: str) -> float:
    """Parse number, fraction, or LaTeX fraction."""
    s = s.strip().replace(",", "")

    # LaTeX fractions: \frac{a}{b} or \dfrac{a}{b}
    m = re.match(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", s)
    if m:
        return _parse_number(m.group(1)) / _parse_number(m.group(2))

    # Simple fractions: a/b
    if "/" in s:
        num, denom = s.split("/", 1)
        return float(num) / float(denom)

    return float(s)


class MathVerifier:
    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if response matches target, 0.0 otherwise."""
        try:
            target_num = _parse_number(target)
        except (ValueError, ZeroDivisionError):
            return 0.0

        answer = _extract_answer(response)
        if answer is None:
            return 0.0

        try:
            parsed = _parse_number(answer)
        except (ValueError, ZeroDivisionError):
            return 0.0

        return 1.0 if abs(target_num - parsed) < self.tolerance else 0.0

    def verify_batch(self, responses: list[str], targets: list[str]) -> list[float]:
        """Verify a batch of responses against targets."""
        return [self.verify(r, t) for r, t in zip(responses, targets)]
