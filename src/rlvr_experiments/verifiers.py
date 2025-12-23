import re
from typing import Optional

import torch


class MathVerifier:
    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def _extract_boxed_expression(self, response: str) -> Optional[str]:
        """Extract the contents of the first \boxed{...} with balanced braces."""
        match = re.search(r"\\boxed\{", response)
        if not match:
            return None
        start = match.end()
        depth = 1
        idx = start
        while idx < len(response) and depth > 0:
            char = response[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1
        if depth != 0:
            return None
        return response[start : idx - 1].strip()

    def _extract_answer_span(self, response: str) -> Optional[str]:
        tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if tag_match:
            return tag_match.group(1).strip()
        return self._extract_boxed_expression(response)

    def parse_number(self, answer_str: str) -> float:
        """Parse a string that might be a number, fraction, or LaTeX fraction."""
        answer_str = answer_str.strip()

        # Handle latex like \frac{a}{b} or \dfrac{a}{b}
        frac_match = re.match(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", answer_str)
        if frac_match:
            num = self.parse_number(frac_match.group(1))
            denom = self.parse_number(frac_match.group(2))
            return num / denom

        # Handle simple fractions like "1/3"
        if '/' in answer_str:
            num, denom = answer_str.split('/')
            return float(num.replace(",", "")) / float(denom.replace(",", ""))

        # Handle plain numbers (remove commas for numbers like "27,000")
        return float(answer_str.replace(",", ""))

    def verify(self, response: str, target: str) -> float:
        target_number = float(target.replace(",", ""))  # TODO: handle better
        answer_span = self._extract_answer_span(response)
        if answer_span is None:
            return 0.0
        try:
            model_answer = self.parse_number(answer_span)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0

        if abs(target_number - model_answer) < self.tolerance:
            return 1.0
        return 0.0

    def verify_batch(self, *, responses=None, targets=None, return_dtype=None):
        results = [
            self.verify(r, t) for r, t in zip(responses, targets)
        ]
        if return_dtype is not None:
            return torch.tensor(results, dtype=return_dtype)
        return results
