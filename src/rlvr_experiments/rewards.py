import random
import re


class MathReward:
    """
    Reward class for evaluating math correctness.
    Copied from torchforge.
    """

    def __init__(self, tolerance: float = 1e-6, partial_credit: float = 0.1):
        self.tolerance = tolerance
        self.partial_credit = partial_credit

    def __call__(self, prompt: str, response: str, target: str) -> float:
        """Compute math correctness reward."""
        target_number = self._to_float(target)
        if target_number is None:
            return 0.0

        # Look for answer in <answer></answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if answer_match:
            model_answer = self._to_float(answer_match.group(1).strip())
            if (
                model_answer is not None
                and abs(target_number - model_answer) < self.tolerance
            ):
                return 1.0  # Correct answer

        # Check for partial credit: target number appears elsewhere in response
        response_without_answer_tags = re.sub(
            r"<answer>.*?</answer>", "", response, flags=re.DOTALL
        )
        # Convert to int if it's a whole number to avoid "117.0" vs "117" mismatch
        target_str = (
            str(int(target_number))
            if target_number.is_integer()
            else str(target_number)
        )
        if target_str in response_without_answer_tags:
            return self.partial_credit

        return 0.0  # No match

    def _to_float(self, text: str) -> float | None:
        """Convert text to float, return None if invalid."""
        try:
            # Remove common non-numeric characters like $, commas, etc.
            cleaned_text = re.sub(r"[$,\s]", "", text.strip())
            return float(cleaned_text)
        except (ValueError, AttributeError):
            return None

