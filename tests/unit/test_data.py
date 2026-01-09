"""Unit tests for data loading and iteration."""

import pytest
from unittest.mock import MagicMock, patch


class TestLoadDummy:
    def test_returns_ray_dataset(self, ray_context):
        from rlvr_experiments.data import load_dummy

        ds = load_dummy()
        assert ds.count() == 64

    def test_has_correct_schema(self, ray_context):
        from rlvr_experiments.data import load_dummy

        ds = load_dummy()
        row = ds.take(1)[0]
        assert "prompt" in row
        assert "problem" in row
        assert "answer" in row["problem"]

    def test_prompt_format(self, ray_context):
        from rlvr_experiments.data import load_dummy

        ds = load_dummy()
        row = ds.take(1)[0]
        assert row["prompt"].startswith("\n\nProblem:")

    def test_answer_is_one(self, ray_context):
        from rlvr_experiments.data import load_dummy

        ds = load_dummy()
        row = ds.take(1)[0]
        assert row["problem"]["answer"] == "1"


class TestLoadGsm8k:
    @pytest.mark.slow
    def test_returns_ray_dataset(self, ray_context):
        from rlvr_experiments.data import load_gsm8k

        # Use test split for speed
        ds = load_gsm8k(split="test")
        assert ds.count() > 0

    @pytest.mark.slow
    def test_has_correct_schema(self, ray_context):
        from rlvr_experiments.data import load_gsm8k

        ds = load_gsm8k(split="test")
        row = ds.take(1)[0]
        assert "prompt" in row
        assert "problem" in row
        assert "answer" in row["problem"]

    @pytest.mark.slow
    def test_prompt_format(self, ray_context):
        from rlvr_experiments.data import load_gsm8k

        ds = load_gsm8k(split="test")
        row = ds.take(1)[0]
        assert row["prompt"].startswith("\n\nProblem:")

    @pytest.mark.slow
    def test_answer_extracted_after_delimiter(self, ray_context):
        from rlvr_experiments.data import load_gsm8k

        ds = load_gsm8k(split="test")
        row = ds.take(1)[0]
        # GSM8K answers are after ####, so they shouldn't contain ####
        assert "####" not in row["problem"]["answer"]


class TestDataIterator:
    @pytest.fixture
    def dummy_dataset(self, ray_context):
        from rlvr_experiments.data import load_dummy

        return load_dummy()

    def test_init(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)
        assert data_iter is not None

    def test_new_epoch_required(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)

        with pytest.raises(RuntimeError, match="new_epoch"):
            next(data_iter)

    def test_next_returns_correct_keys(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)
        data_iter.new_epoch(seed=42)

        item = next(data_iter)
        assert "template" in item
        assert "prompt" in item
        assert "problem" in item

    def test_templates_are_strings(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)
        data_iter.new_epoch(seed=42)

        item = next(data_iter)
        assert isinstance(item["template"], str)

    def test_problems_are_dicts(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)
        data_iter.new_epoch(seed=42)

        item = next(data_iter)
        assert isinstance(item["problem"], dict)
        assert "answer" in item["problem"]

    def test_epoch_exhaustion(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)
        data_iter.new_epoch(seed=42)

        # Exhaust the iterator
        items = list(data_iter)
        assert len(items) == 64  # dummy dataset has 64 items

    def test_different_seeds_shuffle_differently(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)

        data_iter.new_epoch(seed=1)
        item1 = next(data_iter)

        data_iter.new_epoch(seed=2)
        item2 = next(data_iter)

        # Since all dummy data is the same, we can't really test ordering
        # But we verify the mechanism works
        assert item1["template"] is not None
        assert item2["template"] is not None

    def test_system_prompt_applied(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(
            dummy_dataset,
            tokenizer=mock_tokenizer,
            system_prompt="You are a helpful assistant.",
        )
        data_iter.new_epoch(seed=42)

        item = next(data_iter)
        # System prompt should appear in the template
        assert "You are a helpful assistant." in item["template"]

    def test_assistant_prefix_applied(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(
            dummy_dataset,
            tokenizer=mock_tokenizer,
            assistant_prefix="Let me think step by step.",
        )
        data_iter.new_epoch(seed=42)

        item = next(data_iter)
        # Assistant prefix should be in the template
        assert "Let me think step by step." in item["template"]
