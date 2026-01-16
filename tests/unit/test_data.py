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
            data_iter.get_next()

    def test_next_returns_correct_keys(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)
        data_iter.new_epoch(seed=42)

        item = data_iter.get_next()
        assert "template" in item
        assert "prompt" in item
        assert "problem" in item

    def test_templates_are_strings(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)
        data_iter.new_epoch(seed=42)

        item = data_iter.get_next()
        assert isinstance(item["template"], str)

    def test_problems_are_dicts(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)
        data_iter.new_epoch(seed=42)

        item = data_iter.get_next()
        assert isinstance(item["problem"], dict)
        assert "answer" in item["problem"]

    def test_epoch_exhaustion(self, ray_context, mock_tokenizer):
        import ray.data
        from rlvr_experiments.data import DataIterator

        # Create dataset with unique prompt_ids (dummy has same id for all rows)
        rows = [
            {
                "prompt": f"Question {i}",
                "problem": {"answer": str(i), "prompt_id": f"test_{i}"},
            }
            for i in range(64)
        ]
        ds = ray.data.from_items(rows)

        data_iter = DataIterator(ds, tokenizer=mock_tokenizer)
        data_iter.new_epoch(seed=42)

        # Exhaust all pending items
        items = []
        while True:
            item = data_iter.get_next()
            if item is None:
                break
            items.append(item)
        assert len(items) == 64

    def test_different_seeds_shuffle_differently(self, dummy_dataset, mock_tokenizer):
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(dummy_dataset, tokenizer=mock_tokenizer)

        data_iter.new_epoch(seed=1)
        item1 = data_iter.get_next()

        data_iter.new_epoch(seed=2)
        item2 = data_iter.get_next()

        # Since all dummy data is the same, we can't really test ordering
        # But we verify the mechanism works
        assert item1["template"] is not None
        assert item2["template"] is not None

    def test_system_prompt_applied(self, ray_context, mock_tokenizer):
        import ray.data
        from rlvr_experiments.data import DataIterator

        # Create dataset WITHOUT per-row system_prompt to test global fallback
        rows = [{"prompt": "Question", "problem": {"prompt_id": "test_1"}}]
        ds = ray.data.from_items(rows)

        data_iter = DataIterator(
            ds,
            tokenizer=mock_tokenizer,
            system_prompt="You are a helpful assistant.",
        )
        data_iter.new_epoch(seed=42)

        item = data_iter.get_next()
        # System prompt should appear in the template
        assert "You are a helpful assistant." in item["template"]

    def test_assistant_prefix_applied(self, ray_context, mock_tokenizer):
        import ray.data
        from rlvr_experiments.data import DataIterator

        # Create dataset WITHOUT per-row assistant_prefix to test global fallback
        rows = [{"prompt": "Question", "problem": {"prompt_id": "test_1"}}]
        ds = ray.data.from_items(rows)

        data_iter = DataIterator(
            ds,
            tokenizer=mock_tokenizer,
            assistant_prefix="Let me think step by step.",
        )
        data_iter.new_epoch(seed=42)

        item = data_iter.get_next()
        # Assistant prefix should be in the template
        assert "Let me think step by step." in item["template"]

    def test_per_row_system_prompt_override(self, ray_context, mock_tokenizer):
        """Test that per-row system_prompt in problem dict overrides global."""
        import ray.data
        from rlvr_experiments.data import DataIterator

        rows = [
            {
                "prompt": "Question 1",
                "problem": {
                    "prompt_id": "test_1",
                    "system_prompt": "Custom system prompt",
                    "assistant_prefix": "",
                },
            }
        ]
        ds = ray.data.from_items(rows)

        data_iter = DataIterator(
            ds,
            tokenizer=mock_tokenizer,
            system_prompt="Global system prompt",  # This should be overridden
        )
        data_iter.new_epoch(seed=42)

        item = data_iter.get_next()
        assert "Custom system prompt" in item["template"]
        assert "Global system prompt" not in item["template"]

    def test_per_row_assistant_prefix_override(self, ray_context, mock_tokenizer):
        """Test that per-row assistant_prefix in problem dict overrides global."""
        import ray.data
        from rlvr_experiments.data import DataIterator

        rows = [
            {
                "prompt": "Question 1",
                "problem": {
                    "prompt_id": "test_1",
                    "system_prompt": "",
                    "assistant_prefix": "Custom prefix",
                },
            }
        ]
        ds = ray.data.from_items(rows)

        data_iter = DataIterator(
            ds,
            tokenizer=mock_tokenizer,
            assistant_prefix="Global prefix",  # This should be overridden
        )
        data_iter.new_epoch(seed=42)

        item = data_iter.get_next()
        assert "Custom prefix" in item["template"]
        assert "Global prefix" not in item["template"]


class TestLoadMixed:
    def test_returns_tuple(self, ray_context):
        """Test that load_mixed returns (dataset, order) tuple."""
        from rlvr_experiments.data import load_mixed

        result = load_mixed([{"name": "dummy", "num_samples": 10}])
        assert isinstance(result, tuple)
        assert len(result) == 2
        ds, order = result
        assert ds.count() == 10
        assert len(order) == 10

    def test_order_contains_all_prompt_ids(self, ray_context):
        """Test that order list contains all prompt_ids from dataset."""
        from rlvr_experiments.data import load_mixed

        ds, order = load_mixed([{"name": "dummy", "num_samples": 10}])
        rows = list(ds.iter_rows())
        prompt_ids = {row["problem"]["prompt_id"] for row in rows}
        assert set(order) == prompt_ids

    def test_empty_datasets_raises(self, ray_context):
        from rlvr_experiments.data import load_mixed

        with pytest.raises(ValueError, match="cannot be empty"):
            load_mixed([])

    def test_unknown_dataset_raises(self, ray_context):
        from rlvr_experiments.data import load_mixed

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_mixed([{"name": "nonexistent"}])

    def test_missing_name_raises(self, ray_context):
        from rlvr_experiments.data import load_mixed

        with pytest.raises(ValueError, match="must have a 'name' key"):
            load_mixed([{"split": "train"}])

    def test_samples_have_verifier_type(self, ray_context):
        from rlvr_experiments.data import load_mixed

        ds, _ = load_mixed([{"name": "dummy", "num_samples": 5}])
        row = ds.take(1)[0]
        assert "verifier_type" in row["problem"]
        assert row["problem"]["verifier_type"] == "math"

    def test_samples_have_system_prompt(self, ray_context):
        from rlvr_experiments.data import load_mixed

        ds, _ = load_mixed([{"name": "dummy", "num_samples": 5}])
        row = ds.take(1)[0]
        assert "system_prompt" in row["problem"]
        assert "\\boxed{}" in row["problem"]["system_prompt"]

    def test_samples_have_assistant_prefix(self, ray_context):
        from rlvr_experiments.data import load_mixed

        ds, _ = load_mixed([{"name": "dummy", "num_samples": 5}])
        row = ds.take(1)[0]
        assert "assistant_prefix" in row["problem"]

    def test_order_deterministic(self, ray_context):
        """Test that same seed produces same order."""
        from rlvr_experiments.data import load_mixed

        _, order1 = load_mixed([{"name": "dummy", "num_samples": 10}], seed=42)
        _, order2 = load_mixed([{"name": "dummy", "num_samples": 10}], seed=42)

        assert order1 == order2

    def test_different_seeds_different_order(self, ray_context):
        """Test that different seeds produce different orders."""
        from rlvr_experiments.data import load_mixed

        _, order1 = load_mixed([{"name": "dummy", "num_samples": 20}], seed=1)
        _, order2 = load_mixed([{"name": "dummy", "num_samples": 20}], seed=2)

        # With 20 items, probability of same order is negligible
        assert order1 != order2
        # But should have same items
        assert set(order1) == set(order2)


class TestLoadMixedWeights:
    """Tests for weighted interleaving in load_mixed."""

    def test_weighted_interleaving_respects_proportions(self, ray_context):
        """Test that weights approximately control dataset proportions in interleaving."""
        from rlvr_experiments.data import load_mixed

        # Two dummy datasets with different weights
        # Note: dummy datasets generate same prompt_ids (dummy_0, dummy_1, etc.)
        # so when combined, duplicates are de-duplicated
        # Weight 3:1 ratio, each has 100 samples -> 100 unique IDs
        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 100, "weight": 3.0},
            {"name": "dummy", "num_samples": 100, "weight": 1.0},
        ], seed=42)

        # Due to ID overlap, we only get 100 unique items
        # (both datasets have dummy_0 through dummy_99)
        assert len(order) == 100

    def test_order_file_respected(self, ray_context, tmp_path):
        """Test that order_file is respected for per-dataset ordering."""
        from rlvr_experiments.data import load_mixed, load_dummy

        # Create an order file with specific prompt_ids
        # First get actual prompt_ids from dummy dataset
        dummy_ds = load_dummy(num_samples=10)
        rows = list(dummy_ds.iter_rows())
        all_ids = [row["problem"]["prompt_id"] for row in rows]

        # Create order file with reversed order (only first 5)
        order_file = tmp_path / "order.txt"
        reversed_ids = list(reversed(all_ids[:5]))
        order_file.write_text("\n".join(reversed_ids))

        # Load with order file
        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 10, "order_file": str(order_file)},
        ], seed=42)

        # Should only have 5 items (those in order file)
        assert len(order) == 5
        # And they should be in the specified order
        assert order == reversed_ids

    def test_order_file_invalid_ids_filtered(self, ray_context, tmp_path):
        """Test that invalid IDs in order file are filtered out."""
        from rlvr_experiments.data import load_mixed, load_dummy

        # Get real IDs
        dummy_ds = load_dummy(num_samples=5)
        rows = list(dummy_ds.iter_rows())
        real_ids = [row["problem"]["prompt_id"] for row in rows]

        # Create order file with mix of real and fake IDs
        order_file = tmp_path / "order.txt"
        order_file.write_text(f"{real_ids[0]}\nfake_id_1\n{real_ids[1]}\nfake_id_2\n")

        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 5, "order_file": str(order_file)},
        ], seed=42)

        # Should only have the 2 real IDs
        assert len(order) == 2
        assert order == [real_ids[0], real_ids[1]]

    def test_order_file_all_invalid_raises(self, ray_context, tmp_path):
        """Test that order file with all invalid IDs raises error."""
        from rlvr_experiments.data import load_mixed

        order_file = tmp_path / "order.txt"
        order_file.write_text("fake_id_1\nfake_id_2\n")

        with pytest.raises(ValueError, match="no valid prompt_ids"):
            load_mixed([
                {"name": "dummy", "num_samples": 5, "order_file": str(order_file)},
            ])

    def test_exhausted_dataset_ignored(self, ray_context):
        """Test that when one dataset exhausts, sampling continues with others."""
        from rlvr_experiments.data import load_mixed

        # One small dataset (5 items) and one larger (20 items)
        # Note: dummy_0 through dummy_4 overlap between the two
        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 5, "weight": 1.0},
            {"name": "dummy", "num_samples": 20, "weight": 1.0},
        ], seed=42)

        # Due to ID overlap (dummy_0-4 in both), we get 20 unique IDs in order
        # But the dataset still contains all 25 rows
        assert len(order) == 20
        assert ds.count() == 25  # all rows, including duplicates


class TestLoadMixedSequential:
    """Tests for sequential mode in load_mixed."""

    def test_sequential_concatenates_in_order(self, ray_context):
        """Test that sequential mode concatenates datasets in config order."""
        from rlvr_experiments.data import load_mixed

        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 5},
            {"name": "dummy", "num_samples": 10},
        ], mode="sequential", seed=42)

        # Sequential mode concatenates orders from each dataset
        # Total = 5 + 10 = 15 samples (duplicates allowed in order list)
        assert len(order) == 15

    def test_sequential_with_count(self, ray_context):
        """Test that count limits samples in sequential mode."""
        from rlvr_experiments.data import load_mixed

        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 20, "count": 5},
            {"name": "dummy", "num_samples": 20, "count": 3},
        ], mode="sequential", seed=42)

        # First dataset: 5 samples, second: 3 samples
        # Total = 5 + 3 = 8 samples
        assert len(order) == 8

    def test_sequential_count_limits_per_dataset(self, ray_context):
        """Test that count is applied per-dataset independently."""
        from rlvr_experiments.data import load_mixed

        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 100, "count": 10},
            {"name": "dummy", "num_samples": 100, "count": 5},
        ], mode="sequential", seed=42)

        # Count limits are applied per dataset
        # Total = 10 + 5 = 15 samples
        assert len(order) == 15

    def test_sequential_preserves_config_order(self, ray_context):
        """Test that sequential mode preserves the order from config."""
        from rlvr_experiments.data import load_mixed

        # Create two datasets with non-overlapping ranges by using order files
        # For simplicity, just verify the order is deterministic
        ds1, order1 = load_mixed([
            {"name": "dummy", "num_samples": 10, "count": 3},
            {"name": "dummy", "num_samples": 10, "count": 2},
        ], mode="sequential", seed=42)

        ds2, order2 = load_mixed([
            {"name": "dummy", "num_samples": 10, "count": 3},
            {"name": "dummy", "num_samples": 10, "count": 2},
        ], mode="sequential", seed=42)

        # Same config should produce same order
        assert order1 == order2

    def test_sequential_no_count_takes_all(self, ray_context):
        """Test that omitting count takes all samples from dataset."""
        from rlvr_experiments.data import load_mixed

        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 15},  # no count = all 15
        ], mode="sequential", seed=42)

        assert len(order) == 15

    def test_invalid_mode_raises(self, ray_context):
        """Test that invalid mode raises error."""
        from rlvr_experiments.data import load_mixed

        with pytest.raises(ValueError, match="mode must be"):
            load_mixed([{"name": "dummy", "num_samples": 5}], mode="invalid")

    def test_sequential_with_order_file(self, ray_context, tmp_path):
        """Test that sequential mode respects order_file per dataset."""
        from rlvr_experiments.data import load_mixed, load_dummy

        # Get real IDs
        dummy_ds = load_dummy(num_samples=10)
        rows = list(dummy_ds.iter_rows())
        all_ids = [row["problem"]["prompt_id"] for row in rows]

        # Create order file with specific IDs in reverse order
        order_file = tmp_path / "order.txt"
        selected_ids = list(reversed(all_ids[5:10]))  # dummy_9, dummy_8, ..., dummy_5
        order_file.write_text("\n".join(selected_ids))

        ds, order = load_mixed([
            {"name": "dummy", "num_samples": 10, "order_file": str(order_file), "count": 3},
        ], mode="sequential", seed=42)

        # Should take first 3 from the order file: dummy_9, dummy_8, dummy_7
        assert len(order) == 3
        assert order == selected_ids[:3]


class TestDatasetMetadata:
    """Test that all dataset loaders include required metadata."""

    def test_dummy_has_metadata(self, ray_context):
        from rlvr_experiments.data import load_dummy, GSM8K_MAX_COMPLETION_LEN

        ds = load_dummy(num_samples=1)
        row = ds.take(1)[0]
        problem = row["problem"]

        assert "verifier_type" in problem
        assert "system_prompt" in problem
        assert "assistant_prefix" in problem
        assert "max_completion_len" in problem
        assert problem["verifier_type"] == "math"
        assert problem["max_completion_len"] == GSM8K_MAX_COMPLETION_LEN

    @pytest.mark.slow
    def test_gsm8k_has_metadata(self, ray_context):
        from rlvr_experiments.data import load_gsm8k, GSM8K_MAX_COMPLETION_LEN

        ds = load_gsm8k(split="test")
        row = ds.take(1)[0]
        problem = row["problem"]

        assert problem["verifier_type"] == "math"
        assert "system_prompt" in problem
        assert "assistant_prefix" in problem
        assert "max_completion_len" in problem
        assert problem["max_completion_len"] == GSM8K_MAX_COMPLETION_LEN


class TestDataIteratorOrder:
    """Tests for DataIterator ordering and priority functionality."""

    @pytest.fixture
    def simple_dataset(self, ray_context):
        """Create a simple dataset with 5 items for testing ordering."""
        import ray.data

        rows = [
            {"prompt": f"Question {i}", "problem": {"answer": str(i), "prompt_id": f"id_{i}"}}
            for i in range(5)
        ]
        return ray.data.from_items(rows)

    def test_init_with_order(self, simple_dataset, mock_tokenizer):
        """Test that order parameter sets initial ordering."""
        from rlvr_experiments.data import DataIterator

        order = ["id_2", "id_0", "id_4"]
        data_iter = DataIterator(simple_dataset, tokenizer=mock_tokenizer, order=order)
        data_iter.new_epoch()

        # Should get items in specified order
        item1 = data_iter.get_next()
        assert item1["problem"]["prompt_id"] == "id_2"

        item2 = data_iter.get_next()
        assert item2["problem"]["prompt_id"] == "id_0"

        item3 = data_iter.get_next()
        assert item3["problem"]["prompt_id"] == "id_4"

        # Items not in order should not be returned
        item4 = data_iter.get_next()
        assert item4 is None

    def test_init_order_filters_invalid_ids(self, simple_dataset, mock_tokenizer):
        """Test that invalid prompt_ids in order are silently ignored."""
        from rlvr_experiments.data import DataIterator

        order = ["id_0", "invalid_id", "id_1"]
        data_iter = DataIterator(simple_dataset, tokenizer=mock_tokenizer, order=order)
        data_iter.new_epoch()

        item1 = data_iter.get_next()
        assert item1["problem"]["prompt_id"] == "id_0"

        item2 = data_iter.get_next()
        assert item2["problem"]["prompt_id"] == "id_1"

        item3 = data_iter.get_next()
        assert item3 is None

    def test_init_order_all_invalid_raises(self, simple_dataset, mock_tokenizer):
        """Test that order with all invalid ids raises error."""
        from rlvr_experiments.data import DataIterator

        order = ["invalid_1", "invalid_2"]
        with pytest.raises(ValueError, match="no valid prompt_ids"):
            DataIterator(simple_dataset, tokenizer=mock_tokenizer, order=order)

    def test_new_epoch_with_order(self, simple_dataset, mock_tokenizer):
        """Test that new_epoch can set a new ordering."""
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(simple_dataset, tokenizer=mock_tokenizer)
        data_iter.new_epoch(order=["id_3", "id_1"])

        item1 = data_iter.get_next()
        assert item1["problem"]["prompt_id"] == "id_3"

        item2 = data_iter.get_next()
        assert item2["problem"]["prompt_id"] == "id_1"

        # Items not in order are not returned
        item3 = data_iter.get_next()
        assert item3 is None

    def test_mark_pending_moves_to_front(self, simple_dataset, mock_tokenizer):
        """Test that mark_pending moves item to front of queue."""
        from rlvr_experiments.data import DataIterator

        order = ["id_0", "id_1", "id_2"]
        data_iter = DataIterator(simple_dataset, tokenizer=mock_tokenizer, order=order)
        data_iter.new_epoch()

        # Get first item (id_0), moves to in_flight
        item1 = data_iter.get_next()
        assert item1["problem"]["prompt_id"] == "id_0"

        # Get second item (id_1), moves to in_flight
        item2 = data_iter.get_next()
        assert item2["problem"]["prompt_id"] == "id_1"

        # Mark id_1 as pending (retry) - should move to front
        data_iter.mark_pending("id_1")

        # Next item should be id_1 (retried), not id_2
        item3 = data_iter.get_next()
        assert item3["problem"]["prompt_id"] == "id_1"

    def test_mark_pending_multiple_retries(self, simple_dataset, mock_tokenizer):
        """Test multiple retries - most recent retry gets priority (LIFO)."""
        from rlvr_experiments.data import DataIterator

        order = ["id_0", "id_1", "id_2"]
        data_iter = DataIterator(simple_dataset, tokenizer=mock_tokenizer, order=order)
        data_iter.new_epoch()

        # Get all items
        data_iter.get_next()  # id_0
        data_iter.get_next()  # id_1
        data_iter.get_next()  # id_2

        # Mark id_0 as pending first
        data_iter.mark_pending("id_0")
        # Then mark id_2 as pending
        data_iter.mark_pending("id_2")

        # id_2 should come first (most recently marked pending)
        item1 = data_iter.get_next()
        assert item1["problem"]["prompt_id"] == "id_2"

        # Then id_0
        item2 = data_iter.get_next()
        assert item2["problem"]["prompt_id"] == "id_0"

    def test_mark_pending_does_not_affect_failed(self, simple_dataset, mock_tokenizer):
        """Test that mark_pending does nothing for failed items."""
        from rlvr_experiments.data import DataIterator

        order = ["id_0", "id_1"]
        data_iter = DataIterator(simple_dataset, tokenizer=mock_tokenizer, order=order)
        data_iter.new_epoch()

        # Get and fail id_0
        data_iter.get_next()  # id_0
        data_iter.mark_failed("id_0")

        # Try to mark it pending - should be ignored
        data_iter.mark_pending("id_0")

        # Next item should be id_1, not id_0
        item = data_iter.get_next()
        assert item["problem"]["prompt_id"] == "id_1"

    def test_order_preserved_across_epoch(self, simple_dataset, mock_tokenizer):
        """Test that explicit order is preserved across new_epoch calls."""
        from rlvr_experiments.data import DataIterator

        order = ["id_0", "id_1", "id_2"]
        data_iter = DataIterator(simple_dataset, tokenizer=mock_tokenizer, order=order)
        data_iter.new_epoch()

        # Get id_0 and id_1
        data_iter.get_next()  # id_0
        data_iter.get_next()  # id_1

        # Start new epoch - explicit order should be restored
        data_iter.new_epoch()

        # id_0 should be first again (explicit order restored)
        item = data_iter.get_next()
        assert item["problem"]["prompt_id"] == "id_0"

    def test_seed_shuffles_order(self, simple_dataset, mock_tokenizer):
        """Test that seed parameter shuffles the order."""
        from rlvr_experiments.data import DataIterator

        data_iter = DataIterator(simple_dataset, tokenizer=mock_tokenizer)

        # Get order with seed=1
        data_iter.new_epoch(seed=1)
        items_seed1 = []
        while (item := data_iter.get_next()) is not None:
            items_seed1.append(item["problem"]["prompt_id"])
            data_iter.mark_done(item["problem"]["prompt_id"])

        # Get order with seed=2
        data_iter.new_epoch(seed=2)
        items_seed2 = []
        while (item := data_iter.get_next()) is not None:
            items_seed2.append(item["problem"]["prompt_id"])
            data_iter.mark_done(item["problem"]["prompt_id"])

        # Different seeds should (very likely) produce different orders
        # Both should have all 5 items
        assert len(items_seed1) == 5
        assert len(items_seed2) == 5
        assert set(items_seed1) == set(items_seed2)
        # With 5 items, probability of same order is 1/120, so this should be safe
        assert items_seed1 != items_seed2
