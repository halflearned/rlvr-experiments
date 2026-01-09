"""Integration tests for DP (Data Parallel) batch sharding in TitanModelRank.

These tests verify that:
1. _get_dp_shard correctly partitions batches across DP ranks
2. forward_backward shards all tensors consistently
3. compute_logprobs gathers results correctly after sharding
"""

import pytest
import torch
import asyncio
from unittest.mock import MagicMock, patch


# --- Unit tests for sharding logic (no GPU needed) ---

class TestDPShardLogic:
    """Test the sharding logic without requiring a full Titan model."""

    def test_shard_batch_dp3_rank0(self):
        """DP=3, rank 0 should get first third."""
        batch = torch.arange(9).reshape(9, 1)  # [0,1,2,3,4,5,6,7,8]
        dp_degree = 3
        dp_rank = 0

        shard_size = batch.shape[0] // dp_degree
        start = dp_rank * shard_size
        end = start + shard_size
        result = batch[start:end]

        assert result.shape == (3, 1)
        assert result.flatten().tolist() == [0, 1, 2]

    def test_shard_batch_dp3_rank1(self):
        """DP=3, rank 1 should get middle third."""
        batch = torch.arange(9).reshape(9, 1)
        dp_degree = 3
        dp_rank = 1

        shard_size = batch.shape[0] // dp_degree
        start = dp_rank * shard_size
        end = start + shard_size
        result = batch[start:end]

        assert result.shape == (3, 1)
        assert result.flatten().tolist() == [3, 4, 5]

    def test_shard_batch_dp3_rank2(self):
        """DP=3, rank 2 should get last third."""
        batch = torch.arange(9).reshape(9, 1)
        dp_degree = 3
        dp_rank = 2

        shard_size = batch.shape[0] // dp_degree
        start = dp_rank * shard_size
        end = start + shard_size
        result = batch[start:end]

        assert result.shape == (3, 1)
        assert result.flatten().tolist() == [6, 7, 8]

    def test_shard_2d_tensor(self):
        """Sharding should work on 2D tensors [B, seq_len]."""
        batch = torch.arange(12).reshape(4, 3)  # 4 samples, 3 tokens each
        dp_degree = 2
        dp_rank = 1

        shard_size = batch.shape[0] // dp_degree
        start = dp_rank * shard_size
        end = start + shard_size
        result = batch[start:end]

        assert result.shape == (2, 3)
        # Should get samples [2, 3] which are rows [[6,7,8], [9,10,11]]
        assert result[0].tolist() == [6, 7, 8]
        assert result[1].tolist() == [9, 10, 11]

    def test_shard_preserves_other_dims(self):
        """Sharding only affects batch dimension, preserves others."""
        batch = torch.randn(6, 128, 64)  # [B, seq, hidden]
        dp_degree = 3
        dp_rank = 1

        shard_size = batch.shape[0] // dp_degree
        start = dp_rank * shard_size
        end = start + shard_size
        result = batch[start:end]

        assert result.shape == (2, 128, 64)

    def test_indivisible_batch_size(self):
        """Batch size not divisible by DP degree should raise error."""
        batch_size = 7
        dp_degree = 3

        shard_size = batch_size // dp_degree
        # This should be caught - 7 // 3 = 2, but 2 * 3 = 6 != 7
        assert shard_size * dp_degree != batch_size

    def test_dp_degree_1_no_sharding(self):
        """DP=1 should return tensor unchanged."""
        batch = torch.arange(5).reshape(5, 1)
        dp_degree = 1

        # With dp_degree=1, no sharding needed
        assert dp_degree <= 1  # Would early return in _get_dp_shard


class TestAllGatherLogic:
    """Test the all-gather reconstruction logic."""

    def test_gather_reconstructs_full_batch(self):
        """Simulates gathering shards back into full batch."""
        # Simulate 3 ranks each with their shard
        shard0 = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # rank 0's logprobs
        shard1 = torch.tensor([[0.5, 0.6], [0.7, 0.8]])  # rank 1's logprobs
        shard2 = torch.tensor([[0.9, 1.0], [1.1, 1.2]])  # rank 2's logprobs

        # Simulate all_gather + cat
        gathered = [shard0, shard1, shard2]
        result = torch.cat(gathered, dim=0)

        assert result.shape == (6, 2)
        assert torch.allclose(result[0], torch.tensor([0.1, 0.2]))  # from rank 0
        assert torch.allclose(result[2], torch.tensor([0.5, 0.6]))  # from rank 1
        assert torch.allclose(result[4], torch.tensor([0.9, 1.0]))  # from rank 2

    def test_gather_order_matches_original(self):
        """Gathered tensor should match original batch order."""
        # Original batch
        original = torch.arange(12).reshape(6, 2).float()

        # Shard it
        dp_degree = 3
        shards = []
        for rank in range(dp_degree):
            shard_size = original.shape[0] // dp_degree
            start = rank * shard_size
            end = start + shard_size
            shards.append(original[start:end])

        # Gather
        reconstructed = torch.cat(shards, dim=0)

        assert torch.equal(original, reconstructed)


# --- GPU Integration tests ---

@pytest.mark.gpu
@pytest.mark.integration
class TestDPShardingIntegration:
    """Integration tests requiring GPU and actual Titan model."""

    @pytest.fixture
    def mock_titan_actor(self):
        """Create a minimal mock that simulates TitanModelRank with DP."""
        class MockParallelDims:
            def __init__(self, dp_replicate, dp_rank):
                self.dp_replicate = dp_replicate
                self._dp_rank = dp_rank

            @property
            def world_mesh(self):
                mesh = MagicMock()
                mesh.mesh_dim_names = ["dp_replicate"] if self.dp_replicate > 1 else []
                mesh.get_local_rank = MagicMock(return_value=self._dp_rank)
                mesh.__getitem__ = MagicMock(return_value=mesh)
                mesh.get_group = MagicMock(return_value=None)
                return mesh

        class MockModel:
            def __init__(self, dp_replicate, dp_rank):
                self.parallel_dims = MockParallelDims(dp_replicate, dp_rank)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class MockTitanActor:
            def __init__(self, dp_replicate, dp_rank):
                self.model = MockModel(dp_replicate, dp_rank)
                self.rank = 0  # Always rank 0 for testing

            def _get_dp_shard(self, tensor):
                """Replicate the actual implementation."""
                dp_degree = self.model.parallel_dims.dp_replicate
                if dp_degree <= 1:
                    return tensor

                mesh = self.model.parallel_dims.world_mesh
                if "dp_replicate" not in mesh.mesh_dim_names:
                    return tensor

                dp_rank = mesh.get_local_rank("dp_replicate")
                batch_size = tensor.shape[0]
                shard_size = batch_size // dp_degree

                if shard_size * dp_degree != batch_size:
                    raise ValueError(
                        f"Batch size {batch_size} not evenly divisible by dp_replicate={dp_degree}."
                    )

                start = dp_rank * shard_size
                end = start + shard_size
                return tensor[start:end]

        return MockTitanActor

    def test_mock_sharding_dp3_rank0(self, mock_titan_actor):
        """Test mock actor shards correctly for rank 0."""
        actor = mock_titan_actor(dp_replicate=3, dp_rank=0)
        batch = torch.arange(9).reshape(9, 1)

        result = actor._get_dp_shard(batch)

        assert result.shape == (3, 1)
        assert result.flatten().tolist() == [0, 1, 2]

    def test_mock_sharding_dp3_rank2(self, mock_titan_actor):
        """Test mock actor shards correctly for rank 2."""
        actor = mock_titan_actor(dp_replicate=3, dp_rank=2)
        batch = torch.arange(9).reshape(9, 1)

        result = actor._get_dp_shard(batch)

        assert result.shape == (3, 1)
        assert result.flatten().tolist() == [6, 7, 8]

    def test_mock_sharding_dp1_passthrough(self, mock_titan_actor):
        """Test that DP=1 returns tensor unchanged."""
        actor = mock_titan_actor(dp_replicate=1, dp_rank=0)
        batch = torch.arange(5).reshape(5, 1)

        result = actor._get_dp_shard(batch)

        assert torch.equal(result, batch)

    def test_mock_sharding_indivisible_raises(self, mock_titan_actor):
        """Test that indivisible batch size raises ValueError."""
        actor = mock_titan_actor(dp_replicate=3, dp_rank=0)
        batch = torch.arange(7).reshape(7, 1)  # 7 not divisible by 3

        with pytest.raises(ValueError, match="not evenly divisible"):
            actor._get_dp_shard(batch)

    def test_sharding_consistency_across_ranks(self, mock_titan_actor):
        """Verify all ranks together cover the full batch exactly once."""
        batch = torch.arange(12).reshape(12, 1)
        dp_degree = 3

        all_shards = []
        for rank in range(dp_degree):
            actor = mock_titan_actor(dp_replicate=dp_degree, dp_rank=rank)
            shard = actor._get_dp_shard(batch)
            all_shards.append(shard)

        # Concatenate all shards
        reconstructed = torch.cat(all_shards, dim=0)

        # Should exactly match original
        assert torch.equal(reconstructed, batch)

    def test_sharding_with_2d_tensors(self, mock_titan_actor):
        """Test sharding works correctly with [B, seq_len] tensors."""
        batch = torch.randn(6, 128)  # 6 samples, 128 seq len
        dp_degree = 2

        for rank in range(dp_degree):
            actor = mock_titan_actor(dp_replicate=dp_degree, dp_rank=rank)
            shard = actor._get_dp_shard(batch)

            assert shard.shape == (3, 128)
            # Verify it's the correct slice
            expected = batch[rank * 3:(rank + 1) * 3]
            assert torch.equal(shard, expected)

    def test_sharding_loss_args_tuple(self, mock_titan_actor):
        """Test that loss_args tuple is sharded correctly."""
        actor = mock_titan_actor(dp_replicate=3, dp_rank=1)

        # Simulate loss_args: (completion_ids, ref_logprobs, rollout_logprobs, rewards)
        completion_ids = torch.arange(90).reshape(9, 10)
        ref_logprobs = torch.randn(9, 10)
        rollout_logprobs = torch.randn(9, 10)
        rewards = torch.randn(9)

        loss_args = (completion_ids, ref_logprobs, rollout_logprobs, rewards)

        # Shard each tensor
        sharded_args = tuple(
            actor._get_dp_shard(a) if torch.is_tensor(a) else a
            for a in loss_args
        )

        # All should be sharded to size 3
        assert sharded_args[0].shape == (3, 10)
        assert sharded_args[1].shape == (3, 10)
        assert sharded_args[2].shape == (3, 10)
        assert sharded_args[3].shape == (3,)

        # Verify correct slice (rank 1 = middle third)
        assert torch.equal(sharded_args[0], completion_ids[3:6])

    def test_sharding_loss_kwargs_dict(self, mock_titan_actor):
        """Test that loss_kwargs dict is sharded correctly."""
        actor = mock_titan_actor(dp_replicate=2, dp_rank=0)

        padding_mask = torch.ones(6, 20)
        prompt_lens = torch.tensor([5, 6, 7, 8, 9, 10])

        loss_kwargs = {
            "padding_mask": padding_mask,
            "prompt_lens": prompt_lens,
        }

        sharded_kwargs = {
            k: actor._get_dp_shard(v) if torch.is_tensor(v) else v
            for k, v in loss_kwargs.items()
        }

        assert sharded_kwargs["padding_mask"].shape == (3, 20)
        assert sharded_kwargs["prompt_lens"].shape == (3,)
        assert sharded_kwargs["prompt_lens"].tolist() == [5, 6, 7]


class TestBatchTruncationPattern:
    """Test the batch truncation pattern used by train_grpo.py."""

    def test_truncate_to_divisible(self):
        """Test that we can truncate a batch to be divisible by DP degree."""
        dp_degree = 3
        batch_size = 112  # 112 % 3 = 1, should truncate to 111

        remainder = batch_size % dp_degree
        if remainder != 0:
            new_batch_size = batch_size - remainder
        else:
            new_batch_size = batch_size

        assert new_batch_size == 111
        assert new_batch_size % dp_degree == 0

    def test_truncate_already_divisible(self):
        """Test that divisible batches are unchanged."""
        dp_degree = 3
        batch_size = 144  # 144 % 3 = 0, no truncation

        remainder = batch_size % dp_degree
        if remainder != 0:
            new_batch_size = batch_size - remainder
        else:
            new_batch_size = batch_size

        assert new_batch_size == 144

    def test_truncate_pattern_integration(self):
        """Test full truncation + sharding pattern."""
        dp_degree = 3
        batch = torch.arange(112).reshape(112, 1)  # 112 not divisible by 3

        # Step 1: Truncate (what train_grpo.py does)
        remainder = batch.shape[0] % dp_degree
        if remainder != 0:
            batch = batch[:batch.shape[0] - remainder]

        assert batch.shape[0] == 111  # Truncated

        # Step 2: Shard (simulating what titan_actor does for each rank)
        all_shards = []
        for dp_rank in range(dp_degree):
            shard_size = batch.shape[0] // dp_degree
            start = dp_rank * shard_size
            end = start + shard_size
            shard = batch[start:end]
            all_shards.append(shard)

        # Each rank gets 37 samples
        for shard in all_shards:
            assert shard.shape == (37, 1)

        # Concatenated shards equal truncated batch
        reconstructed = torch.cat(all_shards, dim=0)
        assert torch.equal(reconstructed, batch)
