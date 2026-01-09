"""Integration tests for weight synchronization.

These tests verify:
1. Weight sync to vLLM is idempotent (syncing same weights twice = same output)
2. vLLM outputs change after training + syncing
3. Reference model logprobs change after training + syncing
4. Trainer logprobs change after training steps

Runtime is loaded once per module to avoid repeated startup overhead.
"""

import os
import asyncio
import pytest
import ray
import torch
import yaml

from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.losses import GRPOLoss


# --- Module-level fixtures for shared runtime ---

@pytest.fixture(scope="module")
def nightly_assets_path():
    """Path to model assets for weight sync tests."""
    return os.environ.get("NIGHTLY_ASSETS_PATH", "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B")


@pytest.fixture(scope="module")
def config_path(tmp_path_factory, nightly_assets_path):
    """Create config with all roles for weight sync testing."""
    tmp_path = tmp_path_factory.mktemp("config")
    return _create_config(tmp_path, nightly_assets_path, include_vllm=True)


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for module-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def runtime(config_path):
    """Load runtime once for all tests in module.

    Includes trainer, reference, and rollout (vLLM) roles.
    """
    runtime = await Runtime.from_plan(config_path)
    await runtime.start()
    yield runtime
    # No shutdown method - Ray handles actor cleanup


def _create_config(tmp_path, nightly_assets_path, include_vllm=True):
    """Create a minimal config for weight sync testing."""
    roles = [
        {
            "name": "trainer",
            "kind": "titan",
            "config": {
                "trainable": True,
                "profiling": {"enable_profiling": False},
                "metrics": {"log_freq": 1, "enable_tensorboard": False},
                "model": {
                    "name": "qwen3",
                    "flavor": "0.6B",
                    "hf_assets_path": nightly_assets_path,
                },
                "optimizer": {"name": "AdamW", "lr": 1e-3, "eps": 1e-8},  # Moderate LR for visible but stable changes
                "lr_scheduler": {"warmup_steps": 0},
                "training": {
                    "seq_len": 512,
                    "dtype": "bfloat16",
                    "mixed_precision_param": "bfloat16",
                    "mixed_precision_reduce": "float32",
                },
                "parallelism": {
                    "data_parallel_replicate_degree": 1,
                    "data_parallel_shard_degree": 1,
                    "fsdp_reshard_after_forward": "default",
                    "tensor_parallel_degree": 1,
                    "context_parallel_degree": 1,
                    "disable_loss_parallel": True,
                },
                "checkpoint": {
                    "enable": True,
                    "initial_load_in_hf": True,
                    "initial_load_model_only": True,
                },
                "activation_checkpoint": {"mode": "full", "selective_ac_option": "op"},
                "compile": {"enable": False},
            },
        },
        {
            "name": "reference",
            "kind": "titan",
            "config": {
                "trainable": False,
                "model": {
                    "name": "qwen3",
                    "flavor": "0.6B",
                    "hf_assets_path": nightly_assets_path,
                },
                "parallelism": {
                    "data_parallel_replicate_degree": 1,
                    "data_parallel_shard_degree": 1,
                    "fsdp_reshard_after_forward": "default",
                    "tensor_parallel_degree": 1,
                    "context_parallel_degree": 1,
                },
                "training": {
                    "seq_len": 512,
                    "dtype": "bfloat16",
                    "mixed_precision_param": "bfloat16",
                    "mixed_precision_reduce": "float32",
                },
                "checkpoint": {
                    "enable": True,
                    "initial_load_in_hf": True,
                    "initial_load_model_only": True,
                },
                "activation_checkpoint": {"mode": "full", "selective_ac_option": "op"},
                "compile": {"enable": False},
            },
        },
    ]

    wiring = [{"src": "trainer", "dst": "reference"}]

    if include_vllm:
        roles.append({
            "name": "rollout",
            "kind": "vllm",
            "config": {
                "model": nightly_assets_path,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "max_model_len": 512,
                "gpu_memory_utilization": 0.3,
                "dtype": "bfloat16",
                "logprobs_mode": "raw_logprobs",
                "max_num_seqs": 10,
                "max_num_batched_tokens": 2048,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
            },
        })
        wiring.append({"src": "trainer", "dst": "rollout"})

    config = {
        "run": {"name": "weight_sync_test"},
        "model": {"path": nightly_assets_path},
        "tokenizer": {
            "pretrained_model_name_or_path": "Qwen/Qwen3-0.6B",
            "use_fast": False,
        },
        "training": {
            "num_epochs": 1,
            "iterations_per_epoch": 1,
            "sync_reference_every": 1,
            "train_batch_size": 1,
        },
        "verifier": {"num_workers": 1},
        "loss": {"beta": 0.1, "eps": 0.2},
        "data": {"dataset": "dummy"},
        "data_iter": {
            "batch_size": 1,
            "system_prompt": "",
            "assistant_prefix": "",
        },
        "sampling": {
            "temperature": 0.0,  # Greedy for deterministic outputs
            "max_tokens": 32,
            "n": 1,
            "logprobs": 0,
        },
        "buffer": {"max_reads": 1},
        "roles": roles,
        "sync": {
            "chunk_mb": 100,
            "wiring": wiring,
        },
    }

    config_path = tmp_path / "weight_sync_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


async def do_training_step(trainer, reference):
    """Perform a training step to modify trainer weights."""
    batch_size = 2
    seq_len = 32
    completion_len = seq_len // 2

    # Create dummy training data
    # input_ids = [prompt | completion] where prompt_len = seq_len - completion_len
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    completion_ids = torch.randint(100, 1000, (batch_size, completion_len))
    prompt_lens = torch.tensor([seq_len - completion_len] * batch_size)

    # Get reference logprobs for the loss
    ref_logprobs = await reference.compute_logprobs(input_ids, completion_ids, prompt_lens)

    # Use dummy old logprobs and rewards
    old_logprobs = torch.randn(batch_size, completion_len)
    rewards = torch.tensor([1.0, 0.0])
    mask = torch.ones(batch_size, completion_len)

    loss_fn = GRPOLoss(beta=0.1, eps=0.2)

    # Forward + backward
    loss = await trainer.forward_backward(
        loss_fn,
        input_ids,
        loss_args=(completion_ids, ref_logprobs, old_logprobs, rewards),
        loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
    )

    # Optimizer step
    grad_norm = await trainer.optim_step()

    return loss, grad_norm


# --- Tests for weight sync ---
# Note: Tests run in order within each class. Tests that check "initial state"
# should be in earlier classes or at the start of each class.


@pytest.mark.integration
@pytest.mark.gpu
class TestA_InitialState:
    """Tests that verify initial state (must run first before any training)."""

    @pytest.mark.asyncio
    async def test_trainer_reference_logprobs_match_initially(self, runtime):
        """Trainer and reference should produce same logprobs initially (same checkpoint)."""
        trainer = runtime.roles["trainer"]
        reference = runtime.roles["reference"]

        # Test input
        input_ids = torch.randint(100, 1000, (1, 16))
        completion_ids = torch.randint(100, 1000, (1, 8))

        # Get logprobs from both
        trainer_logprobs = await trainer.compute_logprobs(input_ids, completion_ids)
        reference_logprobs = await reference.compute_logprobs(input_ids, completion_ids)

        print(f"Trainer logprobs:   {trainer_logprobs[0, :5]}")
        print(f"Reference logprobs: {reference_logprobs[0, :5]}")

        # Should be very close (same checkpoint)
        assert torch.allclose(trainer_logprobs, reference_logprobs, atol=1e-3), \
            "Trainer and reference should produce same logprobs initially"


@pytest.mark.integration
@pytest.mark.gpu
class TestB_WeightSyncIdempotent:
    """Test that weight sync is idempotent (doesn't change anything when weights match)."""

    @pytest.mark.asyncio
    async def test_vllm_sync_is_idempotent(self, runtime):
        """Syncing same weights twice should produce identical outputs."""
        trainer = runtime.roles["trainer"]
        rollout = runtime.roles["rollout"]

        prompt = "What is 2 + 2?"
        sampling_params = {"temperature": 0.0, "max_tokens": 16, "n": 1}

        # First sync
        await sync_titan_to_vllm(trainer, rollout)
        result_first = await rollout.generate([prompt], **sampling_params)
        tokens_first = list(result_first[0].outputs[0].token_ids)

        # Second sync (same weights, should be no-op)
        await sync_titan_to_vllm(trainer, rollout)
        result_second = await rollout.generate([prompt], **sampling_params)
        tokens_second = list(result_second[0].outputs[0].token_ids)

        print(f"Tokens after first sync:  {tokens_first[:10]}...")
        print(f"Tokens after second sync: {tokens_second[:10]}...")

        # Should be identical (same weights, greedy decoding)
        assert tokens_first == tokens_second, \
            "vLLM outputs should match after syncing same weights twice"


@pytest.mark.integration
@pytest.mark.gpu
class TestC_TrainingAndSync:
    """Tests that involve training and syncing (mutate state)."""

    @pytest.mark.asyncio
    async def test_logprobs_change_after_training(self, runtime):
        """Trainer logprobs should change after forward_backward + optim_step."""
        trainer = runtime.roles["trainer"]
        reference = runtime.roles["reference"]

        # Fixed input for comparison
        input_ids = torch.randint(100, 1000, (1, 16))
        completion_ids = torch.randint(100, 1000, (1, 8))

        # Get trainer logprobs before training
        logprobs_before = await trainer.compute_logprobs(input_ids, completion_ids)

        # Train
        for i in range(5):
            loss, grad_norm = await do_training_step(trainer, reference)
            print(f"Step {i}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

        # Get trainer logprobs after training
        logprobs_after = await trainer.compute_logprobs(input_ids, completion_ids)

        print(f"Trainer logprobs before: {logprobs_before[0, :5]}")
        print(f"Trainer logprobs after:  {logprobs_after[0, :5]}")

        # Logprobs should be different
        assert not torch.allclose(logprobs_before, logprobs_after, atol=1e-3), \
            "Trainer logprobs should change after training"

    @pytest.mark.asyncio
    async def test_reference_unchanged_before_sync(self, runtime):
        """Reference logprobs should not change when trainer is trained (before sync)."""
        trainer = runtime.roles["trainer"]
        reference = runtime.roles["reference"]

        # Sync first so reference matches current trainer state
        await sync_titan_to_titan(trainer, reference)

        # Test input (fixed for comparison)
        input_ids = torch.randint(100, 1000, (1, 16))
        completion_ids = torch.randint(100, 1000, (1, 8))

        # Get reference logprobs before training
        logprobs_before = await reference.compute_logprobs(input_ids, completion_ids)

        # Train the trainer model (multiple steps)
        for _ in range(3):
            loss, grad_norm = await do_training_step(trainer, reference)
            print(f"Training step: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

        # Reference should still have old weights
        logprobs_after = await reference.compute_logprobs(input_ids, completion_ids)

        print(f"Reference logprobs before training: {logprobs_before[0, :5]}")
        print(f"Reference logprobs after training:  {logprobs_after[0, :5]}")

        assert torch.allclose(logprobs_before, logprobs_after, atol=1e-3), \
            "Reference logprobs should not change before sync"

    @pytest.mark.asyncio
    async def test_reference_matches_trainer_after_sync(self, runtime):
        """After syncing, reference should produce same logprobs as trainer."""
        trainer = runtime.roles["trainer"]
        reference = runtime.roles["reference"]

        # Sync current trainer weights to reference
        await sync_titan_to_titan(trainer, reference)

        # Test input
        input_ids = torch.randint(100, 1000, (1, 16))
        completion_ids = torch.randint(100, 1000, (1, 8))

        # Both should produce same logprobs now
        trainer_logprobs = await trainer.compute_logprobs(input_ids, completion_ids)
        reference_logprobs = await reference.compute_logprobs(input_ids, completion_ids)

        print(f"Trainer logprobs:   {trainer_logprobs[0, :5]}")
        print(f"Reference logprobs: {reference_logprobs[0, :5]}")

        assert torch.allclose(trainer_logprobs, reference_logprobs, atol=1e-3), \
            "Trainer and reference should produce same logprobs after sync"

    @pytest.mark.asyncio
    async def test_vllm_outputs_change_after_training_and_sync(self, runtime):
        """After training + sync, vLLM outputs should change."""
        trainer = runtime.roles["trainer"]
        reference = runtime.roles["reference"]
        rollout = runtime.roles["rollout"]

        prompt = "What is 2 + 2?"
        # Request logprobs so we can compare them even if top token is the same
        sampling_params = {"temperature": 0.0, "max_tokens": 16, "n": 1, "logprobs": 1}

        def extract_logprobs(output):
            """Extract logprob values from vLLM output (handles dict or object format)."""
            lps = output.logprobs
            if not lps:
                return []
            result = []
            for lp in lps:
                if isinstance(lp, dict):
                    # Get the logprob of the chosen token (first key's value)
                    for token_id, info in lp.items():
                        if isinstance(info, dict):
                            result.append(info.get("logprob", 0))
                        else:
                            result.append(getattr(info, "logprob", 0))
                        break
                else:
                    result.append(getattr(lp, "logprob", 0))
            return result

        # First sync to establish baseline (ensure vLLM has trainer weights)
        await sync_titan_to_vllm(trainer, rollout)
        result_before = await rollout.generate([prompt], **sampling_params)
        tokens_before = list(result_before[0].outputs[0].token_ids)
        logprobs_before = extract_logprobs(result_before[0].outputs[0])

        # Train the model (multiple steps for larger weight change)
        for _ in range(3):
            loss, grad_norm = await do_training_step(trainer, reference)
            print(f"Training step: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

        # Sync modified weights to vLLM
        await sync_titan_to_vllm(trainer, rollout)

        # Get output after training + sync
        result_after = await rollout.generate([prompt], **sampling_params)
        tokens_after = list(result_after[0].outputs[0].token_ids)
        logprobs_after = extract_logprobs(result_after[0].outputs[0])

        print(f"Tokens before training: {tokens_before[:10]}...")
        print(f"Tokens after training:  {tokens_after[:10]}...")
        print(f"Logprobs before (first 5): {logprobs_before[:5]}")
        print(f"Logprobs after (first 5):  {logprobs_after[:5]}")

        # Either tokens or logprobs should be different (weights changed)
        tokens_changed = tokens_before != tokens_after
        logprobs_changed = logprobs_before != logprobs_after

        assert tokens_changed or logprobs_changed, \
            "vLLM outputs should change after training and weight sync (tokens or logprobs)"


@pytest.mark.integration
@pytest.mark.gpu
class TestD_HFExportRoundTrip:
    """Test that exported HF model produces matching logits.

    NOTE: This test runs after training tests, so weights may be in a diverged state.
    The test uses relaxed tolerances to account for this, while still verifying the
    export mechanism works correctly.
    """

    @pytest.mark.asyncio
    async def test_exported_hf_model_matches_titan(self, runtime, tmp_path_factory):
        """Export Titan model to HF format and verify logits match.

        This test verifies:
        1. The export mechanism produces valid HF-format weights
        2. The exported model can be loaded by transformers
        3. Forward passes produce matching logits (with tolerance for implementation differences)

        Note: There are small numerical differences between Titan and HF implementations
        due to different RoPE/attention implementations and bf16 precision. On fresh weights,
        typical max_diff is ~0.2-0.3, mean_diff ~0.03. After training (especially with
        diverged weights), differences may be larger but should still be proportionally similar.
        """
        from transformers import AutoModelForCausalLM

        trainer = runtime.roles["trainer"]

        # Use a fixed input for reproducibility
        input_ids = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]])

        # Get Titan logits
        titan_logits = await trainer.forward(input_ids)

        # Export to HF format
        export_dir = tmp_path_factory.mktemp("hf_export")
        export_path = str(export_dir / "model")
        await trainer.export_to_hf(export_path)

        # Load as HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(
            export_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        hf_model.eval()

        # Forward through HF model
        with torch.no_grad():
            hf_output = hf_model(input_ids.to("cuda:0"))
            hf_logits = hf_output.logits.cpu()

        print(f"Titan logits shape: {titan_logits.shape}")
        print(f"HF logits shape:    {hf_logits.shape}")
        print(f"Titan logits[0,0,:5]: {titan_logits[0, 0, :5]}")
        print(f"HF logits[0,0,:5]:    {hf_logits[0, 0, :5]}")
        print(f"Titan logits[0,-1,:5]: {titan_logits[0, -1, :5]}")
        print(f"HF logits[0,-1,:5]:    {hf_logits[0, -1, :5]}")

        # Check shapes match
        assert titan_logits.shape == hf_logits.shape, \
            f"Shape mismatch: Titan {titan_logits.shape} vs HF {hf_logits.shape}"

        # Check values are close
        # Use relative comparison: differences should be proportional to the magnitude of values
        max_diff = (titan_logits.float() - hf_logits.float()).abs().max().item()
        mean_diff = (titan_logits.float() - hf_logits.float()).abs().mean().item()
        logit_magnitude = titan_logits.float().abs().mean().item()
        relative_diff = mean_diff / max(logit_magnitude, 1.0)

        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Logit magnitude (mean abs): {logit_magnitude}")
        print(f"Relative difference (mean_diff/magnitude): {relative_diff:.4f}")

        # Relative tolerance check: mean difference should be small relative to value magnitude
        # Typically <5% for valid weights, but can be higher for extreme values
        assert relative_diff < 0.15, \
            f"Relative difference too high: {relative_diff:.4f} (mean_diff={mean_diff}, magnitude={logit_magnitude})"

        # Cleanup
        del hf_model
        torch.cuda.empty_cache()


def _create_hf_compat_config(tmp_path, nightly_assets_path):
    """Create a config with hf_compatible=True for testing HF parity."""
    roles = [
        {
            "name": "trainer",
            "kind": "titan",
            "config": {
                "trainable": True,
                "hf_compatible": True,  # Enable HF-compatible precision
                "profiling": {"enable_profiling": False},
                "metrics": {"log_freq": 1, "enable_tensorboard": False},
                "model": {
                    "name": "qwen3",
                    "flavor": "0.6B",
                    "hf_assets_path": nightly_assets_path,
                },
                "optimizer": {"name": "AdamW", "lr": 1e-3, "eps": 1e-8},
                "lr_scheduler": {"warmup_steps": 0},
                "training": {
                    "seq_len": 512,
                    "dtype": "bfloat16",
                    "mixed_precision_param": "bfloat16",
                    "mixed_precision_reduce": "float32",
                },
                "parallelism": {
                    "data_parallel_replicate_degree": 1,
                    "data_parallel_shard_degree": 1,
                    "fsdp_reshard_after_forward": "default",
                    "tensor_parallel_degree": 1,
                    "context_parallel_degree": 1,
                    "disable_loss_parallel": True,
                },
                "checkpoint": {
                    "enable": True,
                    "initial_load_in_hf": True,
                    "initial_load_model_only": True,
                },
                "activation_checkpoint": {"mode": "full", "selective_ac_option": "op"},
                "compile": {"enable": False},
            },
        },
    ]

    config = {
        "run": {"name": "hf_compat_test"},
        "model": {"path": nightly_assets_path},
        "tokenizer": {
            "pretrained_model_name_or_path": "Qwen/Qwen3-0.6B",
            "use_fast": False,
        },
        "training": {
            "num_epochs": 1,
            "iterations_per_epoch": 1,
            "sync_reference_every": 1,
            "train_batch_size": 1,
        },
        "verifier": {"num_workers": 1},
        "loss": {"beta": 0.1, "eps": 0.2},
        "data": {"dataset": "dummy"},
        "data_iter": {
            "batch_size": 1,
            "system_prompt": "",
            "assistant_prefix": "",
        },
        "sampling": {
            "temperature": 0.0,
            "max_tokens": 32,
            "n": 1,
            "logprobs": 0,
        },
        "buffer": {"max_reads": 1},
        "roles": roles,
        "sync": {
            "chunk_mb": 100,
            "wiring": [],
        },
    }

    config_path = tmp_path / "hf_compat_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


@pytest.mark.integration
@pytest.mark.gpu
class TestE_HFCompatibleMode:
    """Test HF-compatible precision mode for exact parity with HuggingFace.

    When hf_compatible=True is set in config, Titan should produce identical outputs
    to HuggingFace (within bf16 tolerance). This is critical for benchmark
    evaluation where trained models are loaded via transformers.
    """

    @pytest.fixture(scope="class")
    def hf_compat_config_path(self, tmp_path_factory, nightly_assets_path):
        """Create a config with hf_compatible=True."""
        tmp_path = tmp_path_factory.mktemp("hf_compat_config")
        return _create_hf_compat_config(tmp_path, nightly_assets_path)

    @pytest.fixture(scope="class")
    async def hf_compat_runtime(self, hf_compat_config_path):
        """Runtime with hf_compatible=True in config."""
        runtime = await Runtime.from_plan(hf_compat_config_path)
        await runtime.start()
        yield runtime

    @pytest.mark.asyncio
    async def test_hf_compatible_mode_exact_parity(self, hf_compat_runtime, tmp_path_factory, nightly_assets_path):
        """With hf_compatible=True, Titan and HF should produce near-identical logits.

        This verifies:
        1. RMSNorm uses float32 variance (like HF)
        2. RoPE uses float32 computation (like HF)
        3. Exported model matches closely (<1% relative error)
        """
        from transformers import AutoModelForCausalLM

        trainer = hf_compat_runtime.roles["trainer"]

        # Debug: Check rope_cache info
        rope_info = await trainer.get_rope_cache_info()
        print(f"\nRope cache info: {rope_info}")

        # Use a longer sequence to test position-dependent RoPE
        input_ids = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]])

        # Get Titan logits (with HF-compatible mode enabled)
        titan_logits = await trainer.forward(input_ids)

        # Export to HF format
        export_dir = tmp_path_factory.mktemp("hf_compat_export")
        export_path = str(export_dir / "model")
        await trainer.export_to_hf(export_path)

        # Load as HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(
            export_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        hf_model.eval()

        # Forward through HF model
        with torch.no_grad():
            hf_output = hf_model(input_ids.to("cuda:0"))
            hf_logits = hf_output.logits.cpu()

        # Detailed comparison per position
        print("\nPer-position comparison (HF-compatible mode):")
        print("Position | Mean Diff | Max Diff")
        print("-" * 40)
        for pos in range(min(input_ids.shape[1], 8)):
            pos_diff = (titan_logits[0, pos] - hf_logits[0, pos]).abs()
            print(f"{pos:8d} | {pos_diff.mean():.6f} | {pos_diff.max():.6f}")

        # Overall comparison
        max_diff = (titan_logits.float() - hf_logits.float()).abs().max().item()
        mean_diff = (titan_logits.float() - hf_logits.float()).abs().mean().item()
        logit_magnitude = titan_logits.float().abs().mean().item()
        relative_diff = mean_diff / max(logit_magnitude, 1.0)

        print(f"\nOverall:")
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Logit magnitude (mean abs): {logit_magnitude}")
        print(f"Relative difference: {relative_diff:.6f} ({relative_diff*100:.3f}%)")

        # With HF-compatible mode, we expect much tighter tolerance
        # bf16 has ~0.1% relative error, so <1% should be easily achievable
        assert relative_diff < 0.01, \
            f"HF-compatible mode should give <1% relative error, got {relative_diff*100:.2f}%"

        assert max_diff < 0.5, \
            f"Max difference too high with HF-compatible mode: {max_diff}"

        # Cleanup
        del hf_model
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.gpu
class TestF_SyncPerformanceOptimization:
    """Tests for sync performance optimizations.

    The titan-to-titan sync uses prepare_recv_state/clear_recv_state to cache
    the model state dict on the receiver side, avoiding repeated get_model_state_dict
    calls for each chunk. This provides ~10x speedup for large models (e.g., 33s -> 3s
    for Qwen3-32B).
    """

    @pytest.mark.asyncio
    async def test_titan_to_titan_sync_uses_recv_cache(self, runtime):
        """Verify titan-to-titan sync uses receiver-side caching.

        This test verifies correctness after the optimization. The performance
        improvement can be observed in traces (sync.titan_to_titan duration).
        """
        import time
        trainer = runtime.roles["trainer"]
        reference = runtime.roles["reference"]

        # Test input
        input_ids = torch.randint(100, 1000, (1, 16))
        completion_ids = torch.randint(100, 1000, (1, 8))

        # Get reference logprobs before sync
        ref_logprobs_before = await reference.compute_logprobs(input_ids, completion_ids)

        # Do a training step to change trainer weights
        await do_training_step(trainer, reference)

        # Sync trainer -> reference (should use prepare_recv_state optimization)
        t0 = time.perf_counter()
        await sync_titan_to_titan(trainer, reference)
        sync_time = time.perf_counter() - t0

        print(f"Titan-to-titan sync took {sync_time:.3f}s")

        # Verify sync worked correctly
        trainer_logprobs = await trainer.compute_logprobs(input_ids, completion_ids)
        ref_logprobs_after = await reference.compute_logprobs(input_ids, completion_ids)

        print(f"Trainer logprobs:       {trainer_logprobs[0, :5]}")
        print(f"Reference after sync:   {ref_logprobs_after[0, :5]}")

        # Reference should now match trainer
        assert torch.allclose(trainer_logprobs, ref_logprobs_after, atol=1e-3), \
            "Reference should match trainer after sync"

        # Reference should have changed from before
        assert not torch.allclose(ref_logprobs_before, ref_logprobs_after, atol=1e-3), \
            "Reference logprobs should change after sync"

    @pytest.mark.asyncio
    async def test_multiple_syncs_work_correctly(self, runtime):
        """Verify multiple consecutive syncs work correctly with caching."""
        trainer = runtime.roles["trainer"]
        reference = runtime.roles["reference"]

        input_ids = torch.randint(100, 1000, (1, 16))
        completion_ids = torch.randint(100, 1000, (1, 8))

        # First sync
        await sync_titan_to_titan(trainer, reference)
        logprobs_after_sync1 = await reference.compute_logprobs(input_ids, completion_ids)

        # Train
        await do_training_step(trainer, reference)

        # Second sync
        await sync_titan_to_titan(trainer, reference)
        logprobs_after_sync2 = await reference.compute_logprobs(input_ids, completion_ids)

        # Third sync (without training - should be idempotent)
        await sync_titan_to_titan(trainer, reference)
        logprobs_after_sync3 = await reference.compute_logprobs(input_ids, completion_ids)

        print(f"After sync 1: {logprobs_after_sync1[0, :5]}")
        print(f"After sync 2: {logprobs_after_sync2[0, :5]}")
        print(f"After sync 3: {logprobs_after_sync3[0, :5]}")

        # Logprobs should change between sync 1 and 2 (training happened)
        assert not torch.allclose(logprobs_after_sync1, logprobs_after_sync2, atol=1e-3), \
            "Logprobs should change after training + sync"

        # Logprobs should be same between sync 2 and 3 (no training)
        assert torch.allclose(logprobs_after_sync2, logprobs_after_sync3, atol=1e-3), \
            "Logprobs should be stable without training"
