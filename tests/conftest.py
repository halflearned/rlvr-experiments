"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock


# --- Markers ---

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "nightly: mark test as nightly (slow, full pipeline)")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow (requires network/downloads)")


# --- Skips for unavailable hardware ---

def pytest_collection_modifyitems(config, items):
    """Skip GPU-marked tests when CUDA is unavailable."""
    import torch
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="GPU not available in test environment")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# --- Event loop fixture for async tests ---

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# --- Mock fixtures ---

@pytest.fixture
def mock_tokenizer():
    """Real Qwen tokenizer for realistic testing."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=False)
    return tokenizer


@pytest.fixture
def mock_vllm_output():
    """Factory for creating mock vLLM outputs."""
    def make_output(token_ids, text, logprobs_values=None):
        output = MagicMock()
        output.token_ids = token_ids
        output.text = text

        if logprobs_values is None:
            logprobs_values = [0.0] * len(token_ids)

        # Create logprobs structure: list of dicts mapping token_id -> logprob info
        output.logprobs = []
        for i, (tid, lp) in enumerate(zip(token_ids, logprobs_values)):
            lp_info = MagicMock()
            lp_info.logprob = lp
            output.logprobs.append({tid: lp_info})

        return output
    return make_output


@pytest.fixture
def mock_vllm_response(mock_vllm_output):
    """Factory for creating mock vLLM responses."""
    def make_response(prompt_token_ids, outputs):
        response = MagicMock()
        response.prompt_token_ids = prompt_token_ids
        response.outputs = outputs
        return response
    return make_response


# --- Ray fixtures ---

@pytest.fixture(scope="session")
def ray_session():
    """Initialize Ray once per test session, or use existing cluster."""
    import ray
    if not ray.is_initialized():
        # Try to connect to existing cluster first, fall back to local init
        try:
            ray.init(address="auto", ignore_reinit_error=True)
        except ConnectionError:
            ray.init(ignore_reinit_error=True, num_cpus=4)
    yield
    # Don't shutdown - let other tests use it


@pytest.fixture
def ray_context(ray_session):
    """Provide Ray context for tests that need it."""
    import ray
    return ray
