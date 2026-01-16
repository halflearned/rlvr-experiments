"""Shared fixtures for nightly tests."""

import pytest


@pytest.fixture
def ray_ctx():
    """Ensure Ray is available for datasets/queues used in nightly tests."""
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    yield ray
    ray.shutdown()

