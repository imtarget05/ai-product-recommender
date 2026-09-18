"""Dockerfile tests for RecSys service."""

import pathlib

import pytest


def test_dockerfile_has_multistage():
    """Test that the Dockerfile has at least 2 FROM statements (multistage build)."""
    dockerfile_path = pathlib.Path("src/recsys/Dockerfile")
    assert dockerfile_path.exists(), f"Dockerfile not found at {dockerfile_path}"

    dockerfile_content = dockerfile_path.read_text()
    from_count = dockerfile_content.count("FROM")
    assert from_count >= 2, (
        f"Dockerfile must have at least 2 FROM statements for multistage build, "
        f"but found {from_count}"
    )