"""WP4 — Storage factory: STORAGE_BACKEND=local|s3mock|r2 (default local)."""
import os

from .base import Storage
from .local_disk import LocalDisk
from .s3mock import S3MockStorage


def get_storage() -> Storage:
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()

    if backend == "local":
        return LocalDisk(root=os.getenv("OUT_DIR") or "out")

    if backend in ("s3mock", "r2"):
        endpoint = os.getenv("S3_ENDPOINT") or os.getenv("R2_ENDPOINT") or ""
        if not endpoint:
            raise ValueError(f"STORAGE_BACKEND={backend} requires S3_ENDPOINT/R2_ENDPOINT")
        return S3MockStorage(endpoint=endpoint)

    raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r} (expected local|s3mock|r2)")
