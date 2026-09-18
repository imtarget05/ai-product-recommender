"""WP4 — Storage abstraction ABC (stateless free-infra).

Contract: save(key, bytes), load(key) -> bytes, exists(key) -> bool, delete(key).
Chuẩn free-tier: local disk / S3Mock / R2-compat — không phụ thuộc SDK ngoài.
"""
from abc import ABC, abstractmethod


class Storage(ABC):
    @abstractmethod
    def save(self, key: str, data: bytes) -> None:
        """Persist bytes under key (overwrite if exists)."""

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Return bytes for key; raise FileNotFoundError if missing."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True if key exists."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete key (no-op if missing)."""
