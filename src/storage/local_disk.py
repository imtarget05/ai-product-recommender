"""WP4 — LocalDisk storage backend (root = OUT_DIR cấu hình được, atomic write)."""
import os
from typing import Optional

from .base import Storage


class LocalDisk(Storage):
    def __init__(self, root: Optional[str] = None):
        self.root = os.path.abspath(root or os.getenv("OUT_DIR", "out"))
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        key = key.lstrip("/")
        return os.path.join(self.root, *key.split("/"))

    def save(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Atomic write qua tmp + rename (tránh file dở dang khi crash).
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    def load(self, key: str) -> bytes:
        path = self._path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(key)
        with open(path, "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def delete(self, key: str) -> None:
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)
