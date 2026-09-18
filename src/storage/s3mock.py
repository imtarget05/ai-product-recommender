"""WP4 — S3Mock / R2-compat storage backend (stdlib-only urllib, không SDK ngoài).

PUT/GET/HEAD/DELETE trên endpoint cấu hình được — tương thích S3Mock local và
Cloudflare R2 S3-compatible endpoint.
"""
import urllib.error
import urllib.request

from .base import Storage


class S3MockStorage(Storage):
    def __init__(self, endpoint: str):
        if not endpoint:
            raise ValueError("S3MockStorage requires an endpoint URL")
        self.endpoint = endpoint.rstrip("/")

    def _url(self, key: str) -> str:
        return f"{self.endpoint}/{key.lstrip('/')}"

    def save(self, key: str, data: bytes) -> None:
        req = urllib.request.Request(self._url(key), data=data, method="PUT")
        with urllib.request.urlopen(req, timeout=30):
            pass

    def load(self, key: str) -> bytes:
        try:
            with urllib.request.urlopen(self._url(key), timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(key) from exc
            raise

    def exists(self, key: str) -> bool:
        req = urllib.request.Request(self._url(key), method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def delete(self, key: str) -> None:
        req = urllib.request.Request(self._url(key), method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=30):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
