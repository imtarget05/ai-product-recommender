"""WP4 — Stateless Free-Infra tests (RED until src.storage lands).

Spec: STORAGE_BACKEND=local|s3mock|r2; R2_ENDPOINT + DATABASE_URL (Neon/
Supabase free); ItemEmbedder.save/load qua storage interface; routes đọc
artifact qua storage, không path cứng local.
"""
import inspect
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest


# ---------------------------------------------------------------- T1: interface
def test_storage_interface_abc():
    from src.storage.base import Storage

    assert inspect.isabstract(Storage)
    for method in ("save", "load", "exists", "delete"):
        assert hasattr(Storage, method), f"missing Storage.{method}"


# ---------------------------------------------------------------- T2: local
def test_local_disk_roundtrip(tmp_path):
    from src.storage.local_disk import LocalDisk

    store = LocalDisk(root=str(tmp_path))
    store.save("a/b.txt", b"hello")
    assert store.exists("a/b.txt")
    assert store.load("a/b.txt") == b"hello"
    store.delete("a/b.txt")
    assert not store.exists("a/b.txt")


# ---------------------------------------------------------------- T3: s3mock
class _MockS3Handler(BaseHTTPRequestHandler):
    store: dict = {}

    def _key(self) -> str:
        return self.path.lstrip("/")

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        self.store[self._key()] = self.rfile.read(length)
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        body = self.store.get(self._key())
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200 if self._key() in self.store else 404)
        self.end_headers()

    def do_DELETE(self):
        self.store.pop(self._key(), None)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):  # silence
        pass




@pytest.fixture
def s3_endpoint():
    _MockS3Handler.store = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockS3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_s3mock_roundtrip(s3_endpoint):
    from src.storage.s3mock import S3MockStorage

    store = S3MockStorage(endpoint=s3_endpoint)
    store.save("x/y.bin", b"\x00\x01")
    assert store.exists("x/y.bin")
    assert store.load("x/y.bin") == b"\x00\x01"
    store.delete("x/y.bin")
    assert not store.exists("x/y.bin")


def test_factory_storage_backend(monkeypatch, tmp_path):
    from src.storage.factory import get_storage
    from src.storage.local_disk import LocalDisk
    from src.storage.s3mock import S3MockStorage

    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))
    assert isinstance(get_storage(), LocalDisk)

    monkeypatch.setenv("STORAGE_BACKEND", "s3mock")
    monkeypatch.setenv("S3_ENDPOINT", "http://127.0.0.1:9999")
    backend = get_storage()
    assert isinstance(backend, S3MockStorage)
    assert backend.endpoint == "http://127.0.0.1:9999"

    monkeypatch.setenv("STORAGE_BACKEND", "r2")
    monkeypatch.setenv("R2_ENDPOINT", "https://acc.r2.cloudflarestorage.com/bucket")
    assert isinstance(get_storage(), S3MockStorage)

    monkeypatch.setenv("STORAGE_BACKEND", "bogus")
    with pytest.raises(ValueError):
        get_storage()


def _make_embedder(tmp_path):
    from src.database.models import Product
    from src.features.text_embedder import ItemEmbedder

    products = [
        Product(id=1, title="ao thun nam", category="thoi trang", tags="cao cap", description="ao cotton"),
        Product(id=2, title="quan jean", category="thoi trang", tags="den", description="jean nam"),
    ]
    emb = ItemEmbedder()
    emb.fit_transform(products)
    return emb


def test_item_embedder_save_load_via_storage(tmp_path):
    from src.storage.local_disk import LocalDisk

    emb = _make_embedder(tmp_path)
    store = LocalDisk(root=str(tmp_path / "store"))
    emb.save(storage=store)
    assert store.exists("item_embeddings.npy")

    from src.features.text_embedder import ItemEmbedder

    emb2 = ItemEmbedder()
    assert emb2.load(storage=store) is True
    assert np.allclose(emb.embeddings, emb2.embeddings)
    assert emb2.product_ids == emb.product_ids


def test_item_embedder_backcompat_output_dir(tmp_path):
    """Giữ API cũ: save/load(output_dir=...) vẫn ghi file trực tiếp."""
    emb = _make_embedder(tmp_path)
    out = tmp_path / "legacy"
    emb.save(output_dir=out)
    assert (out / "item_embeddings.npy").exists()

    from src.features.text_embedder import ItemEmbedder

    emb2 = ItemEmbedder()
    assert emb2.load(output_dir=out) is True
    assert np.allclose(emb.embeddings, emb2.embeddings)


def test_routes_load_artifact_via_storage(monkeypatch, tmp_path):
    from src.api import routes as routes_mod
    from src.storage.local_disk import LocalDisk

    store = LocalDisk(root=str(tmp_path))
    store.save("artifacts/health.json", b'{"ok": true}')

    seen = {}

    class Spy(LocalDisk):
        def load(self, key):
            seen["key"] = key
            return super().load(key)

    spy = Spy(root=str(tmp_path))
    monkeypatch.setattr(routes_mod, "get_storage", lambda: spy)
    payload = routes_mod._storage_load_artifact("artifacts/health.json")
    assert payload == b'{"ok": true}'
    assert seen["key"] == "artifacts/health.json"


def test_env_example_has_stateless_vars():
    with open(".env.example", encoding="utf-8") as f:
        text = f.read()
    for var in ("STORAGE_BACKEND", "R2_ENDPOINT", "DATABASE_URL"):
        assert var in text, f".env.example missing {var}"

