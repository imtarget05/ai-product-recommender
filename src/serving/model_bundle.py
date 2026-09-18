"""Immutable versioned recommendation model bundle (serving contract)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BundleError(RuntimeError):
    """Raised when a production bundle is missing or incompatible."""


@dataclass(frozen=True)
class ServingBundle:
    version: str
    dataset_fingerprint: str
    schema: dict
    checksums: dict
    payload: dict
    bundle_path: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_bundle(path: str | Path) -> ServingBundle:
    """Load and verify an immutable model bundle directory or manifest."""
    p = Path(path)
    manifest_path = p / "manifest.json" if p.is_dir() else p
    base = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise BundleError(f"missing or invalid bundle manifest: {exc}") from exc
    for key in ("version", "dataset_fingerprint", "schema", "checksums"):
        if key not in manifest:
            raise BundleError(f"bundle manifest missing {key}")
    checksums = dict(manifest["checksums"])
    for name, expected in checksums.items():
        target = base / name
        if not target.exists():
            raise BundleError(f"bundle artifact missing: {name}")
        actual = _sha256_file(target)
        if actual != expected:
            raise BundleError(f"bundle checksum mismatch: {name}")
    payload: dict[str, Any] = {}
    payload_path = base / "serving_payload.json"
    if payload_path.exists():
        payload = json.loads(payload_path.read_text())
    return ServingBundle(
        version=str(manifest["version"]),
        dataset_fingerprint=str(manifest["dataset_fingerprint"]),
        schema=dict(manifest["schema"]),
        checksums=checksums,
        payload=payload,
        bundle_path=str(base),
    )


def write_model_bundle(
    dest: str | Path,
    version: str,
    dataset_fingerprint: str,
    schema: dict,
    artifacts: dict[str, bytes],
    payload: dict | None = None,
) -> ServingBundle:
    """Write artifacts + manifest atomically (tmp dir + rename)."""
    import os
    import tempfile

    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".bundle-", dir=str(dest_p.parent)))
    checksums: dict[str, str] = {}
    try:
        for name, data in artifacts.items():
            (tmp / name).write_bytes(data)
            checksums[name] = hashlib.sha256(data).hexdigest()
        payload = dict(payload or {})
        (tmp / "serving_payload.json").write_bytes(
            json.dumps(payload, ensure_ascii=False, indent=2).encode()
        )
        checksums["serving_payload.json"] = hashlib.sha256(
            (tmp / "serving_payload.json").read_bytes()
        ).hexdigest()
        manifest = {
            "version": version,
            "dataset_fingerprint": dataset_fingerprint,
            "schema": schema,
            "checksums": checksums,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if dest_p.exists():
            import shutil

            shutil.rmtree(dest_p)
        os.replace(tmp, dest_p)
    finally:
        import shutil

        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return load_model_bundle(dest_p)
