"""Content-addressed, atomic caches for derived work; never edit a foundation."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .cache import file_sha256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_identity(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        digest.update(canonical([key, value.dtype.str, value.shape]).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


class WorkCache:
    def __init__(self, root):
        self.root = None if root is None else Path(root)
        self.events = []

    def get(self, kind, identity, compute):
        identity = {"schema_version": 1, "kind": kind, **identity}
        key = hashlib.sha256(canonical(identity).encode()).hexdigest()
        path = None if self.root is None else self.root / kind / key
        if path is not None and path.exists():
            try:
                metadata = json.loads((path / "manifest.json").read_text())
                if metadata["identity"] != identity or metadata["key"] != key:
                    raise ValueError("identity differs")
                if file_sha256(path / "values.npz") != metadata["sha256"]:
                    raise ValueError("checksum differs")
                with np.load(path / "values.npz", allow_pickle=False) as archive:
                    values = {name: archive[name] for name in archive.files}
                if array_identity(values) != metadata["array_sha256"]:
                    raise ValueError("array checksum differs")
            except (OSError, ValueError, KeyError) as error:
                raise ValueError(f"Corrupt/incomplete cache {path}: {error}; remove this entry to rebuild") from error
            hit = True
        else:
            values = {name: np.asarray(value) for name, value in compute().items()}
            if any(value.dtype.hasobject for value in values.values()):
                raise ValueError("Object arrays are prohibited in work caches")
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=path.parent))
                try:
                    with (staging / "values.npz").open("wb") as handle:
                        np.savez_compressed(handle, **values)
                        handle.flush()
                        os.fsync(handle.fileno())
                    metadata = {"identity": identity, "key": key, "sha256": file_sha256(staging / "values.npz"),
                                "array_sha256": array_identity(values)}
                    (staging / "manifest.json").write_text(canonical(metadata) + "\n")
                    try:
                        os.rename(staging, path)
                    except OSError:
                        if not path.exists():
                            raise
                        # Another publisher won. Verify its complete result.
                        concurrent = self.get(kind, {k: v for k, v in identity.items() if k not in {"schema_version", "kind"}}, compute)
                        if array_identity(concurrent) != array_identity(values):
                            raise ValueError("Concurrent cache builders produced different arrays")
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
            hit = False
        self.events.append({"kind": kind, "key": key, "hit": hit, "path": None if path is None else str(path.resolve())})
        return values
