"""Federated (disk) utilities shared by server/client.

Functions:
- parse_precision_to_np_dtype: map 'fp16'/'fp32' to numpy dtype
- robust_to_np_storable: cast/convert JAX/NumPy arrays to desired dtype for storage
- flatten_params: flatten a nested dict of arrays using flax.traverse_util
- save_npz_dict: save a pure params dict to uncompressed .npz with stable keys
- load_npz_dict: read such .npz into a nested dict
- append_jsonl: append a JSON line to a file (resume-friendly)
- build_iid_indices: deterministic IID partition indices
"""
from __future__ import annotations

import json
import pathlib
from typing import List, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

try:
    from flax import traverse_util
except Exception:  # pragma: no cover - allow import fallback when editing statically
    traverse_util = None  # type: ignore


def parse_precision_to_np_dtype(prec: str | None, default: str = "fp16") -> np.dtype:
    p = (prec or default or "fp16").lower()
    return np.float16 if p == "fp16" else np.float32


def robust_to_np_storable(x, dtype: np.dtype) -> np.ndarray:
    """Convert JAX/NumPy array-like to a NumPy array suitable for storage.

    - For inexact tensors, cast to the given dtype using jax.numpy first to handle bf16.
    - For weird/legacy dtypes, attempt safe reinterpretation/astype to desired dtype.
    """
    try:
        ja = jnp.asarray(x)
        if jnp.issubdtype(ja.dtype, jnp.inexact):
            ja = ja.astype(dtype)
        return np.asarray(ja)
    except Exception:
        arr = np.asarray(x)
        if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("V", "O"):
            try:
                arr = arr.view(np.uint16).astype(dtype, copy=False)
            except Exception:
                arr = arr.astype(dtype, copy=False)
        return arr


def flatten_params(pure: dict) -> Tuple[List[tuple], List[np.ndarray]]:
    if traverse_util is None:
        raise RuntimeError("flax.traverse_util is required to flatten params")
    flat = traverse_util.flatten_dict(pure)
    keys = sorted(flat.keys())
    arrays = [np.asarray(flat[k]) for k in keys]
    return keys, arrays


def save_npz_dict(file_path: str | pathlib.Path, pure: dict, *, dtype: np.dtype = np.float16) -> str:
    """Save a pure params dict to uncompressed NPZ with textual keys.

    The NPZ will contain a string array 'keys' and tensors 'a0'..'aN'.
    """
    file_path = pathlib.Path(file_path).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    keys, arrays_raw = flatten_params(pure)
    arrays = [robust_to_np_storable(a, dtype=dtype) for a in arrays_raw]

    data = {"keys": np.array(["/".join(k) for k in keys])}
    for i, arr in enumerate(arrays):
        data[f"a{i}"] = arr
    np.savez(file_path, **data)
    return str(file_path)


def load_npz_dict(file_path: str | pathlib.Path) -> dict:
    """Load a NPZ produced by save_npz_dict back to a nested dict of NumPy arrays."""
    if traverse_util is None:
        raise RuntimeError("flax.traverse_util is required to unflatten params")
    file = pathlib.Path(file_path)
    try:
        with np.load(file, allow_pickle=False) as f:
            key_arr = f["keys"]
            key_strs = [str(s) for s in key_arr.tolist()]
            arrays = [f[f"a{i}"] for i in range(len(key_strs))]
    except ValueError as e:
        if "Object arrays cannot be loaded" in str(e):
            with np.load(file, allow_pickle=True) as f:  # trusted local file
                key_arr = f["keys"]
                key_strs = [str(s) for s in key_arr.tolist()]
                arrays = [f[f"a{i}"] for i in range(len(key_strs))]
        else:
            raise
    fixed = []
    for arr in arrays:
        if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("V", "O"):
            try:
                arr = arr.view(np.uint16).astype(np.float16, copy=False)
            except Exception:
                arr = arr.astype(np.float16, copy=False)
        fixed.append(arr)
    arrays = fixed
    keys = [tuple(s.split("/")) for s in key_strs]
    flat = {k: v for k, v in zip(keys, arrays)}
    return traverse_util.unflatten_dict(flat)


def append_jsonl(file_path: str | pathlib.Path, obj: dict) -> None:
    try:
        file = pathlib.Path(file_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "a", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass


def build_iid_indices(n: int, total_clients: int, client_id: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    chunk = max(1, n // max(1, total_clients))
    start = min(client_id * chunk, n)
    end = n if client_id >= total_clients - 1 else min((client_id + 1) * chunk, n)
    return perm[start:end].tolist()


# -------------------
# Cache/IO utilities
# -------------------
def cleanup_cache_global_rounds(cache_dir: str | pathlib.Path) -> int:
    """Remove legacy cache/global/round_* directories.

    Returns the number of round directories removed.
    """
    import shutil

    base = pathlib.Path(cache_dir)
    gdir = base / "global"
    if not gdir.exists():
        return 0
    removed = 0
    for d in gdir.glob("round_*"):
        try:
            if d.is_dir():
                shutil.rmtree(d)
                removed += 1
        except Exception:
            # best-effort cleanup
            pass
    return removed


def save_params_npz(file_path: str | pathlib.Path, params_pure: dict, *, dtype: np.dtype = np.float16) -> pathlib.Path:
    """Save params (pure dict) to NPZ ensuring parent dir exists; returns Path."""
    p = pathlib.Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    saved = save_npz_dict(p, params_pure, dtype=dtype)
    return pathlib.Path(saved)
