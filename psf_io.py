"""Safe, versioned serialization helpers for PSF data."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def clone_payload(value: Any) -> Any:
    return copy.deepcopy(value)


def _encode_value(value: Any, arrays: Dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):06d}"
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("Object arrays are not supported by safe serialization.")
        arrays[key] = array
        return {
            "__array__": key,
            "shape": [int(v) for v in array.shape],
            "dtype": str(array.dtype),
        }
    if isinstance(value, np.generic):
        return _encode_value(value.item(), arrays)
    if isinstance(value, dict):
        return {str(key): _encode_value(item, arrays) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item, arrays) for item in value]
    if isinstance(value, complex):
        return {"__complex__": [float(value.real), float(value.imag)]}
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return {"__float__": str(value)}
        return value
    raise TypeError(f"Unsupported safe-serialization value: {type(value).__name__}.")


def _decode_value(value: Any, archive) -> Any:
    if isinstance(value, dict) and "__array__" in value:
        key = str(value["__array__"])
        if key not in archive.files:
            raise ValueError(f"Safe result is missing array {key}.")
        array = np.asarray(archive[key])
        expected_shape = tuple(int(v) for v in value.get("shape", []))
        expected_dtype = str(value.get("dtype", ""))
        if expected_shape and tuple(array.shape) != expected_shape:
            raise ValueError(f"Array {key} shape does not match metadata.")
        if expected_dtype and str(array.dtype) != expected_dtype:
            raise ValueError(f"Array {key} dtype does not match metadata.")
        if array.dtype.hasobject:
            raise ValueError("Object arrays are forbidden in safe results.")
        return array.copy()
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(_decode_value(item, archive) for item in value["__tuple__"])
    if isinstance(value, dict) and "__complex__" in value:
        real, imag = value["__complex__"]
        return complex(float(real), float(imag))
    if isinstance(value, dict) and "__float__" in value:
        return float(value["__float__"])
    if isinstance(value, dict):
        return {str(key): _decode_value(item, archive) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item, archive) for item in value]
    return value


def write_safe_container(path, payload: Dict[str, Any], *, format_name: str, version: int) -> None:
    arrays: Dict[str, np.ndarray] = {}
    metadata = {
        "format": str(format_name),
        "version": int(version),
        "payload": _encode_value(payload, arrays),
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def read_safe_container(
    path,
    *,
    expected_format: str,
    supported_versions: Tuple[int, ...],
) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("Safe container is missing metadata_json.")
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        if metadata.get("format") != expected_format:
            raise ValueError(
                f"Unexpected result format {metadata.get('format')!r}; expected {expected_format!r}."
            )
        version = int(metadata.get("version", -1))
        if version not in supported_versions:
            raise ValueError(f"Unsupported result version: {version}.")
        payload = _decode_value(metadata.get("payload"), archive)
    if not isinstance(payload, dict):
        raise ValueError("Safe container payload must be a dictionary.")
    return payload


def has_zip_signature(path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4).startswith(b"PK")
    except OSError:
        return False


def read_unsafe_legacy_pickle(path) -> Dict[str, Any]:
    """Explicit trusted-input compatibility path. Never call automatically."""
    import pickle

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Legacy pickle does not contain a dictionary payload.")
    return payload
