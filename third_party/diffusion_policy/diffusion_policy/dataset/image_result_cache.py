import atexit
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr
from filelock import FileLock
from tqdm import tqdm


CACHE_VERSION = 2
RAM_RESULT_LOCATIONS = {None, "", "ram", "memory", "mem"}
_REGISTERED_CLEANUPS = set()
_CLEANUP_LOGGERS = {}
_SIGNAL_HANDLERS_INSTALLED = False


def use_disk_result_cache(load_result_add) -> bool:
    if load_result_add is None:
        return False
    if isinstance(load_result_add, str):
        return load_result_add.strip().lower() not in RAM_RESULT_LOCATIONS
    return True


def build_cache_metadata(
    *,
    source_type: str,
    dataset_path: str,
    dataset_length: int,
    rgb_keys: Sequence[str],
    image_shapes: Mapping[str, Sequence[int]],
    extra: Optional[Mapping] = None,
) -> Dict:
    metadata = {
        "cache_version": CACHE_VERSION,
        "source_type": source_type,
        "dataset_path": os.path.abspath(os.path.expanduser(dataset_path)),
        "dataset_length": int(dataset_length),
        "rgb_keys": list(rgb_keys),
        "image_shapes": {
            key: [int(v) for v in image_shapes[key]]
            for key in rgb_keys
        },
        "dtype": "float32",
    }
    if extra:
        metadata["extra"] = _jsonable(extra)
    return metadata


def open_or_build_image_result_cache(
    *,
    load_result_add,
    dataset_path: str,
    metadata: Mapping,
    build_frame_fn: Callable[[int], Mapping[str, np.ndarray]],
    desc: str,
    chunk_frames: int = 1,
    logger=None,
) -> Tuple[Dict[str, zarr.Array], str]:
    cache_path = _resolve_cache_path(load_result_add, dataset_path, metadata)
    lock_path = cache_path + ".lock"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    _register_cache_cleanup(cache_path, logger=logger)

    with FileLock(lock_path):
        if _is_valid_cache(cache_path, metadata):
            if logger is not None:
                logger.info("Using decoded image result cache from: %s", cache_path)
        else:
            if logger is not None:
                logger.info("Building decoded image result cache at: %s", cache_path)
            _build_cache(
                cache_path=cache_path,
                metadata=metadata,
                build_frame_fn=build_frame_fn,
                desc=desc,
                chunk_frames=chunk_frames,
            )

    root = zarr.open(cache_path, mode="r")
    return {key: root[key] for key in metadata["rgb_keys"]}, cache_path


def read_image_result(image_data: Mapping[str, object], key: str, indices) -> np.ndarray:
    array = image_data[key]
    try:
        result = array[indices, ...]
    except Exception:
        index_array = np.asarray(indices).reshape(-1)
        result = np.stack([array[int(idx)] for idx in index_array], axis=0)
    return np.asarray(result)


def _build_cache(
    *,
    cache_path: str,
    metadata: Mapping,
    build_frame_fn: Callable[[int], Mapping[str, np.ndarray]],
    desc: str,
    chunk_frames: int,
) -> None:
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    try:
        root = zarr.open(tmp_path, mode="w")
        root.attrs["metadata"] = dict(metadata)
        arrays = {}
        dataset_length = int(metadata["dataset_length"])
        for key in metadata["rgb_keys"]:
            shape = (dataset_length,) + tuple(metadata["image_shapes"][key])
            chunks = (min(max(1, chunk_frames), max(1, dataset_length)),) + tuple(
                metadata["image_shapes"][key]
            )
            arrays[key] = root.create_dataset(
                key,
                shape=shape,
                chunks=chunks,
                dtype=np.float32,
                compressor=None,
                overwrite=True,
            )

        chunk_frames = min(max(1, int(chunk_frames)), max(1, dataset_length))
        buffers = {
            key: np.empty(
                (chunk_frames,) + tuple(metadata["image_shapes"][key]),
                dtype=np.float32,
            )
            for key in metadata["rgb_keys"]
        }

        for chunk_start in tqdm(range(0, dataset_length, chunk_frames), desc=desc):
            chunk_end = min(chunk_start + chunk_frames, dataset_length)
            for frame_idx in range(chunk_start, chunk_end):
                frame_data = build_frame_fn(frame_idx)
                local_idx = frame_idx - chunk_start
                for key, buffer in buffers.items():
                    image = np.asarray(frame_data[key])
                    expected_shape = tuple(metadata["image_shapes"][key])
                    if image.shape != expected_shape:
                        raise ValueError(
                            f"Decoded image {key!r} at frame {frame_idx} has shape {image.shape}, "
                            f"expected {expected_shape}."
                        )
                    buffer[local_idx] = image

            for key, array in arrays.items():
                array[chunk_start:chunk_end] = buffers[key][:(chunk_end - chunk_start)]

        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        os.replace(tmp_path, cache_path)
    except BaseException:
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        raise


def _register_cache_cleanup(cache_path: str, logger=None) -> None:
    if multiprocessing.current_process().name != "MainProcess":
        return

    cache_path = os.path.abspath(cache_path)
    if cache_path in _REGISTERED_CLEANUPS:
        return
    _REGISTERED_CLEANUPS.add(cache_path)
    _CLEANUP_LOGGERS[cache_path] = logger

    def cleanup():
        _cleanup_cache_path(cache_path, logger=logger)

    atexit.register(cleanup)
    _install_signal_cleanup_handlers()


def _cleanup_cache_path(cache_path: str, logger=None) -> None:
    lock_path = cache_path + ".lock"
    removed = False
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
        removed = True
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except OSError:
            pass
    if removed and logger is not None:
        logger.info("Removed decoded image result cache: %s", cache_path)


def _cleanup_registered_caches() -> None:
    for cache_path in list(_REGISTERED_CLEANUPS):
        _cleanup_cache_path(cache_path, logger=_CLEANUP_LOGGERS.get(cache_path))


def _install_signal_cleanup_handlers() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    _SIGNAL_HANDLERS_INSTALLED = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handler = signal.getsignal(signum)

        def handler(received_signum, frame, previous_handler=previous_handler):
            _cleanup_registered_caches()
            if callable(previous_handler):
                previous_handler(received_signum, frame)
                return
            if received_signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + received_signum)

        signal.signal(signum, handler)


def _is_valid_cache(cache_path: str, metadata: Mapping) -> bool:
    if not os.path.isdir(cache_path):
        return False
    try:
        root = zarr.open(cache_path, mode="r")
        cached_metadata = root.attrs.get("metadata", None)
        if cached_metadata != dict(metadata):
            return False
        dataset_length = int(metadata["dataset_length"])
        for key in metadata["rgb_keys"]:
            if key not in root:
                return False
            expected_shape = (dataset_length,) + tuple(metadata["image_shapes"][key])
            if tuple(root[key].shape) != expected_shape:
                return False
            if np.dtype(root[key].dtype) != np.dtype(np.float32):
                return False
        return True
    except Exception:
        return False


def _resolve_cache_path(load_result_add, dataset_path: str, metadata: Mapping) -> str:
    location = os.path.expanduser(str(load_result_add))
    if location.lower() == "ssd":
        location = os.path.join(
            os.path.dirname(os.path.abspath(os.path.expanduser(dataset_path))),
            ".decoded_image_result_cache",
        )

    if location.endswith(".zarr"):
        return os.path.abspath(location)

    fingerprint = _metadata_fingerprint(metadata)
    dataset_name = os.path.basename(os.path.abspath(os.path.expanduser(dataset_path)).rstrip(os.sep))
    if not dataset_name:
        dataset_name = "dataset"
    cache_name = f"{dataset_name}_decoded_images_{fingerprint}.zarr"
    return os.path.abspath(os.path.join(location, cache_name))


def _metadata_fingerprint(metadata: Mapping) -> str:
    encoded = json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
