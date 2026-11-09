from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ShmArray:
    """Thin wrapper around multiprocessing.shared_memory for 1D/2D/ND arrays.

    - Lazily (re)allocates a SharedMemory block when shape/dtype changes.
    - Provides numpy view access and convenience write/describe helpers.
    - Call close() to release and unlink explicitly if needed.
    """

    shm: Optional[SharedMemory] = None
    shape: Tuple[int, ...] = ()
    dtype: np.dtype = np.dtype("float32")

    def ensure(self, shape: Tuple[int, ...], dtype: np.dtype) -> None:
        """Ensure shared memory block matches shape and dtype."""
        dtype = np.dtype(dtype)
        if self.shm is not None and self.shape == shape and self.dtype == dtype:
            return
        # Recreate if changed
        self.close()
        nbytes = int(np.prod(shape) * dtype.itemsize)
        self.shm = SharedMemory(create=True, size=nbytes)
        self.shape = shape
        self.dtype = dtype

    def as_array(self) -> np.ndarray:
        assert self.shm is not None
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def write(self, data: np.ndarray) -> None:
        arr = np.asarray(data)
        self.ensure(tuple(arr.shape), arr.dtype)
        np.copyto(self.as_array(), arr)

    def desc(self) -> Dict[str, Any]:
        assert self.shm is not None
        return {
            "name": self.shm.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
        }

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None


def attach_ndarray(name: str, shape: Tuple[int, ...], dtype: str):
    """Attach to an existing shared memory block and return (shm, ndarray).

    Caller is responsible for closing and unlinking if desired.
    """
    shm = SharedMemory(name=name)
    arr = np.ndarray(shape=shape, dtype=np.dtype(dtype), buffer=shm.buf)
    return shm, arr

