from __future__ import annotations

from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

import numpy as np
import numpy.typing as npt

from zarr.buffer import ArrayLike, Buffer, BufferPrototype, NDArrayLike, NDBuffer

if TYPE_CHECKING:
    from typing_extensions import Self

try:
    import cupy
except ImportError as e:
    raise ImportError("please install cupy") from e


class CudaBuffer(Buffer):
    @classmethod
    def create_zero_length(cls) -> Self:
        return cls(cupy.array([], dtype="b"))

    @classmethod
    def from_array_like(cls, array_like: ArrayLike) -> Self:
        return cls(cupy.asarray(array_like))

    def as_numpy_array(self) -> npt.NDArray[Any]:
        return cast(npt.NDArray[Any], cupy.asnumpy(self._data))

    def __add__(self, other: Buffer) -> Self:
        other_array = other.as_array_like()
        assert other_array.dtype == np.dtype("b")
        return self.__class__(
            cupy.concatenate((cupy.asarray(self._data), cupy.asarray(other_array)))
        )


class CudaNDBuffer(NDBuffer):
    @classmethod
    def create(
        cls,
        *,
        shape: Iterable[int],
        dtype: npt.DTypeLike,
        order: Literal["C", "F"] = "C",
        fill_value: Any | None = None,
    ) -> Self:
        ret = cls(cupy.empty(shape=tuple(shape), dtype=dtype, order=order))
        if fill_value is not None:
            ret.fill(fill_value)
        return ret

    @classmethod
    def from_ndarray_like(cls, ndarray_like: NDArrayLike) -> Self:
        return cls(cupy.asarray(ndarray_like))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, NDBuffer):
            value = value._data
        self._data.__setitem__(key, cupy.asarray(value))

    def __getitem__(self, key: Any) -> Self:
        return self.__class__(cupy.asanyarray(self._data.__getitem__(key)))


cuda_prototype = BufferPrototype(buffer=CudaBuffer, nd_buffer=CudaNDBuffer)
