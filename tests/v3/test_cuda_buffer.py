from __future__ import annotations

import numpy as np
import pytest

from zarr.array import AsyncArray
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.cuda_buffer import cuda_prototype
from zarr.store.core import StorePath
from zarr.store.memory import MemoryStore

cupy = pytest.importorskip("cupy")


@pytest.mark.asyncio
async def test_async_array():
    expect = cupy.zeros(10, dtype="uint16", order="F")
    a = await AsyncArray.create(
        StorePath(MemoryStore(mode="w")) / "test_async_array",
        shape=expect.shape,
        chunk_shape=(5,),
        dtype=expect.dtype,
        fill_value=0,
        codecs=[
            BytesCodec(),
            BloscCodec(),
            Crc32cCodec(),
            GzipCodec(),
            ZstdCodec(),
        ],
    )
    expect[5:] = cupy.arange(5)

    # Write into device memory
    await a.setitem(
        selection=(slice(5, 10),),
        value=cupy.arange(5),
        prototype=cuda_prototype,
    )
    # Read into device memory
    got = await a.getitem(selection=(slice(0, 10),), prototype=cuda_prototype)
    assert isinstance(got, cupy.ndarray)
    assert cupy.array_equal(expect, got)

    # Write into host memory
    await a.setitem(
        selection=(slice(0, 5),),
        value=np.arange(5),
    )
    expect[0:5] = cupy.arange(5)

    # Read into host memory
    got = await a.getitem(selection=(slice(0, 10),))
    assert isinstance(got, np.ndarray)
    assert np.array_equal(cupy.asnumpy(expect), got)
