from __future__ import annotations

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

    await a.setitem(
        selection=(slice(5, 10),),
        value=cupy.arange(5),
        prototype=cuda_prototype,
    )
    got = await a.getitem(selection=(slice(0, 10),), prototype=cuda_prototype)
    assert cupy.array_equal(expect, got)
