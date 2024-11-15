from __future__ import annotations

from abc import ABC, abstractmethod
from asyncio import gather
from itertools import starmap
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable
    from types import TracebackType
    from typing import Any, Self, TypeAlias

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import AccessModeLiteral, BytesLike

__all__ = ["AccessMode", "ByteGetter", "ByteSetter", "Store", "set_or_delete"]

ByteRangeRequest: TypeAlias = tuple[int | None, int | None]


class AccessMode(NamedTuple):
    """Access mode flags."""

    str: AccessModeLiteral
    readonly: bool
    overwrite: bool
    create: bool
    update: bool

    @classmethod
    def from_literal(cls, mode: AccessModeLiteral) -> Self:
        """
        Create an AccessMode instance from a literal.

        Parameters
        ----------
        mode : AccessModeLiteral
            One of 'r', 'r+', 'w', 'w-', 'a'.

        Returns
        -------
        AccessMode
            The created instance.

        Raises
        ------
        ValueError
            If mode is not one of 'r', 'r+', 'w', 'w-', 'a'.
        """
        if mode in ("r", "r+", "a", "w", "w-"):
            return cls(
                str=mode,
                readonly=mode == "r",
                overwrite=mode == "w",
                create=mode in ("a", "w", "w-"),
                update=mode in ("r+", "a"),
            )
        raise ValueError("mode must be one of 'r', 'r+', 'w', 'w-', 'a'")


class Store(ABC):
    """
    Abstract base class for Zarr stores.
    """

    _mode: AccessMode
    _is_open: bool

    def __init__(self, *args: Any, mode: AccessModeLiteral = "r", **kwargs: Any) -> None:
        self._is_open = False
        self._mode = AccessMode.from_literal(mode)

    @classmethod
    async def open(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Create and open the store.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the store constructor.
        **kwargs : Any
            Keyword arguments to pass to the store constructor.

        Returns
        -------
        Store
            The opened store instance.
        """
        store = cls(*args, **kwargs)
        await store._open()
        return store

    def __enter__(self) -> Self:
        """Enter a context manager that will close the store upon exiting."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the store."""
        self.close()

    async def _open(self) -> None:
        """
        Open the store.

        Raises
        ------
        ValueError
            If the store is already open.
        FileExistsError
            If ``mode='w-'`` and the store already exists.

        Notes
        -----
        * When ``mode='w'`` and the store already exists, it will be cleared.
        """
        if self._is_open:
            raise ValueError("store is already open")
        if self.mode.str == "w":
            await self.clear()
        elif self.mode.str == "w-" and not await self.empty():
            raise FileExistsError("Store already exists")
        self._is_open = True

    async def _ensure_open(self) -> None:
        """Open the store if it is not already open."""
        if not self._is_open:
            await self._open()

    @abstractmethod
    async def empty(self) -> bool:
        """
        Check if the store is empty.

        Returns
        -------
        bool
            True if the store is empty, False otherwise.
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """
        Clear the store.

        Remove all keys and values from the store.
        """
        ...

    @abstractmethod
    def with_mode(self, mode: AccessModeLiteral) -> Self:
        """
        Return a new store of the same type pointing to the same location with a new mode.

        The returned Store is not automatically opened. Call :meth:`Store.open` before
        using.

        Parameters
        ----------
        mode : AccessModeLiteral
            The new mode to use.

        Returns
        -------
        store
            A new store of the same type with the new mode.

        Examples
        --------
        >>> writer = zarr.store.MemoryStore(mode="w")
        >>> reader = writer.with_mode("r")
        """
        ...

    @property
    def mode(self) -> AccessMode:
        """Access mode of the store."""
        return self._mode

    def _check_writable(self) -> None:
        """Raise an exception if the store is not writable."""
        if self.mode.readonly:
            raise ValueError("store mode does not support writing")

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        """Equality comparison."""
        ...

    @abstractmethod
    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRangeRequest | None = None,
    ) -> Buffer | None:
        """Retrieve the value associated with a given key.

        Parameters
        ----------
        key : str
        byte_range : tuple[int | None, int | None], optional

        Returns
        -------
        Buffer
        """
        ...

    @abstractmethod
    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
        """Retrieve possibly partial values from given key_ranges.

        Parameters
        ----------
        key_ranges : Iterable[tuple[str, tuple[int | None, int | None]]]
            Ordered set of key, range pairs, a key may occur multiple times with different ranges

        Returns
        -------
        list of values, in the order of the key_ranges, may contain null/none for missing keys
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the store.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """
        ...

    @property
    @abstractmethod
    def supports_writes(self) -> bool:
        """Does the store support writes?"""
        ...

    @abstractmethod
    async def set(self, key: str, value: Buffer) -> None:
        """Store a (key, value) pair.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        ...

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        """
        Store a key to ``value`` if the key is not already present.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        # Note for implementers: the default implementation provided here
        # is not safe for concurrent writers. There's a race condition between
        # the `exists` check and the `set` where another writer could set some
        # value at `key` or delete `key`.
        if not await self.exists(key):
            await self.set(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        """
        Insert multiple (key, value) pairs into storage.
        """
        await gather(*starmap(self.set, values))

    @property
    @abstractmethod
    def supports_deletes(self) -> bool:
        """Does the store support deletes?"""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a key from the store

        Parameters
        ----------
        key : str
        """
        ...

    @property
    @abstractmethod
    def supports_partial_writes(self) -> bool:
        """Does the store support partial writes?"""
        ...

    @abstractmethod
    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        """Store values at a given key, starting at byte range_start.

        Parameters
        ----------
        key_start_values : list[tuple[str, int, BytesLike]]
            set of key, range_start, values triples, a key may occur multiple times with different
            range_starts, range_starts (considering the length of the respective values) must not
            specify overlapping ranges for the same key
        """
        ...

    @property
    @abstractmethod
    def supports_listing(self) -> bool:
        """Does the store support listing?"""
        ...

    @abstractmethod
    def list(self) -> AsyncIterator[str]:
        """Retrieve all keys in the store.

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    @abstractmethod
    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """
        Retrieve all keys in the store that begin with a given prefix. Keys are returned relative
        to the root of the store.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    @abstractmethod
    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """
        Retrieve all keys and prefixes with a given prefix and which do not contain the character
        “/” after the given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    async def delete_dir(self, prefix: str) -> None:
        """
        Remove all keys and prefixes in the store that begin with a given prefix.
        """
        if not self.supports_deletes:
            raise NotImplementedError
        if not self.supports_listing:
            raise NotImplementedError
        self._check_writable()
        if not prefix.endswith("/"):
            prefix += "/"
        async for key in self.list_prefix(prefix):
            await self.delete(key)

    def close(self) -> None:
        """Close the store."""
        self._is_open = False

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRangeRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        """
        Retrieve a collection of objects from storage. In general this method does not guarantee
        that objects will be retrieved in the order in which they were requested, so this method
        yields tuple[str, Buffer | None] instead of just Buffer | None
        """
        for req in requests:
            yield (req[0], await self.get(*req))


@runtime_checkable
class ByteGetter(Protocol):
    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRangeRequest | None = None
    ) -> Buffer | None: ...


@runtime_checkable
class ByteSetter(Protocol):
    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRangeRequest | None = None
    ) -> Buffer | None: ...

    async def set(self, value: Buffer, byte_range: ByteRangeRequest | None = None) -> None: ...

    async def delete(self) -> None: ...

    async def set_if_not_exists(self, default: Buffer) -> None: ...


async def set_or_delete(byte_setter: ByteSetter, value: Buffer | None) -> None:
    """Set or delete a value in a byte setter

    Parameters
    ----------
    byte_setter : ByteSetter
    value : Buffer | None

    Notes
    -----
    If value is None, the key will be deleted.
    """
    if value is None:
        await byte_setter.delete()
    else:
        await byte_setter.set(value)
