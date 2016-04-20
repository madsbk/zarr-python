# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from functools import reduce  # TODO PY2 compatibility
import operator
import itertools
import multiprocessing


import numpy as np


from zarr import blosc
from zarr.sync import SynchronizedAttributes
from zarr.util import is_total_slice, normalize_array_selection, \
    get_chunk_range, human_readable_size


# import sys
#
# def log(*msg):
#     print(*msg, file=sys.stderr)
#     sys.stderr.flush()


_blosc_use_context = False


def set_blosc_options(use_context=False, nthreads=None):
    """Set options for how the blosc compressor is used.

    Parameters
    ----------
    use_context : bool, optional
        If False, blosc will be used in non-contextual mode, which is best
        when using zarr in a single-threaded environment because it allows
        blosc to use multiple threads internally. If True, blosc will be used
        in contextual mode, which is better when using zarr in a
        multi-threaded environment like dask.array because it avoids the blosc
        global lock and so multiple blosc operations can be running
        concurrently.
    nthreads : int, optional
        Number of internal threads to use when running blosc in non-contextual
        mode.

    """
    global _blosc_use_context
    _blosc_use_context = use_context
    if not use_context:
        if nthreads is None:
            # diminishing returns beyond 4 threads?
            nthreads = min(4, multiprocessing.cpu_count())
        blosc.set_nthreads(nthreads)


_repr_shuffle = [
    '0 (NOSHUFFLE)',
    '1 (BYTESHUFFLE)',
    '2 (BITSHUFFLE)',
]


class Array(object):

    def __init__(self, store):
        """Instantiate an array.

        Parameters
        ----------
        store : zarr.store.base.ArrayStore
            Array store.

        """

        self._store = store
        self._shape = store.meta['shape']
        self._chunks = store.meta['chunks']
        self._dtype = store.meta['dtype']
        self._cname = store.meta['cname']
        self._clevel = store.meta['clevel']
        self._shuffle = store.meta['shuffle']
        self._fill_value = store.meta['fill_value']
        self._attrs = store.attrs

    @property
    def store(self):
        return self._store

    @property
    def shape(self):
        return self._shape

    @property
    def chunks(self):
        return self._chunks

    @property
    def dtype(self):
        return self._dtype

    @property
    def cname(self):
        return self._cname

    @property
    def clevel(self):
        return self._clevel

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def attrs(self):
        return self._attrs

    @property
    def cbytes(self):
        # pass through
        return self._store.cbytes

    @property
    def initialized(self):
        # pass through
        return self._store.initialized

    @property
    def size(self):
        return reduce(operator.mul, self._shape)

    @property
    def itemsize(self):
        return self._dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

    @property
    def cdata_shape(self):
        return tuple(
            int(np.ceil(s / c)) for s, c in zip(self._shape, self._chunks)
        )

    # methods

    def __getitem__(self, item):

        # normalize selection
        selection = normalize_array_selection(item, self._shape)

        # determine output array shape
        out_shape = tuple(stop - start for start, stop in selection)

        # setup output array
        out = np.empty(out_shape, dtype=self._dtype)

        # determine indices of chunks overlapping the selection
        chunk_range = get_chunk_range(selection, self._chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self._chunks)]

            # determine region within output array
            out_selection = tuple(
                slice(max(0, o - start), min(o + c - start, stop - start))
                for (start, stop), o, c, in zip(selection, offset, self._chunks)
            )

            # determine region within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self._chunks)
            )

            # obtain the destination array as a view of the output array
            dest = out[out_selection]

            # load chunk selection into output array
            self._chunk_getitem(cidx, chunk_selection, dest)

        return out

    def __array__(self):
        return self[:]

    def __setitem__(self, key, value):

        # normalize selection
        selection = normalize_array_selection(key, self._shape)

        # determine indices of chunks overlapping the selection
        chunk_range = get_chunk_range(selection, self._chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self._chunks)]

            # determine required index range within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self._chunks)
            )

            if np.isscalar(value):

                # put data
                self._chunk_setitem(cidx, chunk_selection, value)

            else:
                # assume value is array-like

                # determine index within value
                value_selection = tuple(
                    slice(max(0, o - start), min(o + c - start, stop - start))
                    for (start, stop), o, c, in zip(selection, offset,
                                                    self._chunks)
                )

                # put data
                self._chunk_setitem(cidx, chunk_selection,
                                    value[value_selection])

    def _chunk_getitem(self, cidx, item, dest):
        """Obtain part or whole of a chunk.

        Parameters
        ----------
        cidx : tuple of ints
            Indices of the chunk.
        item : tuple of slices
            Location of region within the chunk.
        dest : ndarray
            Numpy array to store result in.

        """

        # override this in sub-classes, e.g., if need to use a lock

        try:

            # obtain compressed data for chunk
            ckey = '.'.join(map(str, cidx))
            cdata = self._store.data[ckey]

        except KeyError:

            # chunk not initialized
            if self._fill_value is not None:
                dest.fill(self._fill_value)

        else:

            if is_total_slice(item, self._chunks) and dest.flags.c_contiguous:

                # optimisation: we want the whole chunk, and the destination is
                # C contiguous, so we can decompress directly from the chunk
                # into the destination array
                blosc.decompress(cdata, dest, _blosc_use_context)

            else:

                # decompress chunk
                chunk = np.empty(self._chunks, dtype=self._dtype)
                blosc.decompress(cdata, chunk, _blosc_use_context)

                # set data in output array
                # (split into two lines for profiling)
                tmp = chunk[item]
                dest[:] = tmp

    def _chunk_setitem(self, cidx, key, value):
        """Replace part or whole of a chunk.

        Parameters
        ----------
        cidx : tuple of ints
            Indices of the chunk.
        key : tuple of slices
            Location of region within the chunk.
        value : scalar or ndarray
            Value to set.

        """

        # override this in sub-classes, e.g., if need to use a lock

        if is_total_slice(key, self._chunks):

            # optimisation: we are completely replacing the chunk, so no need
            # to access the existing chunk data

            if np.isscalar(value):

                # setup array filled with value
                chunk = np.empty(self._chunks, dtype=self._dtype)
                chunk.fill(value)

            else:

                # ensure array is C contiguous
                chunk = np.ascontiguousarray(value, dtype=self._dtype)

        else:
            # partially replace the contents of this chunk

            try:

                # obtain compressed data for chunk
                ckey = '.'.join(map(str, cidx))
                cdata = self._store.data[ckey]

            except KeyError:

                # chunk not initialized
                chunk = np.empty(self._chunks, dtype=self._dtype)
                if self._fill_value is not None:
                    chunk.fill(self._fill_value)

            else:

                # decompress chunk
                chunk = np.empty(self._chunks, dtype=self._dtype)
                blosc.decompress(cdata, chunk, _blosc_use_context)

            # modify
            chunk[key] = value

        # compress
        cdata = blosc.compress(chunk, self._cname, self._clevel,
                               self._shuffle, _blosc_use_context)

        # store
        ckey = '.'.join(map(str, cidx))
        self._store.data[ckey] = cdata

    def __repr__(self):
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        r += '%s' % str(self._shape)
        r += ', %s' % str(self._dtype)
        r += ', chunks=%s' % str(self._chunks)
        r += ')'
        r += '\n  cname: %s' % str(self._cname, 'ascii')
        r += '; clevel: %s' % self._clevel
        r += '; shuffle: %s' % _repr_shuffle[self._shuffle]
        r += '\n  nbytes: %s' % human_readable_size(self.nbytes)
        r += '; cbytes: %s' % human_readable_size(self.cbytes)
        if self.cbytes > 0:
            r += '; ratio: %.1f' % (self.nbytes / self.cbytes)
        n_chunks = reduce(operator.mul, self.cdata_shape)
        r += '; initialized: %s/%s' % (self.initialized, n_chunks)
        r += '\n  store: %s.%s' % (type(self._store).__module__,
                                   type(self._store).__name__)
        return r

    def __str__(self):
        return repr(self)

    def resize(self, *args):
        # pass through
        self._store.resize(*args)
        # update shape after resize
        self._shape = self._store.meta['shape']

    def append(self, data, axis=0):
        """Append `data` to `axis`.

        Parameters
        ----------
        data : array_like
            Data to be appended.
        axis : int
            Axis along which to append.

        Notes
        -----
        The size of all dimensions other than `axis` must match between this
        array and `data`.

        """

        # ensure data is array-like
        if not hasattr(data, 'shape') or not hasattr(data, 'dtype'):
            data = np.asanyarray(data)

        # ensure shapes are compatible for non-append dimensions
        self_shape_preserved = tuple(s for i, s in enumerate(self._shape)
                                     if i != axis)
        data_shape_preserved = tuple(s for i, s in enumerate(data.shape)
                                     if i != axis)
        if self_shape_preserved != data_shape_preserved:
            raise ValueError('shapes not compatible')

        # remember old shape
        old_shape = self._shape

        # determine new shape
        new_shape = tuple(
            self._shape[i] if i != axis else self._shape[i] + data.shape[i]
            for i in range(len(self._shape))
        )

        # resize
        self.resize(new_shape)

        # store data
        # noinspection PyTypeChecker
        append_selection = tuple(
            slice(None) if i != axis else slice(old_shape[i], new_shape[i])
            for i in range(len(self._shape))
        )
        self[append_selection] = data


class SynchronizedArray(Array):

    def __init__(self, store, synchronizer):
        super(SynchronizedArray, self).__init__(store)
        self._synchronizer = synchronizer
        # wrap attributes
        self._attrs = SynchronizedAttributes(store.attrs, synchronizer)

    def _chunk_setitem(self, cidx, key, value):
        with self._synchronizer.lock_chunk(cidx):
            super(SynchronizedArray, self)._chunk_setitem(cidx, key, value)

    def resize(self, *args):
        with self._synchronizer.lock_array():
            super(SynchronizedArray, self).resize(*args)

    def append(self, data, axis=0):
        with self._synchronizer.lock_array():
            super(SynchronizedArray, self).append(data, axis=axis)

    def __repr__(self):
        r = super(SynchronizedArray, self).__repr__()
        r += ('\n  synchronizer: %s.%s' %
              (type(self._synchronizer).__module__,
               type(self._synchronizer).__name__))
        return r
