=======
Writers
=======

This section covers the common writer abstractions and format-specific writer
details.

Writer State Machine
====================

Every DWIO writer follows a state machine defined in
``dwio/common/Writer.h``:

.. code-block:: text

   Init ──write()──► Running ──finish()──► Finishing ──close()──► Closed
                        │                      │
                        │ abort()              │ (time-sliced)
                        ▼                      │
                     Aborted                   │
                                               ▼
                                            Closed

   Running may also transition directly to Closed via close() when the
   writer has nothing to finalize (e.g. some physical file writers).

* ``write(RowVectorPtr)`` -- appends a batch. Transitions from Init to
  Running on the first call.
* ``finish()`` -- signals that no more data will be written. For sorting
  writers this triggers the sort and output phase, which may be time-sliced
  across multiple ``finish()`` calls.
* ``close()`` -- writes the file footer and closes the ``FileSink``.
* ``abort()`` -- discards all buffered data. The writer cannot be reused.

FlushPolicy
===========

``FlushPolicy`` (``dwio/common/FlushPolicy.h``) decides when a stripe (or
row group) should be flushed to the output:

.. code-block:: cpp

   bool shouldFlush(const StripeProgress& progress);

``StripeProgress`` provides:

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - Field
     - Description
   * - ``stripeIndex``
     - Index of the current stripe being written.
   * - ``stripeRowCount``
     - Number of rows written to the current stripe.
   * - ``totalMemoryUsage``
     - Total memory usage across all buffered data.
   * - ``stripeSizeEstimate``
     - Estimated size of the stripe in bytes.

A typical policy flushes when ``stripeSizeEstimate`` exceeds a threshold
(default 64 MB for DWRF, 128 MB for Parquet).

SortingWriter
=============

``SortingWriter`` (``dwio/common/SortingWriter.h``) wraps another writer to
produce globally sorted output:

1. ``write()`` -- all incoming batches are appended to a ``SortBuffer`` in
   memory.
2. ``finish()`` -- the ``SortBuffer`` sorts all rows by the specified sort
   columns. Output is written to the underlying writer in time-sliced chunks
   so that a single ``finish()`` call does not block the thread for too long.
3. When memory pressure is high, the ``SortBuffer`` can spill sorted runs to
   disk and merge-sort them during output.

FileSink
========

``FileSink`` (``dwio/common/FileSink.h``) is the abstract output interface.
Writers write bytes to a ``FileSink``; the sink handles storage-specific
details.

Built-in sinks:

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - Sink
     - Description
   * - ``LocalFileSink``
     - Writes to the local filesystem.
   * - ``WriteFileSink``
     - Writes through an abstract ``WriteFile`` interface (e.g. S3, HDFS).
   * - ``MemorySink``
     - Writes to an in-memory buffer. Useful for testing.

Custom sinks are registered through a factory pattern similar to
``ReaderFactory``.

DWRF Writer Specifics
=====================

The DWRF writer (``dwio/dwrf/writer/Writer.h``) adds several layers on top
of the common writer framework:

**ColumnWriter tree.** Each column in the schema has a corresponding
``ColumnWriter`` that encodes values into streams. Complex types (struct,
list, map) create a tree of writers mirroring the schema.

**Dictionary encoding selection.** String columns start in dictionary mode.
If the dictionary grows beyond a configurable fraction of distinct values
(``DICTIONARY_NUMERIC_KEY_SIZE_THRESHOLD``), the writer falls back to direct
encoding for subsequent stripes.

**Memory pools.** The DWRF writer uses three memory pools:

.. list-table::
   :widths: 25 40
   :header-rows: 1

   * - Pool
     - Purpose
   * - ``DICTIONARY``
     - Stores dictionary entries during encoding.
   * - ``OUTPUT_STREAM``
     - Buffers compressed stream output before writing to the sink.
   * - ``GENERAL``
     - General allocations (row index, stripe footer serialization).

**Compression ratio tracking.** The writer tracks the observed compression
ratio per stream to estimate stripe sizes before flushing.

Parquet Writer Specifics
========================

The Parquet writer (``dwio/parquet/writer/Writer.h``) bridges Velox to the
Arrow Parquet library:

* **Arrow bridge.** Velox vectors are exported via the Arrow C Data Interface
  and written by Arrow's ``parquet::arrow::FileWriter``.
* **WriterOptions.** Key options include:

  * ``compression`` -- per-column or global compression codec.
  * ``encoding`` -- prefer dictionary, plain, or format-default encoding.
  * ``timestampUnit`` -- MILLIS or MICROS for timestamp columns.
  * ``parquetFieldIds`` -- Iceberg-style field IDs written into the schema
    metadata.
