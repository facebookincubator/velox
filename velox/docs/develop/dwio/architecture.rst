============
Architecture
============

DWIO provides a layered architecture for reading and writing columnar files.
Format-specific logic is isolated behind abstract interfaces so that connectors
and operators interact with a single API regardless of the underlying file
format.

Reader/Writer Factory Pattern
=============================

Readers and writers are created through static registries.

**Reader registration** (``dwio/common/ReaderFactory.h``):

.. code-block:: cpp

   // Registration (typically called once during startup)
   dwio::common::registerReaderFactory(std::make_shared<DwrfReaderFactory>());

   // Lookup
   auto factory = dwio::common::getReaderFactory(FileFormat::DWRF);
   auto reader = factory->createReader(input, options);

**Writer registration** (``dwio/common/WriterFactory.h``):

.. code-block:: cpp

   dwio::common::registerWriterFactory(std::make_shared<DwrfWriterFactory>());
   auto factory = dwio::common::getWriterFactory(FileFormat::DWRF);
   auto writer = factory->createWriter(sink, options);

Each factory is keyed by a ``FileFormat`` enum value (``DWRF``, ``PARQUET``,
``ORC``, ``TEXT``, ``NIMBLE``, etc.).

Reader Hierarchy
================

The read path is a three-level hierarchy defined in
``dwio/common/Reader.h``:

.. list-table::
   :widths: 15 45
   :header-rows: 1

   * - Class
     - Role
   * - ``Reader``
     - Wraps a file handle. Parses file-level metadata (footer, schema,
       encryption). Provides ``createRowReader()`` to begin scanning.
   * - ``RowReader``
     - Iterates over row groups / stripes. Owns the ``ScanSpec`` (column
       projection + filters). Calls ``next()`` to produce output
       ``RowVector`` batches.
   * - ``SelectiveColumnReader``
     - Per-column decoder. Reads encoded data, applies inline filters via
       ``ColumnVisitor``, and materializes passing rows into vectors. Delegates
       format-specific operations to ``FormatData``.

Writer Hierarchy
================

The write path is modeled as a state machine defined in
``dwio/common/Writer.h``:

.. code-block:: text

   ┌──────┐   write()   ┌─────────┐   flush()   ┌──────────┐   close()   ┌────────┐
   │ Init ├────────────►│ Running ├────────────►│ Finishing ├───────────►│ Closed │
   └──────┘             └────┬────┘             └──────────┘            └────────┘
                             │ abort()
                        ┌────▼────┐
                        │ Aborted │
                        └─────────┘

* ``write(RowVectorPtr)`` -- appends a batch of rows.
* ``flush()`` -- writes buffered data (e.g. a complete stripe).
* ``close()`` -- finalizes the file (writes footer, closes sink).
* ``abort()`` -- discards state; no file is produced.

FormatData Abstraction
======================

``SelectiveColumnReader`` is format-agnostic. It delegates format-specific
operations to a ``FormatData`` object (``dwio/common/FormatData.h``):

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - Method
     - Purpose
   * - ``readNulls()``
     - Decodes the null bitmap in the format's native encoding.
   * - ``seekToRowGroup()``
     - Positions streams to the start of a specific row group or stride.
   * - ``shouldSkipRowGroup()``
     - Tests column-level statistics against the filter to decide whether a row
       group can be skipped entirely.

Each file format (DWRF, Parquet, ...) provides its own ``FormatData``
subclass.

End-to-End Read Pipeline
========================

.. code-block:: text

   Connector
     │
     ▼
   ReaderFactory::createReader(input, options)
     │
     ▼
   Reader          ← parses footer, schema, encryption
     │
     │ createRowReader(scanSpec)
     ▼
   RowReader        ← iterates stripes / row groups
     │
     │ next(batchSize, output)
     ▼
   UnitLoader       ← loads / prefetches stripe data
     │
     ▼
   BufferedInput    ← coalesces I/O, optional caching
     │
     ▼
   SelectiveColumnReader  ← decodes + filters per column
     │
     ▼
   Output RowVector

End-to-End Write Pipeline
=========================

.. code-block:: text

   Connector
     │
     ▼
   WriterFactory::createWriter(sink, options)
     │
     ▼
   Writer           ← state machine (Init → Running → Closed)
     │
     │ write(RowVector)
     ▼
   ColumnWriter     ← encodes columns (dictionary, RLE, etc.)
     │
     │ flush triggered by FlushPolicy
     ▼
   FlushPolicy      ← decides stripe boundaries
     │
     ▼
   FileSink         ← writes bytes to storage
