****
DWIO
****

DWIO (Data Warehouse I/O) is Velox's format-agnostic I/O layer for reading and
writing columnar file formats. It provides a unified abstraction over multiple
file formats so that the rest of the engine -- connectors, operators, and the
execution framework -- can work with data files without depending on
format-specific details.

Supported Formats
=================

.. list-table::
   :widths: 15 50
   :header-rows: 1

   * - Format
     - Description
   * - DWRF
     - Meta's columnar format derived from ORC. Supports FlatMap encoding,
       column-level encryption, and stripe-level indexing.
   * - Parquet
     - Apache Parquet. Supports Dremel encoding for nested types, dictionary
       encoding, and multiple compression codecs.
   * - ORC
     - Apache ORC. Read support shares much of the DWRF code path.
   * - Text
     - Delimited text (CSV / TSV) files.
   * - Nimble
     - Experimental next-generation columnar format.

Relationship to Connectors
==========================

DWIO sits below the :doc:`Connector <connectors>` layer. A connector such as the
Hive Connector creates a ``HiveDataSource`` which in turn uses a DWIO
``Reader`` to decode file data. The boundary is:

* **Connector** -- understands splits, partitioning, bucketing, and dynamic
  filter pushdown. Owns the ``DataSource`` / ``DataSink`` interfaces.
* **DWIO** -- understands file layouts, encoding, I/O scheduling, and column
  projection. Owns the ``Reader`` / ``Writer`` interfaces and their factories.

.. code-block:: text

   ┌──────────────┐
   │  Connector   │   HiveDataSource / HiveDataSink
   └──────┬───────┘
          │ creates
   ┌──────▼───────┐
   │  DWIO Layer  │   ReaderFactory / WriterFactory
   └──────┬───────┘
          │ reads / writes
   ┌──────▼───────┐
   │  Storage     │   S3, HDFS, GCS, ABFS, local FS
   └──────────────┘

Topics
======

.. toctree::
    :maxdepth: 1

    dwio/architecture
    dwio/scan-and-filter
    dwio/io-buffering
    dwio/dwrf
    dwio/parquet
    dwio/nimble
    dwio/writers
