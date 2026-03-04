==============
Parquet Format
==============

Velox supports reading and writing Apache Parquet files through the DWIO
layer. The Parquet reader is a native C++ implementation that plugs into the
``SelectiveColumnReader`` framework. The writer bridges to the Arrow Parquet
library.

File Layout
===========

.. code-block:: text

   ┌────────────────────────────────┐
   │  Magic: "PAR1" (4 bytes)      │
   ├────────────────────────────────┤
   │  Row Group 0                  │
   │  ┌─────────────────────────┐  │
   │  │  Column Chunk 0         │  │
   │  │  ┌───────────────────┐  │  │
   │  │  │ Dictionary Page   │  │  │  Optional dictionary
   │  │  ├───────────────────┤  │  │
   │  │  │ Data Page 0       │  │  │  Encoded values + rep/def levels
   │  │  │ Data Page 1       │  │  │
   │  │  │ ...               │  │  │
   │  │  └───────────────────┘  │  │
   │  │  Column Chunk 1         │  │
   │  │  ...                    │  │
   │  └─────────────────────────┘  │
   ├────────────────────────────────┤
   │  Row Group 1                  │
   │  ...                          │
   ├────────────────────────────────┤
   │  File Footer (Thrift)         │  Schema, row group metadata, statistics
   ├────────────────────────────────┤
   │  Footer Length (4 bytes)       │
   ├────────────────────────────────┤
   │  Magic: "PAR1" (4 bytes)      │
   └────────────────────────────────┘

Encodings
=========

The Velox Parquet reader supports the following page encodings:

.. list-table::
   :widths: 25 40
   :header-rows: 1

   * - Encoding
     - Description
   * - ``PLAIN``
     - Values stored directly in their binary representation.
   * - ``RLE_DICTIONARY``
     - Dictionary encoding with RLE/bit-packed indices.
   * - ``DELTA_BINARY_PACKED``
     - Delta encoding for integers. See ``dwio/parquet/reader/DeltaBpDecoder.h``.
   * - ``DELTA_BYTE_ARRAY``
     - Prefix + suffix delta encoding for byte arrays.
   * - ``BYTE_STREAM_SPLIT``
     - Byte-interleaved encoding for floating-point values.

Repetition and definition levels are always RLE/bit-packed. See
``dwio/parquet/reader/RleBpDecoder.h``.

Dremel Encoding for Nested Types
================================

Parquet uses the Dremel encoding scheme to flatten nested structures (structs,
lists, maps) into flat columns with *repetition levels* and *definition
levels*:

* **Definition level** -- how many optional/repeated ancestors are defined
  (non-null) for this value.
* **Repetition level** -- which repeated ancestor has a new entry at this
  position.

Velox preloads all repetition and definition levels for an entire column chunk
before decoding values. ``NestedStructureDecoder``
(``dwio/parquet/reader/NestedStructureDecoder.h``) converts the flat
rep/def arrays into the offsets and null bitmaps needed to reconstruct the
nested Velox vectors.

Type Mapping
============

Parquet physical types are mapped to Velox types using a combination of the
physical type and the logical type annotation:

.. list-table::
   :widths: 20 20 20
   :header-rows: 1

   * - Parquet Physical
     - Logical Type
     - Velox Type
   * - ``BOOLEAN``
     - --
     - ``BOOLEAN``
   * - ``INT32``
     - --
     - ``INTEGER``
   * - ``INT32``
     - ``DATE``
     - ``DATE``
   * - ``INT32``
     - ``DECIMAL(p <= 9)``
     - ``SHORT_DECIMAL``
   * - ``INT64``
     - --
     - ``BIGINT``
   * - ``INT64``
     - ``TIMESTAMP_MILLIS``
     - ``TIMESTAMP``
   * - ``INT64``
     - ``TIMESTAMP_MICROS``
     - ``TIMESTAMP``
   * - ``INT96``
     - --
     - ``TIMESTAMP`` (legacy Hive/Impala)
   * - ``FLOAT``
     - --
     - ``REAL``
   * - ``DOUBLE``
     - --
     - ``DOUBLE``
   * - ``BYTE_ARRAY``
     - --
     - ``VARCHAR``
   * - ``BYTE_ARRAY``
     - ``STRING``
     - ``VARCHAR``
   * - ``FIXED_LEN_BYTE_ARRAY``
     - ``DECIMAL(p > 18)``
     - ``LONG_DECIMAL``

Schema evolution rules and type promotion are handled in
``dwio/parquet/reader/ParquetTypeWithId.h``.

Read Path
=========

1. **Footer parsing.** The reader reads the file footer (Thrift-encoded) to
   obtain the schema, row group metadata, and column statistics.
2. **Row group selection.** ``filterRowGroups()`` tests column statistics
   against the ``ScanSpec`` filters to skip entire row groups.
3. **Page-level decoding.** ``PageReader``
   (``dwio/parquet/reader/PageReader.h``) reads dictionary and data pages
   within each column chunk. It feeds decoded values into the
   ``SelectiveColumnReader`` pipeline.

Write Path
==========

The Velox Parquet writer (``dwio/parquet/writer/Writer.h``) uses the Arrow C
Data Interface to pass Velox vectors to the Arrow Parquet ``FileWriter``:

.. code-block:: text

   RowVector
     │
     │  Arrow C Data Interface (zero-copy when possible)
     ▼
   Arrow RecordBatch
     │
     ▼
   Arrow Parquet FileWriter
     │
     ▼
   FileSink (DWIO)

Writer options include compression codec selection, encoding preferences
(e.g. dictionary vs. plain), timestamp units (millis vs. micros), and Iceberg
field IDs.
