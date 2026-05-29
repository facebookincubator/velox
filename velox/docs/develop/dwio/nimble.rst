=============
Nimble Format
=============

Nimble is Meta's next-generation columnar file format designed for
high-performance analytics. The format provides a flexible encoding system
with cost-based encoding selection and supports advanced features such as
FlatMap encoding, cluster indexing, and chunk-based memory management. The
implementation lives under ``dwio/nimble/``.

.. note::

   Nimble is currently experimental and under active development. The API
   and file format may change without notice.

File Layout
===========

A Nimble file (called a *tablet*) organizes data into stripes, grouped into
*stripe groups*. The file is read from the tail (last 20 bytes):

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │  Stripe 0..M Streams (Stripe Group 0)   │
   ├─────────────────────────────────────────┤
   │  Stripe Group 0 Metadata                │  Per-stream offsets and sizes
   ├─────────────────────────────────────────┤
   │  (optional) Index Group 0 Metadata      │  Cluster index data
   ├─────────────────────────────────────────┤
   │  ... more stripe groups ...             │
   ├─────────────────────────────────────────┤
   │  Stripes Metadata                       │  Row counts, offsets, sizes,
   │                                         │  group indices per stripe
   ├─────────────────────────────────────────┤
   │  Optional Sections                      │  "schema", "stats", "index"
   ├─────────────────────────────────────────┤
   │  Footer                                 │  Total row count, section refs
   ├─────────────────────────────────────────┤
   │  Postscript (20 bytes)                  │
   │  ┌───────────────────────────────────┐  │
   │  │ Footer Size (4B)                  │  │
   │  │ Compression Type (1B)             │  │
   │  │ Checksum Type (1B)                │  │
   │  │ Checksum (8B, XXH3_64)            │  │
   │  │ Major Version (2B)                │  │
   │  │ Minor Version (2B)                │  │
   │  │ Magic "NI" (2B)                   │  │
   │  └───────────────────────────────────┘  │
   └─────────────────────────────────────────┘

Reading starts from the last 2 bytes (magic ``"NI"``), then reads the full
20-byte Postscript, then the Footer, and finally seeks to individual stripes
on demand. The ``TabletReader`` (``tablet/TabletReader.h``) performs a
speculative tail read (default 8 MB) to fetch the postscript, footer, and
trailing metadata in a single I/O.

Metadata is serialized using FlatBuffers (``tablet/Footer.fbs``,
``velox/Schema.fbs``, ``velox/Stats.fbs``).

Schema and Type System
======================

Nimble has its own schema system bridged to Velox types during read and write.

**Schema kinds** (``velox/SchemaTypes.h``):

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - Kind
     - Description
   * - ``Scalar``
     - Single data stream (integers, floats, bools, strings).
   * - ``Row``
     - Nulls stream + named children (struct).
   * - ``Array``
     - Lengths stream + elements.
   * - ``Map``
     - Lengths stream + keys + values.
   * - ``FlatMap``
     - Nulls stream + per-key ``IN_MAP`` descriptors + per-key value children.
   * - ``ArrayWithOffsets``
     - Offsets + lengths + elements (dictionary arrays).
   * - ``SlidingWindowMap``
     - Offsets + lengths + keys + values.

The schema is stored as a flat DFS list of ``SchemaNode`` entries in the file
footer. ``SchemaReader`` (``velox/SchemaReader.h``) reconstructs the tree on
read. ``SchemaBuilder`` (``velox/SchemaBuilder.h``) constructs it
incrementally during writing, supporting dynamic FlatMap key addition across
stripes.

Encoding Types
==============

Nimble supports a variety of encodings (``common/Types.h``):

.. list-table::
   :widths: 20 50
   :header-rows: 1

   * - Encoding
     - Description
   * - ``Trivial``
     - Native encoding for numerics, packed chars with offsets for strings,
       bitpacked for booleans. All data types supported.
   * - ``RLE``
     - Run-length encoding. Run lengths are bit-packed; run values use
       trivial encoding. All data types supported.
   * - ``Dictionary``
     - Unique values stored in a dictionary with indices into that
       dictionary. All data types except booleans supported.
   * - ``FixedBitWidth``
     - Integer types packed into a fixed number of bits (smallest width
       required to represent the largest element). Non-negative values only.
   * - ``Nullable``
     - Wraps one sub-encoding for non-null values with another sub-encoding
       marking which rows are null.
   * - ``Sentinel``
     - Stores nullable data using a sentinel value to represent nulls in a
       single non-nullable encoding.
   * - ``SparseBool``
     - Stores indices to set (or unset) bits. Useful for sparse data such as
       columns where only a few rows are non-null.
   * - ``Varint``
     - Variable-length integer encoding for non-negative values.
   * - ``Delta``
     - Stores integer types with delta encoding (positive deltas only).
   * - ``Constant``
     - Stores constant data (single unique value repeated for all rows).
   * - ``MainlyConstant``
     - Stores mainly-constant data. One value is treated as special; a bool
       child vector marks which rows have the special value; non-special
       values are stored separately.
   * - ``Prefix``
     - Sorted string prefix compression.

Encodings can be nested. For example, a ``Nullable`` encoding wraps a
non-null encoding and a null-indicator encoding.

Encoding Prefix
===============

Every Nimble encoding begins with a 6-byte prefix
(``encodings/Encoding.h``):

.. list-table::
   :widths: 15 15 35
   :header-rows: 1

   * - Offset
     - Size
     - Field
   * - 0
     - 1 byte
     - ``EncodingType`` (Trivial, RLE, Dictionary, etc.)
   * - 1
     - 1 byte
     - ``DataType`` (Int8, Int32, Float, String, etc.)
   * - 2
     - 4 bytes
     - Row count (number of values in this encoding)

This uniform prefix allows the decoder to quickly determine how to
interpret the remaining bytes.

Encoding Selection
==================

Nimble uses a cost-based encoding selection framework. The default
``ManualEncodingSelectionPolicy`` (``encodings/EncodingSelectionPolicy.h``)
works as follows:

1. Compute ``Statistics<T>`` for the data: run-length characteristics
   (min/max repeat counts), value range (min/max), cardinality (unique
   counts), and string-specific metrics.
2. Iterate candidate encodings and estimate compressed size via
   ``EncodingSizeEstimation``.
3. Apply read-factor weights to balance write size against read cost
   (e.g. ``Trivial`` gets a 0.7 boost since it is cheapest to decode).
4. Pick the encoding with minimum weighted cost.
5. Recursively encode nested sub-streams (e.g. dictionary indices).

Post-encoding, a ``CompressionPolicy`` decides whether to apply Zstd
compression (accepted if the compression ratio is below 0.98).

An ``EncodingLayoutTree`` (``velox/EncodingLayoutTree.h``) can capture and
replay encoding decisions, bypassing runtime selection for deterministic
output.

Data Types
==========

Nimble supports the following data types (``common/Types.h``):

.. list-table::
   :widths: 20 35
   :header-rows: 1

   * - DataType
     - Description
   * - ``Int8`` / ``Uint8``
     - 8-bit signed / unsigned integers
   * - ``Int16`` / ``Uint16``
     - 16-bit signed / unsigned integers
   * - ``Int32`` / ``Uint32``
     - 32-bit signed / unsigned integers
   * - ``Int64`` / ``Uint64``
     - 64-bit signed / unsigned integers
   * - ``Float``
     - 32-bit floating-point
   * - ``Double``
     - 64-bit floating-point
   * - ``Bool``
     - Boolean values (bitpacked)
   * - ``String``
     - Variable-length byte arrays

Compression
===========

Nimble supports chunk-level compression:

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - CompressionType
     - Description
   * - ``Uncompressed``
     - No compression applied.
   * - ``Zstd``
     - Zstandard compression. Level is configurable (default 1).
   * - ``MetaInternal``
     - Internal Meta compression codec.

Reader Architecture
===================

Nimble provides two reader implementations.

**Batch reader** (``velox/VeloxReader.h``): A standalone reader that reads
Nimble files into ``velox::VectorPtr`` batches. It owns a ``TabletReader``
for file I/O and a tree of ``FieldReader`` objects for per-column decoding.
Each ``FieldReader`` wraps a ``Decoder`` that interprets the Nimble encoding.
Supports column projection and FlatMap feature selection.

**Selective reader** (``velox/selective/SelectiveNimbleReader.h``): Integrates
with the DWIO ``SelectiveColumnReader`` framework for filter pushdown.
``SelectiveNimbleReaderFactory`` registers as the DWIO reader factory for
``FileFormat::NIMBLE``. Key classes:

.. list-table::
   :widths: 25 45
   :header-rows: 1

   * - Class
     - Role
   * - ``SelectiveNimbleReader``
     - Implements ``dwio::common::Reader``. Creates row readers with
       pushdown support.
   * - ``ColumnReader``
     - Builds ``SelectiveColumnReader`` trees for pushdown evaluation via
       ``buildColumnReader()``.
   * - ``ChunkedDecoder``
     - Handles multi-chunk decoding within a single stripe, enabling
       efficient ``readWithVisitor`` pushdown.

Read Path
---------

1. ``SelectiveNimbleReaderFactory::createReader()`` initializes a
   ``TabletReader`` (parses postscript and footer).
2. ``createRowReader()`` builds a tree of ``SelectiveColumnReader`` nodes.
3. On each ``next()`` call:

   a. Load the current stripe's streams from ``TabletReader::load()``.
   b. Decode using ``ChunkedDecoder`` which wraps Nimble ``Encoding`` objects.
   c. Apply pushdown filters during decoding via ``readWithVisitor``.

Writer Architecture
===================

``VeloxWriter`` (``velox/VeloxWriter.h``) is the top-level writer:

.. code-block:: cpp

   VeloxWriter writer(type, std::move(file), pool, options);
   writer.write(vector);   // append a batch
   writer.flush();         // flush a stripe
   writer.close();         // finalize the file

Key classes:

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - Class
     - Role
   * - ``VeloxWriter``
     - Top-level writer. Owns a ``FieldWriter`` tree, a ``TabletWriter``,
       and optionally an ``IndexWriter``.
   * - ``FieldWriter``
     - Abstract per-field writer. Decomposes Velox vectors into typed
       ``StreamData`` buffers. Supports row, array, map, flat map, dictionary
       array, sliding window map, and array-with-offsets types.
   * - ``TabletWriter``
     - Low-level file writer. Writes stripes, stripe groups, optional
       sections, footer, and postscript. Supports stream deduplication and
       checksum computation (XXH3_64).
   * - ``FlushPolicy``
     - Controls when to flush stripes. Default: ``StripeRawSizeFlushPolicy``
       at 256 MB raw size threshold.

Write Path
----------

1. ``VeloxWriter::write(VectorPtr)`` decomposes the vector via the
   ``FieldWriter`` tree into typed ``StreamData`` buffers.
2. ``FlushPolicy`` checks if a stripe should be flushed.
3. On flush, each stream's data is encoded via ``EncodingFactory::encode()``
   using the configured ``EncodingSelectionPolicy``.
4. Encoded streams are written to the file via ``TabletWriter::writeStripe()``.
5. On ``close()``, schema, metadata, column statistics, and optional sections
   are written, followed by the footer and postscript.

Statistics and Index Filtering
==============================

**Column statistics** (``velox/stats/ColumnStatistics.h``): The writer
collects per-column statistics during encoding, including value count, null
count, logical size, and physical size. Subclasses provide type-specific
statistics (min/max for integrals and floats, total length for strings).
Statistics are serialized via ``Stats.fbs`` and stored as an optional section
in the file.

**Cluster index filtering** (``index/IndexFilter.h``): When a file is written
with indexing enabled, ``IndexWriter`` records key boundaries per chunk and
stripe for designated index columns. At read time,
``convertFilterToIndexBounds()`` converts ``ScanSpec`` filters into
``IndexBounds`` for stripe and chunk pruning. This supports prefix-contiguous
filters on sorted columns, analogous to a B-tree key prefix scan.

Configuration
=============

Key writer options (``velox/VeloxWriterOptions.h``):

.. list-table::
   :widths: 30 15 30
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``flushPolicyFactory``
     - 256 MB
     - Stripe flush threshold (raw data size).
   * - ``flatMapColumns``
     - (empty)
     - Columns to encode as FlatMaps.
   * - ``dictionaryArrayColumns``
     - (empty)
     - Columns to encode with dictionary arrays.
   * - ``encodingSelectionPolicyFactory``
     - Manual
     - Factory for encoding selection (cost-based by default).
   * - ``compressionOptions``
     - Zstd
     - Compression codec and settings.
   * - ``enableChunking``
     - false
     - Enable chunk-based writing for memory pressure management.
   * - ``indexConfig``
     - (none)
     - Cluster index configuration for indexed writes.
