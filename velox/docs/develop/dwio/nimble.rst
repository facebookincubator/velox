=============
Nimble Format
=============

Nimble is Meta's next-generation experimental columnar file format designed
for high-performance analytics with GPU acceleration support. The format
is optimized for modern hardware architectures and provides a flexible
encoding system. The implementation lives under
``experimental/wave/dwio/nimble/``.

.. note::

   Nimble is currently experimental and under active development. The API
   and file format may change without notice.

File Layout
===========

A Nimble file organizes data into stripes. Each stripe contains multiple
columns, and each column is stored as a sequence of *chunks*. A chunk
represents a contiguous segment of encoded data:

.. code-block:: text

   ┌─────────────────────────────────┐
   │       Stripe 0                  │
   │  ┌───────────────────────────┐  │
   │  │  Column 0                 │  │
   │  │  ┌─────────────────────┐  │  │
   │  │  │ Chunk 0             │  │  │
   │  │  │ Chunk 1             │  │  │
   │  │  │ ...                 │  │  │
   │  │  └─────────────────────┘  │  │
   │  │  Column 1                 │  │
   │  │  ...                      │  │
   │  └───────────────────────────┘  │
   ├─────────────────────────────────┤
   │       Stripe 1                  │
   │       ...                       │
   ├─────────────────────────────────┤
   │       File Footer               │  Schema, stripe metadata
   └─────────────────────────────────┘

Encoding Prefix
===============

Every Nimble encoding begins with a 6-byte prefix
(``dwio/nimble/Encoding.h``):

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

Encoding Types
==============

Nimble supports a variety of encodings (``dwio/nimble/Types.h``):

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

Encodings can be nested. For example, a ``Nullable`` encoding wraps a
non-null encoding and a null-indicator encoding.

Data Types
==========

Nimble supports the following data types (``dwio/nimble/Types.h``):

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

GPU Decoding Support
====================

Nimble is designed with GPU acceleration in mind. The Wave DWIO framework
provides GPU-based decoding for Nimble files:

* ``NimbleFormatData`` (``dwio/nimble/NimbleFormatData.h``) implements the
  ``FormatData`` interface and manages GPU decode pipelines.
* ``NimbleChunkDecodePipeline`` orchestrates the decoding of nested
  encodings level by level.
* Each encoding type provides a ``makeStep()`` method that creates a
  ``GpuDecode`` step for GPU-based decoding.

Read Path
=========

1. **Stripe loading.** The reader loads stripe metadata and identifies the
   columns and chunks to read.
2. **Chunk parsing.** ``NimbleChunk`` parses the encoding prefix and creates
   the appropriate ``NimbleEncoding`` subclass (``NimbleTrivialEncoding``,
   ``NimbleRLEEncoding``, ``NimbleDictionaryEncoding``, etc.).
3. **Pipeline construction.** ``NimbleChunkDecodePipeline`` builds a
   decode pipeline by traversing nested encodings.
4. **Decoding.** Each encoding step reads from device memory and writes
   decoded values to the output buffer.

Configuration
=============

Key ``OptimalSearchParams`` entries control encoding selection:

.. list-table::
   :widths: 30 15 35
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``allowedRecursions``
     - 1
     - Maximum recursion depth for nested encodings.
   * - ``enableEntropyEncodings``
     - true
     - Enable entropy-based encodings (slower but more compact).
   * - ``requireDictionaryEnabled``
     - false
     - Force dictionary-enabled encodings for frequently grouped columns.
