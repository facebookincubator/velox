===========
DWRF Format
===========

DWRF is Meta's columnar file format derived from Apache ORC. It adds FlatMap
encoding, column-level encryption, and several performance optimizations. The
reader and writer live under ``dwio/dwrf/``.

File Layout
===========

A DWRF file is structured as follows (read from tail to head):

.. code-block:: text

   ┌─────────────────────────────────┐
   │           Stripe 0              │
   │  ┌──────────────────────────┐   │
   │  │  Row Index Streams       │   │  Per-column stride statistics + positions
   │  ├──────────────────────────┤   │
   │  │  Data Streams            │   │  Encoded column data (PRESENT, DATA, etc.)
   │  ├──────────────────────────┤   │
   │  │  Stripe Footer           │   │  Stream directory + encoding info
   │  └──────────────────────────┘   │
   ├─────────────────────────────────┤
   │           Stripe 1              │
   │           ...                   │
   ├─────────────────────────────────┤
   │    (optional) Stripe Cache      │  Inline cache of stripe index/footer
   ├─────────────────────────────────┤
   │       File Footer               │  Schema, stripe offsets, file statistics
   ├─────────────────────────────────┤
   │       PostScript                │  Footer length, compression codec, version
   ├─────────────────────────────────┤
   │    PostScript length (1 byte)   │  Last byte of file
   └─────────────────────────────────┘

Reading starts from the last byte (PostScript length), then reads the
PostScript, then the File Footer, and finally seeks to individual stripes on
demand.

Streams
=======

Each column in a stripe is stored as one or more *streams*. The ``StreamKind``
enum identifies the stream type:

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - StreamKind
     - Purpose
   * - ``PRESENT``
     - Null bitmap. One bit per row (ByteRLE encoded).
   * - ``DATA``
     - Primary data stream (integer values, string bytes, etc.).
   * - ``LENGTH``
     - Lengths for variable-width types (strings, arrays, maps).
   * - ``DICTIONARY_DATA``
     - Dictionary entries for dictionary-encoded string columns.
   * - ``IN_MAP``
     - FlatMap: per-key bitmap indicating whether the key is present in each
       row.
   * - ``STRIDE_DICT``
     - Per-stride dictionary (used when the global dictionary is too large).

A stream is uniquely identified by a ``DwrfStreamIdentifier`` consisting of
``(node, sequence, column, kind)``.

Encoding
========

**RLE v1** (``dwio/dwrf/common/RLEv1.h``): Simple run-length encoding. A
*run* header encodes either a run of repeated values or a sequence of
literals.

**RLE v2** (``dwio/dwrf/common/RLEv2.h``): A more compact encoding with four
sub-encodings:

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - Sub-encoding
     - When used
   * - ``SHORT_REPEAT``
     - Short runs of identical values (3--10 values).
   * - ``DIRECT``
     - Values stored with bit-packing at the minimum required width.
   * - ``PATCHED_BASE``
     - Values close to a base with a few outliers stored as patches.
   * - ``DELTA``
     - Monotonic or near-monotonic sequences stored as a base + deltas.

**ByteRLE** (``dwio/dwrf/common/ByteRLE.h``): Specialized for boolean and
byte streams. Encodes runs and literals at the byte level.

Compression
===========

Each stream is independently compressed using 3-byte page headers. A page
header stores the compressed length and an ``isOriginal`` flag (set when the
compressed output would be larger than the original). Supported codecs
include ZLIB, ZSTD, LZO, LZ4, and Snappy. The codec is recorded in the
PostScript. See ``dwio/dwrf/common/Compression.h``.

Row Group Index (Strides)
=========================

DWRF divides each stripe into *strides* (default 10,000 rows). For each
stride, a ``RowIndexEntry`` records:

* The byte offset into each stream.
* The decompressor position (bytes consumed in the current compression page).
* The RLE run position (values consumed in the current RLE run).

This three-level positioning allows the reader to seek directly to any stride
within a stripe, skipping strides whose statistics prove no rows can match the
filter.

FlatMap Encoding
================

FlatMap is a DWRF-specific encoding for MAP columns where the set of keys is
known at write time.

Instead of storing a single ``key`` stream and a single ``value`` stream, the
writer creates a separate ``value`` stream and ``IN_MAP`` bitmap for each
distinct key. This turns a map into a struct-like layout:

.. code-block:: text

   MAP<K, V>  →  key₁ : (IN_MAP₁, value₁),
                 key₂ : (IN_MAP₂, value₂),
                 ...

Benefits:

* **Key-level projection pushdown.** The reader can skip keys that are not
  referenced by the query.
* **Per-key statistics.** Stride statistics are maintained per key, enabling
  fine-grained stride skipping.
* **Efficient scanning.** The ``IN_MAP`` bitmap is cheaper to decode than a
  variable-length key stream.

Writer: ``dwio/dwrf/writer/FlatMapColumnWriter.h``.
Reader: ``dwio/dwrf/reader/SelectiveFlatMapColumnReader.h``.

Column-Level Encryption
=======================

DWRF supports encrypting individual columns or groups of columns.

``EncryptionHandler`` (``dwio/dwrf/common/Encryption.h``,
``Decryption.h``) maps column node IDs to ``Encrypter`` / ``Decrypter``
objects. Encryption is applied **after** compression so that the compression
ratio is not affected by the randomness of encrypted data.

Encrypted columns store their stripe footer and statistics in separate
encrypted sections that are only accessible with the appropriate key.

Configuration
=============

Key ``Config`` entries (``dwio/dwrf/common/Config.h``):

.. list-table::
   :widths: 30 15 30
   :header-rows: 1

   * - Config Key
     - Default
     - Description
   * - ``STRIPE_SIZE``
     - 64 MB
     - Target uncompressed stripe size.
   * - ``ROW_INDEX_STRIDE``
     - 10,000
     - Number of rows per stride (row group).
   * - ``FLATTEN_MAP``
     - false
     - Enable FlatMap encoding for map columns.
   * - ``MAP_FLAT_COLS``
     - (empty)
     - Column IDs to flatten.
   * - ``DICTIONARY_NUMERIC_KEY_SIZE_THRESHOLD``
     - 0.8
     - Fraction of distinct values above which dictionary encoding is
       abandoned in favor of direct encoding.

Stripe Cache
============

When a DWRF file is written with stripe caching enabled, the index and footer
of each stripe are duplicated at the end of the file (just before the File
Footer). This allows the reader to fetch all stripe metadata in a single read
during file open, avoiding one seek per stripe.

Writer Version Evolution
========================

DWRF writer versions track format changes over time, from ``ORIGINAL``
through ``DWRF_7_0``. The version is stored in the PostScript and controls
which features the reader expects to find in the file.
