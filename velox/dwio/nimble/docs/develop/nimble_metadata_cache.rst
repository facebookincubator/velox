


Nimble Metadata Cache
=====================

File layout, MetadataSection, CachedMetadataInput, and TabletReader init flow

5 Why TabletReader Doesn't Use CachedBufferedInput
--------------------------------------------------

Contiguous vs Non-Contiguous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FlatBuffers (metadata, indexes)

.. code-block:: text

    flatbuffers::GetRoot<Footer>(data)
      data + offset → random jump to any byte
      data + offset + vtable → another jump

    Needs the ENTIRE buffer at one address.
    Non-contiguous PageRuns → wrong memory → crash.

    → must be contiguous (flat malloc)
    → CachedMetadataInput (contiguous=true)

Nimble Encodings (column data)

.. code-block:: text

    ChunkedDecoder::ensureInput(size)
      input_->Next(&buf, &len) → next chunk
      reads forward only, never random jumps

    PageRun boundary? Next() returns rest of
    current run, next call returns next run.
    ensureInput() copies across boundary.

    → non-contiguous PageRuns work fine
    → CachedBufferedInput (contiguous=false)

Three Reasons for the Split
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Number
     - Reason
     - Detail
   * - 1
     - Contiguity
     - FlatBuffers needs contiguous random access. CachedBufferedInput stores as non-contiguous PageRuns — would require an extra readFully() copy to flatten.
   * - 2
     - Decompression
     - CachedBufferedInput caches compressed bytes → decompress on every access. CachedMetadataInput caches decompressed → zero-cost on cache hit.
   * - 3
     - Shared component
     - TabletReader is used by both the selective reader (has CachedBufferedInput ) and batch VeloxReader (has no BufferedInput at all). Can't depend on it.

What Goes Where
^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Content
     - Format
     - Access
     - I/O Path
   * - Footer, stripes metadata
     - FlatBuffers
     - Random
     - TabletReader → CachedMetadataInput
   * - Stripe group metadata
     - FlatBuffers
     - Random
     - TabletReader → CachedMetadataInput (lazy)
   * - Cluster / chunk stats
     - FlatBuffers
     - Random
     - TabletReader → CachedMetadataInput
   * - Column data streams
     - Nimble encodings
     - Sequential
     - ReaderBase → CachedBufferedInput

The boundary isn't metadata vs data

FlatBuffers (contiguous, random access) vs encoded streams (non-contiguous, sequential)

6 Nimble File Layout
--------------------

Nimble

Stripe 0 — Data Streams

Independently encoded per stream (Nimble encodings)

···


Stream 0: [chunk0|chunk1|...] Stream 1: [chunk0|chunk1|...] Stream 2: [chunk0|chunk1|...]

stripe_count:

3

stream_offsets:

[0, 10K, 22K, 0, 11K, 23K, 0, 9K, 20K]

stream_sizes:

[10K, 12K, 8K, 11K, 12K, 9K, 9K, 11K, 8K]

flattened [stripe × stream] = 3 stripes × 3 streams = 9


1

stream_offsets[], stream_sizes[] per (stripe × stream)

4


stream_count:

5

chunk_rows:

[0, 2500, 5000, ...]

chunk_offsets:

[0, 1200, 2400, ...]

per (stripe × stream × chunk)

Stripe N

1

4

3

Pointers to StripeChunkStats 0, 1…


stripe_indexes: [MetadataSection]

[0]

{off:0x16900, sz:1024}


group 0 ptr

4

[1]

{off:0x25900, sz:980}


group 1 ptr

4

stripe_keys:

["a".."m", "m".."z"]

sort_orders:

["ASC NULLS FIRST"]


3

Stripe keys, sort orders, index pointers

row_counts:

[10000, 10000, 8000]

offsets:

[0x0, 0x7800, 0xF800]

sizes:

[30KB, 32KB, 28KB]

group_indices:

[0, 0, 0]


2

row_counts[], offsets[], sizes[], group_indices[]

3

columnar.schema — column type tree


nodes:

[{Row, "root"}, {Int64, "id"}, {String, "name"}, ...]

{"nimble.writer.version": "1.2.3", ...}


3

columnar.metadata — key-value pairs

3

columnar.vectorized_stats — per-column min/max/count

row_count:

57000

each pointer = MetadataSection{offset, size, comp, uncomp_size}

stripe_groups:


[{0x8000, 256, Zstd, 1024}, ...]

1 ptr[]

stripes:


{0xA000, 512, Zstd, 2048}

2 ptr

optional_sections:


[{0xB000, 256, Zstd, 800}, ...]

3 ptr[] (named)


File Footer

Directory of pointers (MetadataSection) to data elsewhere in the file

MetadataSection
^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Form
     - Representation
   * - On disk
     - FlatBuffers table ``{offset, size, comp, uncomp_size}``
   * - In memory
     - C++ class with ``optional<uint32> uncompressedSize``. ``nullopt`` is used for old files where the FlatBuffers default ``0`` maps to no uncompressed size.

Metadata Cache Contents
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Cache key {fileId, fileSize}
      Footer + PostScript:
        row_count: 57000
        stripe_groups: [{0x8000, 256, Zstd, 1024}, ...]  -> MetadataSection refs
        stripes:       {0xA000, 512, Zstd, 2048}         -> MetadataSection ref
        optional_sections:
          [{0xB000, 256, Zstd, 800}, ...]                -> named MetadataSection refs
        serialized postscript

    Cache key {fileId, sg[0].offset}
      StripeGroup[0]:
        stripe_count: 2
        stream_offsets: per-stripe stream starts
        stream_sizes:   per-stripe stream lengths

    Cache key {fileId, sg[1].offset}
      StripeGroup[1]:
        stripe_count: 1
        stream_offsets: per-stripe stream starts
        stream_sizes:   per-stripe stream lengths

    Cache key {fileId, stripes.offset}
      Stripes:
        row_counts:    [1000, 1000, 1000]
        offsets:       [0, 5000, 9800]
        sizes:         [5000, 4800, 5200]
        group_indices: [0, 0, 1]

    Cache key {fileId, cli.offset}
      ClusterIndex root:
        partitions: [MetadataSection{off, sz, comp, uncomp}, ...]

    Cache key {fileId, ci.offset}
      ChunkStats root:
        stripe_indexes: [MetadataSection{off, sz, comp, uncomp}, ...]

    Cache key {fileId, cig[0].offset}
      ChunkStats group[0]:
        Per-chunk stream boundaries within stripe group 0

    Cached values are decompressed contiguous bytes.
    Cached size is uncomp_size, used to allocate and validate the cache entry.
    Old files have uncomp_size=nullopt and skip the individually cached sections.

footerSize:

uint32 (compressed)

compressionType:

uint8

checksumType:

uint8

checksum:

uint64

version:

uint16 + uint16

magic:

uint16


PostScript (20 bytes)

footerSize, compressionType, checksum, version, magic

.. code-block:: text

    NIMBLE FILE — PHYSICAL LAYOUT (read bottom-up, written top-down)

    ┌──────────────────────────────────────────────────────────────────────────┐ Byte 0
    │ STRIPE DATA — GROUP 0                                                    │
    │ ┌──────────────────────────────────────────────────────────────────────┐ │
    │ │ Stripe 0:                                                            │ │
    │ │   Stream 0 [chunk0|chunk1|...]  Stream 1 [chunk0|chunk1|...]  ...    │ │
    │ │ Stripe 1:  ...                                                       │ │
    │ │ Stripe M:  (last stripe in group 0)                                  │ │
    │ └──────────────────────────────────────────────────────────────────────┘ │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ StripeGroup 0 Metadata (FlatBuffers: stream_offsets[], stream_sizes[])  │
    │ Chunk Stats Group 0    (FlatBuffers: chunk_rows[], chunk_offsets[])     │
    │ Cluster Index Group 0  (FlatBuffers, optional: key boundaries)          │
    │ StripeStrideIndex 0    (FlatBuffers, *** NEW ***: min/max per stride)   │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ STRIPE DATA — GROUP 1  ...  StripeGroup 1  ...  Index Groups 1          │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ GLOBAL METADATA                                                        │
    │   Stripes        (row_counts[], offsets[], sizes[], group_indices[])    │
    │   Schema         "columnar.schema"          — type tree                 │
    │   Metadata       "columnar.metadata"        — key-value pairs           │
    │   Stats          "columnar.vectorized_stats"— per-column stats          │
    │   ChunkStats     "columnar.chunk.stats"     — root → per-group blobs   │
    │   FileIndexes    "columnar.indexes"         — named index manifest     │
    │   StrideIndex    "columnar.stride.index"    — root → per-group blobs   │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ FOOTER (FlatBuffers, possibly Zstd compressed)                         │
    │   row_count, stripes ref, stripe_groups[] refs, optional_sections      │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ POSTSCRIPT (20 bytes fixed)                                              │
    │   FooterSize(4B) ComprType(1B) ChecksumType(1B) Checksum(8B)          │
    │   MajorVer(2B) MinorVer(2B) Magic 0xA1FA(2B)                          │
    └──────────────────────────────────────────────────────────────────────────┘ EOF

Per-Group FlatBuffer Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    StripeGroup Metadata (one per group, points back to stripe data)
      stripe_count:    uint32
      stream_offsets:  [uint32]   flattened [stripe x stream]
      stream_sizes:    [uint32]   flattened [stripe x stream]
          ↑ locates individual stream bytes within stripe data

    StripeChunkStats (one per group)
      stream_count:          uint32
      stream_chunk_counts:   [uint32]   prefix-sum [stripe x stream]
      stream_chunk_rows:     [uint32]   prefix-sum row counts per chunk
      stream_chunk_offsets:  [uint32]   byte offset per chunk

    StripeStrideIndex (one per group, *** NEW ***)
      stride_size:           uint32    (e.g. 10,000 rows)
      stride_count:          uint32
      stripe_stride_counts:  [uint32]  (strides per stripe in group)
      stream_stats: [StreamStrideStats] per stream
        null_counts: [int64]   per stride
        min_values:  [ubyte]   concatenated raw bytes
        max_values:  [ubyte]   concatenated raw bytes
        value_size:  uint32    bytes per value (e.g. 4, 8)

Footer = Directory of Pointers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    byte 0                                                          byte end
    ↓                                                               ↓
    ┌─────────────────┬───────────────────────────┬──────────┬──────────┐
    │  Stream data    │  Metadata blobs           │  Footer  │ PS (20B) │
    │  (columns)      │  (scattered in here)      │          │          │
    └─────────────────┴───────────────────────────┴──────────┴──────────┘
                       ↑                           ↑
                       │                           └─ Footer is just pointers:
                       │                              row_count: 57000  ← only inline field
                       │                              stripes:       → {0xA000, 512, Zstd}
                       │                              stripe_groups: → [{0x8000, 256, Zstd},
                       │                                                {0x9000, 248, Zstd}]
                       │                              optional_sections:
                       │                                "columnar.schema"        → {0xB000, ...}
                       │                                "columnar.chunk.stats"   → {0xC000, ...}
                       │                                "columnar.indexes"       → {0xD000, ...}
                       │
                       ├─ 0x8000: StripeGroup 0 blob
                       ├─ 0x9000: StripeGroup 1 blob
                       ├─ 0xA000: Stripes blob
                       ├─ 0xB000: Schema blob
                       ├─ 0xC000: ChunkStats root blob
                       └─ 0xD000: File index manifest blob

    One field (row_count) gives the answer directly.
    Everything else says "go read that blob if you need it."

Footer.stripes — Coarse Index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Footer.stripes {offset=0xA000, size=512, comp=Zstd}  → Stripes blob

      Stripe ID   row_counts   offsets     sizes    group_indices
      stripe 0    10000        0x0000      30KB     0
      stripe 1    10000        0x7800      32KB     0
      stripe 2     8000        0xF800      28KB     0
      stripe 3    10000        0x16C00     31KB     1
      stripe 4    10000        0x1E400     29KB     1

      One row per stripe. Tells about the stripe AS A WHOLE.
      "stripe 3 has 10000 rows, starts at 0x16C00, is 31KB, belongs to group 1"
      But 31KB of WHAT? All streams mashed together. Can't pick a single column.

Footer.stripe_groups[] — Fine Index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Footer.stripe_groups[0] {offset=0x16800, size=256, comp=Zstd}  → StripeGroup 0

      stripe_count = 3
                        stream 0 (col:id)  stream 1 (col:name)  stream 2
      stripe 0 offsets: 0                  10KB                  22KB
      stripe 0 sizes:   10KB              12KB                  8KB
      stripe 1 offsets: 0                  11KB                  23KB
      stripe 1 sizes:   11KB              12KB                  9KB

      One entry per (stripe, stream) pair.
      "within stripe 1, stream 2 starts at offset 23KB and is 9KB long"
      NOW you can read just one column.

    Two-level index:
      Stripes     → coarse: where is each stripe? (row count, total size, group)
      StripeGroup → fine:   where is each stream within the stripe?

Optional Sections — Extensible Design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Footer.optional_sections:
      names:              ["columnar.schema",  "columnar.metadata",  ...]
      offsets:            [0xB000,             0xB100,               ...]
      sizes:              [256,                128,                  ...]
      compression_types:  [Zstd,               Zstd,                ...]

    columnar.schema — column type tree, maps columns to stream numbers:
      nodes: [
        {kind: Row,    children: 3, name: null,   offset: 0},  ← root
        {kind: Int64,  children: 0, name: "id",   offset: 1},  ← stream 1
        {kind: String, children: 0, name: "name", offset: 2},  ← stream 2
        {kind: Map,    children: 2, name: "tags", offset: 3},
        {kind: String, children: 0, name: null,   offset: 4},  ← map key
        {kind: Int32,  children: 0, name: null,   offset: 5},  ← map value
      ]

    columnar.metadata — arbitrary key-value pairs:
      {"nimble.writer.version": "1.2.3", "hive.table.name": "warehouse.clicks"}

    Schema tells what columns exist and which stream carries each.
    Metadata tells who wrote the file and with what settings.
    Together with StripeGroup, you can read any column.

Section Status
^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Section
     - Status
   * - columnar.schema
     - Active
   * - columnar.metadata
     - Active
   * - columnar.stats
     - Legacy (replaced by vectorized_stats)
   * - columnar.vectorized_stats
     - Active
   * - columnar.chunk.stats
     - Active
   * - columnar.indexes
     - Active file index manifest
   * - columnar.stride.index
     - New
   * - columnar.dense.index
     - Defined, not implemented

Write Order Example
^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    [stripe 0..66 data]
                              ← metadata threshold hit, flush group 0
    [StripeGroup 0] [ChunkStats 0] [ClusterIndex 0] [StrideIndex 0]

    [stripe 67..120 data]
                              ← close(), force flush group 1
    [StripeGroup 1] [ChunkStats 1] [ClusterIndex 1] [StrideIndex 1]

    [ChunkStats root]         ← pointers to ChunkStats 0, 1
    [ClusterIndex root]       ← pointers to ClusterIndex 0, 1
    [StrideIndex root]        ← pointers to StrideIndex 0, 1
    [Stripes blob]            ← row counts, offsets for all 121 stripes
    [Footer]                  ← pointers to everything above
    [Postscript 20B]

    Stripes blob   — flushed at close(). One per file. Coarse info.
    StripeGroup    — flushed incrementally at metadata threshold.
                     Writer treats each group as self-contained unit:
                     write, free memory, move on.

7 TabletReader Init Flow
------------------------

Metadata Compression
^^^^^^^^^^^^^^^^^^^^

Each metadata section is compressed independently when it is larger than the default 64KB threshold.

.. code-block:: text

    TabletWriter::writeMetadata(section_data):
      if section_data.size() > 64KB:
        compressed = Zstd(section_data)
        write compressed bytes -> file
        return CompressionType::Zstd
      else:
        write raw bytes -> file
        return CompressionType::Uncompressed

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Section
     - Example Size
     - Compressed?
     - Why
   * - StripeGroup 0
     - 80KB
     - Zstd
     - 80KB > 64KB
   * - StripeGroup 1
     - 40KB
     - No
     - 40KB < 64KB
   * - Stripes
     - 120KB
     - Zstd
     - 120KB > 64KB
   * - Schema
     - 2KB
     - No
     - 2KB < 64KB
   * - ChunkStats Root
     - 90KB
     - Zstd
     - 90KB > 64KB
   * - Footer
     - 50KB
     - No
     - 50KB < 64KB

Reader Entry
^^^^^^^^^^^^

.. code-block:: text

    init()
    -> initFromCache()
       cached() false?                  -> Cold Path  // no CachedMetadataInput
       findCachedMetadata(fileSize_) miss? -> Cold Path  // Footer+PS not in AsyncDataCache
       hit                              -> Warm Path

    "cold" means the footer is loaded from the file. Other sections, such as
    Stripes and StripeGroup, may still hit cache individually through
    loadSections() -> CachedMetadataInput::load().

Cold Path
^^^^^^^^^

.. code-block:: text

    loadFooter(maxFooterIoBytes)                // 1 pread
      one speculative read of last 128KB
      -> buf, extract PostScript, decompress Footer

    initOptionalSections()                      // no IO
      parse optional_sections()
      build optionalSections_: name -> {off, sz, comp}

    collectStripesSection()                     // no IO
      Stripes ptr from Footer
      -> tryExtractFromBuffer(ptr)

    collectOptionalSections()
      each optional section ptr:
        ClusterIdx, ChunkIdx   // stored by cacheMetadata()
        Schema, Stats          // stored on first miss by load()
      -> tryExtractFromBuffer(ptr)

    tryExtractFromBuffer(ptr)
      in buf     -> decompress
      not in buf -> enqueue to loadSections list

    loadSections(list)                          // IO if needed
      metadataInput_->load(sections)
        RAM hit -> done
        SSD hit -> done
        miss    -> collect, coalesce nearby,
                   preadv from file -> decompress -> store in cache

    initStripes(buf)                            // no IO
      wire up stripeOffsets_ and stripeRows_
      if one stripe group and the group is in buf:
        eagerly pin StripeGroup[0] into stripeGroupCache_

    initClusterIndex() / initChunkStats(buf)    // no IO
      parse index roots from cache
      if chunk stats group 0 is in buf:
        pin it

    cacheMetadata(buf)                          // no IO
      decompress -> AsyncDataCache
      populates entries for the warm path

    Small file: 1 pread
    Large file: 1 pread + 1 coalesced preadv

Warm Path
^^^^^^^^^

.. code-block:: text

    loadFooterFromCache()                       // no IO
      findCachedMetadata(fileSize_) -> footer_ + ps_

    initOptionalSections()                      // no IO
      parse cached Footer -> build optionalSections_ map

    collectStripesSection() + collectOptionalSections()
      no footerBuf -> enqueue all sections to loadSections

    loadSections(list)
      CachedMetadataInput::load() per section:
        RAM hit -> done
        SSD hit -> done
        miss    -> preadv
      file fallback auto-populates cache through store()

    initStripes() + initIndexes()               // no IO
      wire up members from loaded FlatBuffers
      no footerBuf -> stripe group is not eagerly pinned

    cacheMetadata() is not called

    All in RAM: 0 IO
    Evicted: SSD
    Full miss: 1 coalesced preadv

Lazy Path
^^^^^^^^^

.. code-block:: text

    stripeIdentifier(N)
      stripeGroupCache_ hit?
        yes -> return            // zero IO, zero cache lookup
        no  -> loadStripeGroupMetadata(N)

    loadStripeGroupMetadata(N)
      CachedMetadataInput::load([sg[N], cig[N]])
        RAM hit   -> done
        SSD hit   -> done
        file miss -> coalesced preadv
                     sg + cig are adjacent and use one read
                     store() -> AsyncDataCache

      result stored in stripeGroupCache_ as a weak ptr

8 Cache Write / Read Path Asymmetry
-----------------------------------

The cache write path ( cacheMetadata ) and read path ( loadFromCache ) use MetadataSection fields differently. Understanding this asymmetry is key to avoiding bugs like the compressed-bytes caching issue (D108389825).

cacheSection()

offset()

Locate section in speculative buffer

size()

Compressed on-disk size — bytes to extract

compressionType()

Determines whether to decompress

uncompressedSize()

Not used — decompressed size comes from MetadataBuffer::decompress() output

section.size()

section.compressionType()

section.offset()

// memcpy into cache entry

loadFromCache()

offset()

Cache key lookup: {fileId, offset}

size()

Not used for cache lookup

compressionType()

Not used — cached data is already decompressed

uncompressedSize()

Required — pre-allocates cache pin at correct decompressed size

uncompressedSize

// if hit → zero-copy MetadataBuffer from cache pin

// if miss → file IO, decompress, promoteCachePin

Key Invariant

* The write path must store decompressed data.
* The read path expects entry- > size() == uncompressedSize .
* If the write path stores compressed bytes (size S c ) but the read path requests uncompressedSize (S u > S c ): RAM : findOrCreate evicts the stale entry — cache miss on every warm open. SSD : crash via NIMBLE_CHECK_EQ in loadFromSsd() .

/visualize

💬 Feedback (Pixelcloud only)
