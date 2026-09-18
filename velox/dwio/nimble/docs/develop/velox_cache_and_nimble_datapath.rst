


Velox Cache & Nimble Data Path
==============================

AsyncDataCache, CachedBufferedInput, Cache Entry, Format-Specific Usage

1 Overview
----------

.. code-block:: text

    BufferedInput (concrete base class, also used directly)
      │   enqueue/load with coalesced preadv, no cache      (Nimble, when no cache)
      │
      ├── DirectBufferedInput   quantum-based loading, no cache     (DWRF, when no cache)
      └── CachedBufferedInput   cache-backed (RAM + SSD)            (all formats, when cache exists)

    Read path without cache:
      Reader → BufferedInput → enqueue/load → ReadFile → preadv() → storage
               (coalescing only)

    Read path with cache:
      Reader → CachedBufferedInput.enqueue() → CachedBufferedInput.load()
                                                  │
                                                  ├─ 1. AsyncDataCache.findOrCreate() → RAM hit? done
                                                  ├─ 2. SsdFile.find() → SSD hit? load SSD → RAM
                                                  └─ 3. ReadFile.preadv() → storage → store in RAM
                                                  (all three steps inside CachedBufferedInput)

Two-Tier Layout
^^^^^^^^^^^^^^^

(RAM)

key = {fileId, offset}

process-wide, sharded one AsyncDataCacheEntry per region

▼ eviction (async, bytes as-is)

(local SSD)

key = {fileId, offset}

SsdFile per shard SsdRun{offset, size, checksum}

2 CachedBufferedInput
---------------------

API (inherited from BufferedInput base class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Method
     - Returns
     - Description
   * - enqueue(region, sid)
     - unique_ptr < SeekableInputStream >
     - Register a byte range to read. Returns a stream handle (empty until load() )
   * - load(logType)
     - void
     - Execute all enqueued reads — coalesce, check cache, read storage for misses
   * - clone()
     - unique_ptr < BufferedInput >
     - New instance sharing same file + pool, empty enqueue list. Used for lazy column I/O
   * - preload()
     - void
     - Enqueue + load the entire file. For small files, eliminates separate reads
   * - shouldPrefetchStripes()
     - bool
     - Whether caller should prefetch stripe metadata. CachedBufferedInput returns true
   * - read(offset, len)
     - unique_ptr < SeekableInputStream >
     - Unplanned read — bypasses enqueue/load, reads directly from file

Three-Phase Pattern
^^^^^^^^^^^^^^^^^^^

DWRF

.. code-block:: text

    Plan:
    DwrfRowReader::loadCurrentStripe()
      DwrfUnit::load()
        ensureDecoders()
          ColumnReaderFactory::build()
            each column reader calls:
              StripeStreamsImpl::getStream()
                stripeInput->enqueue(region)

    Execute:
        loadDecoders()
          stripeStreams_->loadReadPlan()
            stripeInput->load(STREAM_BUNDLE)

    Consume:
      IntegerRleDecoder → s1->Next() → cache
      StringDecoder → s2->Next() → cache

Nimble

.. code-block:: text

    Plan:
    SelectiveNimbleRowReader::loadCurrentStripe()
      streams_.setStripe(currentStripe_)
      buildColumnReader(...)
        IntegerColumnReader / StringColumnReader / ...
          NimbleData constructor
            NimbleData::makeDecoder()
              streams_->enqueue(streamId)
                BufferedInput::enqueue(region)

    Execute:
      streams_.load()
        BufferedInput::load(STREAM_BUNDLE)

    Consume:
      ChunkedDecoder::ensureInput()
        input_->Next() → zero-copy ptr into cache

Inside load()
^^^^^^^^^^^^^

.. code-block:: text

    input.load()

      Stream count is fixed at enqueue time (1 enqueue = 1 SeekableInputStream).
      load() only decides where each stream's DATA comes from.

      Step 1: Split large requests (loadQuantum = 8MB default)
      Requests > loadQuantum are split into multiple CacheRequests.
      Small requests pass through as-is. Stream count unchanged.

      Step 2: Classify each CacheRequest (decides data source, not stream count)
        ├─ cache_->exists(key)       → RAM hit? skip (data already there)
        ├─ ssdFile->find(key)        → SSD hit? add to ssdLoad group
        └─ neither?                  → add to storageLoad group

      Step 3: Coalesce + Execute
      Nearby storage misses merged into fewer preadv calls.
      SSD hits loaded via SsdFile::load().
      Gap bytes read but discarded (no request owns them).

Examples
^^^^^^^^

Small streams with gap

enqueue

{0, 50KB}

{80K, 30KB}

{90K, 20KB}

Step 1: Split

all < 8MB → no split, 3 CacheRequests as-is

Step 2: Classify

s1: miss

s2: miss

s3: miss

Step 3: Coalesce + Execute

→ 1 preadv(0..110KB)

50KB

gap

30KB

20KB

▼

discarded

▼

Warm Storage

50KB

gap

30KB

20KB

preadv ▼

AsyncDataCache

Entry_1 {fid,0} 50KB

Entry_2 {fid,80K} 30KB

Entry_3 {fid,90K} 20KB

Large stream (loadQuantum)

enqueue

{0, 20MB}

Step 1: Split

20MB > 8MB → 3 CacheRequests:

{fid,0} 8MB

{fid,8M} 8MB

{fid,16M} 4MB

Step 2: Classify

r1: miss

r2: miss

r3: miss

Step 3: Coalesce + Execute

contiguous → 1 preadv(0..20MB)

8MB

4MB

▼

Warm Storage

8MB

4MB

preadv ▼

AsyncDataCache

Entry_1 {fid,0} 8MB

Entry_2 {fid,8M} 8MB

Entry_3 {fid,16M} 4MB

1 stream swaps pin\_ at boundaries

Mixed (RAM + SSD + storage)

enqueue

{0, 1MB}

{1MB, 1MB}

{3MB, 1MB}

Step 1: Split

all < 8MB → no split, 3 CacheRequests as-is

Step 2: Classify

s1: storage miss

s2: RAM hit → skip

s3: SSD hit → ssdLoad

Step 3: Execute (no coalescing — each path independent)

Warm Storage preadv {fid,0} 1MB

SsdCache SsdFile::load() {fid,3M}

▼

AsyncDataCache

Entry_1 {fid,0} preadv

Entry_2 {fid,1M} was in RAM

Entry_3 {fid,3M} SSD→RAM

all three Next() look identical

How CacheInputStream Manages Multiple Entries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CacheInputStream holds one pin at a time — no array of entries. It computes quantum boundaries on the fly:

.. code-block:: text

    CacheInputStream (one per enqueue)
      region_:      {offset: 0, length: 20MB}     ← original enqueued range
      loadQuantum_: 8MB                           ← max entry size
      position_:    current read offset           ← advances via Next()
      pin_:         CachePin                      ← ONE entry at a time

      Next() at position 7MB → pin_ points to Entry_1 {fid,0}
      Next() at position 8MB → boundary!
        1. drop pin_ (unpin Entry_1)
        2. quantum_start = floor(8MB / 8MB) * 8MB = 8MB
        3. cache_->findOrCreate({fid, 8MB}, 8MB)
        4. pin_ = Entry_2
        5. Next() returns ptr into Entry_2

Overlapping Streams
^^^^^^^^^^^^^^^^^^^

Two streams can request cache entries at the same key. findOrCreate() handles it:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Existing entry size
     - Requested size
     - Behavior
   * - None
     - any
     - Create new entry
   * - > = requested
     - any
     - Cache hit. Larger entry satisfies smaller request
   * - < requested
     - any
     - Evict stale entry, create new one at requested size

3 Cache Entry
-------------

AsyncDataCacheEntry is the unit of storage. One entry = one contiguous byte range from one file.

Key Fields
^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - key\_
     - FileCacheKey
     - {fileNum, offset} — identifies the source byte range
   * - size\_
     - int32_t
     - Logical data size (bytes actually meaningful)
   * - numPins\_
     - atomic < int32_t >
     - Reference count. kExclusive (-10000) = being written. > 0 = shared readers
   * - accessStats\_
     - AccessStats
     - LRU-like scoring for eviction priority
   * - ssdSaveable\_
     - bool
     - Whether this entry should be written to SSD on eviction

Three Storage Modes
^^^^^^^^^^^^^^^^^^^

How the actual bytes are held in memory depends on the entry size and allocation mode:

size < 2KB

tinyData\_

std::string

.. code-block:: text

    "..bytes.."

contiguous, inline

default (contiguous=false)

nonContiguousData\_

memory::Allocation

.. code-block:: text

    PageRun[0]: ptr→[██4KB██]
    PageRun[1]: ptr→[████16KB████]
    PageRun[2]: ptr→[██8KB██]

scattered in memory, 4KB pages

contiguous=true

contiguousData\_

void* (malloc)

.. code-block:: text

    [═══ flat buffer ═══]

contiguous, single alloc

PageRun

uint64_t

ceil(50KB / 4KB) = 13 pages = 52KB

Memory Layout
^^^^^^^^^^^^^

AsyncDataCache is sharded by CPU core. Inside one shard:

.. code-block:: text

    AsyncDataCache = CacheShard[0] | CacheShard[1] | ... | CacheShard[N]   (N = cores)
    Inside one shard — entries_ hash set holds all entries:

TINY

tinyData\_:

"..1200 bytes.."

std::string (inline)

DEFAULT

Allocation:

PageRun[0..2047] = 2048 pages

non-contiguous, 4KB-aligned

DEFAULT

.. code-block:: text

    key: {file:42, off:200KB}    size: 50000    pins: 1

    Allocation
      PageRun[0]            PageRun[1]           PageRun[2]
      [4K][4K][4K][4K]      [4K][4K][4K]         [4K][4K]
      0x7fa00000            0x7fb80000           0x7fc40000
      ----16KB----          ---12KB---           --8KB--
      contiguous            contiguous           contiguous
      within run            within run           within run

      <-------- runs are scattered across memory -------->

PageRun = uint64_t: [numPages : 16 bits | ptr : 48 bits] ceil(50000 / 4096) = 13 pages = 52KB allocated (2KB waste)

CONTIG

.. code-block:: text

    key: {file:42, off:80KB}    size: 72584    pins: 2

    contiguousData_:
    +================================================================+
    |                       72584 bytes                              |
    |  single flat malloc -- one pointer, random access anywhere     |
    +================================================================+

    FlatBuffers reads: flatbuffers::GetRoot<Footer>(contiguousData_)

Used by CachedMetadataInput (contiguous=true)

TINY size < 2KB

DEFAULT 4KB page runs

CONTIG flat malloc

Pin Lifecycle
^^^^^^^^^^^^^

.. code-block:: text

    exclusive  ──setExclusiveToShared()──▶  shared  ──last unpin──▶  evictable
    (being written)                           (readable)              (can be freed)
    numPins=-10000                            numPins > 0             numPins = 0
                                                                           │
                                                                ┌──────────┴──────────┐
                                                                ▼                     ▼
                                                         ssdSaveable?          freed immediately
                                                           write to SSD
                                                           then free RAM

4 Format-Specific Usage
-----------------------

How Formats Use the Cache
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1

   * -
     - Data Streams
     - Metadata
   * - DWRF
     - CachedBufferedInput All I/O (data + metadata) through one path. Caches raw/compressed bytes.
     - Writer embeds StripeMetadataCache in file tail. Reader parses once at init → in-memory map. No re-reads.
   * - Nimble
     - CachedBufferedInput StripeStreams::enqueue() → ChunkedDecoder consumes via Next() incrementally. Zero-copy from cache.
     - CachedMetadataInput Separate path. Caches decompressed bytes ( contiguous=true ) for FlatBuffers random access.
   * - Parquet
     - CachedBufferedInput Same as DWRF.
     - No format-specific caching. Speculative tail read only.

Nimble: Stream Ownership Chain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each column's data within a stripe is one or more Nimble streams. Each stream gets its own SeekableInputStream and ChunkedDecoder :

.. code-block:: text

    Column A (integer)
      └─ NimbleData
           ├─ nulls   → ChunkedDecoder_1 → SeekableInputStream_1 → CacheEntry {fid, off_1}
           └─ values  → ChunkedDecoder_2 → SeekableInputStream_2 → CacheEntry {fid, off_2}

    Column B (string)
      └─ NimbleData
           ├─ nulls   → ChunkedDecoder_3 → SeekableInputStream_3 → CacheEntry {fid, off_3}
           ├─ lengths → ChunkedDecoder_4 → SeekableInputStream_4 → CacheEntry {fid, off_4}
           └─ values  → ChunkedDecoder_5 → SeekableInputStream_5 → CacheEntry {fid, off_5}

    1 Nimble stream = 1 ChunkedDecoder = 1 SeekableInputStream = 1 CacheEntry

Nimble: Two I/O Owners, One Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TabletReader and ReaderBase each own a separate I/O abstraction. They share the same AsyncDataCache but never touch each other's entries (different file offsets).

.. code-block:: text

    SelectiveNimbleReader
      │
      ├── ReaderBase
      │     └── input_: CachedBufferedInput          ← owns data I/O
      │           │  receives from connector
      │           │  enqueue/load/Next pattern
      │           │  caches compressed bytes
      │           │  non-contiguous (PageRuns)
      │           │  returns SeekableInputStream
      │           │
      │           ▼
      │     StripeStreams → ChunkedDecoder → column data
      │
      └── TabletReader
            └── metadataInput_: CachedMetadataInput   ← owns metadata I/O
                  │  built internally from ReadFile + cache + fileHandle
                  │  does NOT use CachedBufferedInput
                  │  caches decompressed bytes
                  │  contiguous (flat malloc)
                  │  returns MetadataBuffer
                  │
                  ▼
            footer, stripes, stripe groups, indexes

    Both write to the same AsyncDataCache + SsdCache
      key = {fileId, offset}
      data streams at stripe offsets (start of file)
      metadata at section offsets (end of file)
      → different keys, no collision

/visualize

💬 Feedback (Pixelcloud only)
