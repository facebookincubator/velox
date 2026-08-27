Architecture

Call Flow

File Layout

Phase 1: Write

Phase 2: Chunking

Phase 3: Stripe Flush

Nimble Writer Internals
=======================

Standalone core logic — append-only columnar file writer ( facebook::nimble::Writer )

Architecture Overview
---------------------

Writer

(orchestrator)

├──

FieldWriter tree

(per-column writers → StreamData)

├──

WriterContext

(owns streams\_, MemoryPool, FlushPolicy)

├──

IndexWriters

(ChunkIndex, ClusterIndex)

├──

TabletWriter

(file I/O, stripes, footer)

└──

Disk

(TabletWriter → file\_- > append())

Call Flow write()
-----------------

.. code-block:: text

    rootWriter_ (RowFieldWriter)

    write(VectorPtr)  ← input: RowVector with 1000 rows
    │
    ├─ 1. rootWriter_->write(input, ranges[0..1000])
    │      │
    │      └─ RowFieldWriter::write()
    │           ├─ nullsStream_: Vector<bool> ← append 1000 non-null flags
    │           │
    │           ├─ fields_[0]->write(childAt(0))   ← ScalarFieldWriter<int32_t>
    │           │    └─ data_: Vector<int32_t> ← append 1000 ints
    │           │       e.g. [42, 7, -3, 100, ...]  (now 1000 elements)
    │           │
    │           ├─ fields_[1]->write(childAt(1))   ← ScalarFieldWriter<double>
    │           │    └─ data_: Vector<double> ← append 1000 doubles
    │           │
    │           └─ fields_[2]->write(childAt(2))   ← ScalarFieldWriter<string_view>
    │                └─ data_: Vector<string_view> ← append 1000 string_views
    │                   + StringBuffer grows with raw char data
    │
    ├─ 2. addIndexKey(input)
    │
    └─ 3. evaluateFlushPolicy()
           │
           │  streams = context_->streams()  ← flat list of ALL streams:
           │    stream[0] = row nulls       (Vector<bool>)
           │    stream[1] = col a data      (Vector<int32_t>)
           │    stream[2] = col b data      (Vector<double>)
           │    stream[3] = col c lengths   (Vector<int32_t>)
           │    stream[4] = col c chars     (Vector<char>)
           │
           ├─ IF enableChunking && shouldChunk():
           │    └─ flushChunks() → writeChunks()
           │         └─ for each streamIndex:
           │              encodeStreamChunk(streams[streamIndex])
           │                │
           │                │  while (chunker->next()):
           │                │
           │                │    iteration 1:
           │                │    1. form StreamDataView (zero-copy):
           │                │       ├─ nextChunkSize() → elements that fit
           │                │       ├─ data: string_view → [42, 7, -3, 100]
           │                │       │    e.g. [42, 7, -3, 100, | 101, 88, ...]
           │                │       │         ^^^^chunk 1^^^^    ^^stays^^
           │                │       └─ IF enableChunkStats: computeChunkStats()
           │                │            → {hasNulls=false, min=-3, max=100}
           │                │
           │                │    2. encode StreamDataView → Chunk:
           │                │       chunk.content = encodeChunk(chunkView)
           │                │         1. encodeStreamData() → RLE/Dict/Trivial
           │                │         2. compress (Zstd)
           │                │       chunk.stats = chunkView.stats()
           │                │
           │                │    iteration 2: [next 20MB slice...] → same
           │
           └─ IF shouldFlush():
                └─ writeStripe()
                     ├─ encode remaining (lastChunk=true)
                     │
                     └─ tabletWriter_->writeStripe(encodedStreams_)
                          │
                          │  for each stream:
                          │    writeStreamWithChecksum(stream):
                          │      for each chunk:
                          │        file_->append(chunk.content) ← disk
                          │
                          │    addStreamIndex(stream.chunks)
                          │      chunkRows, chunkOffsets, chunkStats
                          │
                          │  tryWriteStripeGroup():
                          │    metadata large → flush index to disk
                          │    else → no-op

    close()
    │
    ├─ writeStripe()                         ← flush remaining data
    ├─ tryWriteStripeGroup(force=true)       ← flush remaining index
    ├─ writeRootIndex()                      ← root index → optional section
    ├─ writeMetadata(), writeColumnStats(), writeSchema()
    ├─ tabletWriter_->close()                ← footer + postscript
    └─ file_->close()

Write Loop

Data Pipeline (1000-row batch)

File Layout on disk
-------------------

Key Insight

It ’ s an append-only write. The index can appear in the middle of stripes — stripe groups and index groups are flushed periodically between stripes to bound metadata memory.

Magic

writeStreamWithChecksum()

writeStripeGroup()

writeIndexGroup()

Stripe 2 data

Stripe 3 data

Stripe Group 1

Index Group 1

...

metadata, stats, schema, root index

points to everything above

Postscript

Magic

Nimble File Structure

Phase 1: write() → StreamData Buffers accumulation
--------------------------------------------------

Write

.. code-block:: text

    TableWriter feeds RowVectors
    ─────────────────────────────

    RowVector 1 ── Writer::write()
    RowVector 2 ──     │
    RowVector 3 ──     ├─ RowFieldWriter::write()
       ...      ──     │       │
    RowVector N ──     │       ├─ FieldWriter A ──append── StreamData A (Vector<int64>)
                       │       ├─ FieldWriter B ──append── StreamData B (Vector<int64>)
                       │       └─ FieldWriter C ──append── StreamData C (Vector<string_view>)
                       │
                       │   (original RowVector can be freed now)
                       │
                       └─ evaluateFlushPolicy()

Memory State

Write

Memory State (accumulated)

(no nulls in data → no nulls stream)

Phase 2: Chunking memory management
-----------------------------------

Chunk

.. code-block:: text

    evaluateFlushPolicy()  220MB ≥ 200MB → shouldChunk = YES
    SOFT CHUNKING: pick streams ≥ 20MB (A data_: 120MB, B data_: 90MB)

Column A
~~~~~~~~

.. code-block:: text

    A data_  (120MB ≥ 20MB → chunk)
      StreamChunker → 6 × 20MB views
      Each view → encodeChunk() (RLE/Dict/Trivial + Zstd) → ~5MB each
      Result: 120MB raw → 6 chunks (~30MB encoded)
    ────────────────────────────────────────────
    A nulls_ (~0.1MB < 20MB → untouched)

    re-check: [30 enc] + 90 + 10 raw = 130MB > 100MB → CONTINUE

Column B
~~~~~~~~

.. code-block:: text

    B data_  (90MB ≥ 20MB → chunk)
      StreamChunker → 4 × 20MB views + 10MB leftover (stays raw)
      Result: 80MB → 4 chunks (~20MB encoded), 10MB leftover
    ────────────────────────────────────────────
    B has no nulls → no nulls stream

    re-check: [50 enc] + 0 + 10 + 10 raw = 70MB < 100MB → STOP

Column C
~~~~~~~~

.. code-block:: text

    C buffer_: 8MB  |  C lengths_: 1.5MB  |  C nulls_: ~0.1MB
    All < 20MB → SKIP (pressure already relieved)

Result

KEEP WRITING

Chunk

.. code-block:: text

    more write() → A:80MB, B:60MB, C:17MB raw + 50MB enc = 207MB
    evaluateFlushPolicy()  207MB ≥ 200MB → shouldChunk = YES

    Soft: A(80MB) → 20MB enc  |  B(60MB) → 15MB enc
    re-check: 17 + 85 = 102MB > 100MB → still above!

    HARD CHUNK: all streams, ensureFullChunks=false
      C(17MB) → 1 undersized chunk → ~4MB enc
      re-check: 0 + 89 = 89MB < 100MB → STOP

Result

Chunk

.. code-block:: text

    more write() → A:25MB, B:15MB, C:3MB raw (43MB) + 89MB enc = 132MB
    shouldChunk?  132MB < 200MB → NO
    shouldFlush?  89 + 43/3.7 ≈ 100.6MB ≥ 100MB → YES → writeStripe()!

Phase 3: writeStripe() I/O
--------------------------

Flush

.. code-block:: text

    writeStripe()
    │
    ├─ 1. Encode ALL remaining raw (lastChunk=true, minChunkSize=0)
    │     A: 25MB → [20MB, 5MB] → ~6.3MB encoded
    │     B: 15MB → [15MB]      → ~3.8MB encoded
    │     C: 3MB  → [3MB]       → ~0.8MB encoded
    │
    ├─ 2. Final chunk inventory:
    │     encodedStreams_[A]: c1..c10 + c11,c12 = 12 chunks
    │     encodedStreams_[B]: c1..c7 + c8       = 8 chunks
    │     encodedStreams_[C]: c1 + c2           = 2 chunks
    │
    ├─ 3. Compact, pass to TabletWriter
    │
    ├─ 4. TabletWriter::writeStripe()
    │     for each stream:
    │       writeStreamWithChecksum → chunks to disk
    │       addStreamIndex → chunk metadata
    │     tryWriteStripeGroup()
    │
    └─ 5. Reset everything, start next stripe

On Disk (one stripe)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ┌───────────────────────────────────────┐
    │ A-c1..c12 │ B-c1..c8 │ C-c1 C-c2    │
    │  Stream A │ Stream B │  Stream C     │
    │                                       │
    │◄──────────── ONE STRIPE ────────────►│
    └───────────────────────────────────────┘

Chunk Index (metadata)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Stream A: 12 chunks │ chunkRows, chunkOffsets, min/max stats
    Stream B:  8 chunks │ chunkRows, chunkOffsets, min/max stats
    Stream C:  2 chunks │ chunkRows, chunkOffsets, min/max stats
    → Reader skips chunks via min/max (filter pushdown)

Memory Layout
