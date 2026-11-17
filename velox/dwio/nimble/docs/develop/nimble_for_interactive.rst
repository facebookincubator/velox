Nimble for Interactive
======================

Background • Architecture • Filter Pushdown

Row Store vs Columnar Store
===========================

users (id INT, name STRING, age INT)

Row Store

CSV, JSON, TEXT

1

alice

25

2

bob

30

3

charlie

28

4

dave

35

.. list-table::
   :widths: auto
   :header-rows: 1

   * - id
     - name
     - age
   * - 1
     - "alice"
     - 25
   * - 2
     - "bob"
     - 30
   * - 3
     - "charlie"
     - 28
   * - 4
     - "dave"
     - 35
   * - more
     - rows
     - omitted

Columnar Store

DWRF(ORC), Parquet, Nimble

1

2

3

4

alice

bob

charlie

dave

25

30

28

35

Why Columnar Store?
===================

SELECT

FROM

Row Store

CSV, JSON, TEXT

1

alice

25

2

bob

30

3

charlie

28

4

dave

35

every row

Columnar Store

DWRF(ORC), Parquet, Nimble

1

2

3

4

alice

bob

charlie

dave

25

30

28

35

only the age column

* Column Pruning — read only needed columns, skip the rest
* Better Compression — same-type values compress more efficiently
* Less I/O — dramatically less data read from disk

Columnar Store — Scaling with Stripes
=====================================

users (id INT, name STRING, age INT)

.. list-table::
   :widths: auto
   :header-rows: 1

   * - id
     - name
     - age
   * - 1
     - "alice"
     - 25
   * - more
     - rows
     - omitted
   * - 10000
     - "kate"
     - 31
   * - 10001
     - "leo"
     - 28
   * - more
     - rows
     - omitted
   * - 20000
     - "zoe"
     - 44

(rows 1–10,000)

Worker A

(rows 10,001–20,000)

Worker B

Why not store each column as one giant block?

* Parallelism — each stripe is a self-contained unit, assigned to a different worker
* Reliability — process one stripe at a time (~256 MB), never load entire file
* I/O efficiency — per-stripe min/max stats enable predicate pushdown, skip irrelevant stripes entirely

DWRF (ORC) vs Nimble
====================

File Layout • Metadata Path • Encoding Path

Why Nimble?
-----------

Why invent a new format when we already have DWRF? After a decade of evolution in hardware, query engines, and data scale, DWRF's core assumptions have become limiting. Nimble is a ground-up redesign that advances three areas:

* Advanced file layout
* Metadata efficiency
* Encoding flexibility

Area 1: Advanced File Layout
----------------------------

Richer metadata — Better data pruning

* StripeGroup — targeted column reads
* ChunkIndex — sub-stripe predicate pushdown
* ClusterIndex — sort-key pruning

Extensible structure — Optional section design allows for future flexibility and extensibility to enrich the metadata.

Area 2: Metadata Path Efficiency
--------------------------------

DWRF (Protobuf): O(N) — must deserialize the entire blob to access any field. Nimble (FlatBuffers): O(1) — the buffer is the data structure; zero-copy, selective deserialization and random access.

Must parse entire Footer + entire Stripe Footer to read 1 column

O(1) access — only read what you need

DWRF:

Nimble:

Area 3: Data Path Efficiency (Encoding)
---------------------------------------

Nimble's encoding system improves on DWRF across three layers — more algorithms, smarter selection, and finer granularity — resulting in smaller files on disk and less CPU spent on encoding/decoding.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Layer
     - DWRF
     - Nimble
   * - Encoding Algorithms
     - 4 kinds — DIRECT, DICTIONARY, V2 variants Flat structure — sub-encodings hardcoded
     - 12 types — Constant, Trivial, FixedBitWidth, MainlyConst, SparseBool, Dictionary, RLE, Varint, Delta, Prefix, Nullable, Sentinel Cascading structure — each node picks its own best sub-encoding → Compounding savings at each tree level
   * - Encoding Selection
     - Threshold-based — binary cardinality check Only 2 choices: DIRECT or DICTIONARY
     - Cost-based — estimates size for all 12 candidates, picks smallest → Optimal encoding per data shape
   * - Encoding Granularity
     - Per-file — locked after first stripe All subsequent stripes use the same encoding
     - Per-stripe-group — re-evaluated as data changes → Adapts as data patterns shift

Look Ahead — Nimble for Interactive

Where we started

ML workload rollout

Next focus

interactive workloads

Step 1: Stats and Index

Interactive queries are highly selective — they touch a small fraction of data.

Encoded column chunks


Stripe N — Data Streams

Stream offsets/sizes

NEW Per-stride min/max/nullCount

NEW Per-chunk stream positions

NEW Sorted-column key lookup

NEW File-level min/max/size per column

Schema, stripe locations

Footer size, checksum

Stride Stats

UNDER REVIEW

Chunk Index

MERGED

Cluster Index

MERGED

VectorizedFileStats

ROLLED OUT

Levels of Pruning

user_id BETWEEN 1000 AND 2000

NEW

No overlap → ✗ skip entire file

Overlaps → ✓ open file

NEW

File B

Stripe 0

0–499

skip

500–999

skip

1000–1499

read

2 of 3 skipped

Stripe 1

2500–2999

skip

3000–3499

skip

3500–3999

skip

All skipped — never decompressed

Level 3: Residual Predicate (Row-Level) — Zooming into Stride 2

BETWEEN 1000 AND 2000

.. list-table::
   :widths: auto

   * - user_id
     - 1000 ✓
     - 1001 ✓
     - …
     - 1200 ✓
     - 1201 ✗
     - 1202 ✗
     - …
     - 1499 ✗

201 of 500 rows pass → 60% filtered out at row level

Result:

Cross-Column: Skip decisions propagate to all projected columns

.. list-table::
   :widths: auto

   * -
     - S0
     - S1
     - S2
   * - user_id (filter)
     - 0–499
     - 500–999
     - 1000–1499
   * - name (proj)
     - skip
     - inherit
     - inherit
   * - amount (proj)
     - skip
     - inherit
     - inherit

Gold

Grey

Dashed

Multiplier effect:

NEW

When columns skip strides, they need to jump past skipped data. Without ChunkIndex, they must sequentially decompress every chunk in between. With ChunkIndex: O(1) seek.

Without ChunkIndex

Chunk 0 decompress & discard

Chunk 1 decompress & discard

Chunk 2 decompress & discard

Chunk 3 target

Must decompress 3 chunks to reach target

With ChunkIndex

Chunk 0 skipped

Chunk 1 skipped

Chunk 2 skipped

Chunk 3 O(1) seek

lookupChunk(row) → direct file seek, zero decompression

PROTOTYPE

Learning from Impulse

FixedBitWidthEncoding

Wins:

Before: FixedBitWidth (global)

One global min/max → same bit-width for all:

Chunk 0

100 – 105

17b

17 bits

Chunk 1

5K – 5.2K

17b

17 bits

Chunk 2

42 – 42

17b

17 bits

Chunk 3

0 – 100K

17b

17 bits

Chunk 4

0 – 10

17b

17 bits

global range 0–100K forces 17b everywhere

After: ChunkedBitPacking (per-chunk)

Each 1024-row chunk uses its own local range:

Chunk 0

100 – 105

3b

3 bits

Chunk 1

5K – 5.2K

8b

8 bits

Chunk 2

42 – 42

0 bits !

Chunk 3

0 – 100K

17b

17 bits

Chunk 4

0 – 10

4b

4 bits

only chunk 3 pays 17b — rest use 0–8b

PROTOTYPE

Float / double columns.

Wins:

Before: Trivial (raw float)

Chunk 0

prices

64b

64 bits

Chunk 1

whole #s

64b

64 bits

Chunk 2

all 0.0

64b

64 bits

every float = 64 bits, no compression

After: ChunkedALP (per-chunk)

Chunk 0

e=2 (×100)

8b

8 bits

Chunk 1

e=0 (×1)

13b

13 bits

Chunk 2

constant

0 bits !

64-bit floats → 0–13 bit integers, lossless

Step 2: Better Encoding — TPCH Benchmark

CPU time before and after ALP+CBP encodings

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Query
     - Before
     - After (ALP+CBP)
     - Speedup
   * - q1
     - 15.58m
     - 8.13m
     - 1.9x
   * - q5
     - 11.02m
     - 4.09m
     - 2.7x
   * - q6
     - 3.62m
     - 2.24m
     - 1.6x
   * - q7
     - 9.42m
     - 6.49m
     - 1.5x
   * - q8
     - 2.95h
     - 2.75h
     - 1.1x
   * - q9
     - 36.12m
     - 35.20m
     - 1.0x
   * - q10
     - 14.95m
     - 8.78m
     - 1.7x
   * - q11
     - 1.40m
     - 31.45s
     - 2.7x
   * - q12
     - 14.47m
     - 10.54m
     - 1.4x
   * - q13
     - 17.79m
     - 12.32m
     - 1.4x
   * - q14
     - 5.84m
     - 3.12m
     - 1.9x
   * - q15
     - 13.41m
     - 6.42m
     - 2.1x
   * - q16
     - 6.00m
     - 5.69m
     - 1.1x
   * - q18
     - 32.45m
     - 31.28m
     - 1.0x
   * - q19
     - 7.89m
     - 3.66m
     - 2.2x
   * - Median
     -
     -
     - 1.6x

≥1.4x significant speedup

~1.0x minimal change

11 of 15 queries improved by ≥1.4x

Stride stats write: D103095055 • Stride stats read: D103095054 StrideIndex FBS: velox/dwio/nimble/tablet/StrideIndex.fbs • Writer: velox/dwio/nimble/tablet/StrideIndexWriter.cpp Reader: velox/dwio/nimble/selective/NimbleData.cpp • Skip logic: SelectiveNimbleReader.cpp VectorizedFileStats: velox/dwio/nimble/stats/VectorizedStatistics.h
