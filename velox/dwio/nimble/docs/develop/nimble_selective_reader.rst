



Nimble Selective Reader
=======================

A concrete example — 5 columns in table, 3 projected

.. code-block:: text

      SELECT col4
      FROM t
      WHERE col1 > 50
      AND col3 < 3.14

        Table schema: col1 int32, col2 int64, col3 double, col4 string, col5 float
        — col2 and col5 not read


Selective & Batch Reader

Two read paths from the same Velox operator — both end at TabletReader, but the middle layers differ

Selective Path

selective_nimble_reader_enabled = true

(default)

.. code-block:: text

    TableScan
    ├── HiveDataSource
        ├── FileSplitReader
            ├── SelectiveNimbleReader (dwio::common::Reader)
            │
            │   createReader(): uses VeloxReader temporarily to extract schema/type
            │   createRowReader(): builds SelectiveNimbleRowReader
            │
            └── SelectiveNimbleRowReader (dwio::common::RowReader)
                  │   next() → own column reader tree
                  └── TabletReader (direct, no VeloxReader in the loop)

Batch Path

selective_nimble_reader_enabled = false

.. code-block:: text

    TableScan
    ├── HiveDataSource
        ├── FileSplitReader
            ├── NimbleReader (dwio::common::Reader)
            │
            │   createRowReader(): builds NimbleRowReader
            │
            └── NimbleRowReader (dwio::common::RowReader)
                  ├── VeloxReader.next()
                  │     ├── FieldReader tree
                  │     └── TabletReader

ⓘ

NimbleReaderFactory::createReader()

selectiveNimbleReaderEnabled()

ReaderOptions

Part A

Orchestration

Velox → Nimble reader → stripe → batch → struct reader → column readers

0

SelectiveNimbleReader → SelectiveNimbleRowReader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SelectiveNimbleReader (file-level) creates a SelectiveNimbleRowReader which drives the entire read loop, iterating through stripes internally.

VELOX

TableScan

getOutput()

HiveDataSource

getOutput()

FileSplitReader

createReader()

SelectiveNimbleReader

createRowReader()

SelectiveNimbleRowReader

next()

rowReader- > next()

SelectiveNimbleReader()
"""""""""""""""""""""""

ReaderBase::create() → read footer, cluster index, footer column stats

createRowReader()
"""""""""""""""""

→ new SelectiveNimbleRowReader

initReadRange(): split byte range → which stripes to read [startStripe\_, endStripe\_)

per stripe

nextRowNumber()
"""""""""""""""

loadCurrentStripe()
"""""""""""""""""""

setStripe():

computeSkipRowRanges():

buildColumnReader():

streams\_.load():

per batch

nextReadSize(batchSize)
"""""""""""""""""""""""

STEP 1: APPLY ROW SKIPS → SelectiveStructColumnReader- > seekTo(targetRow)

col1(IntegerColumnReader).seekTo(targetRow)

decoder\_.skip(targetRow - currentRow)

col3(FloatingPointColumnReader).seekTo(targetRow)

decoder\_.skip(targetRow - currentRow)

col4(StringColumnReader).seekTo(targetRow)

decoder\_.skip(targetRow - currentRow)

STEP 2: COMPUTE READ SIZE

return min(batch, stripeEnd, nextSkipStart)

SelectiveStructColumnReader.next(size, result)
""""""""""""""""""""""""""""""""""""""""""""""

for each child in selectivity order: child- > read(offset, rows) → child- > getValues() → pass survivors to next

PHASE 1: READ (sequential, narrowing rows)

col1(IntegerColumnReader).read(0, [0..10K]) 10K → 1K
""""""""""""""""""""""""""""""""""""""""""""""""""""

visitor captures:

BigintRange

decoder\_.readWithVisitor(visitor < BigintRange > )

visitor populates:

col3(FloatingPointColumnReader).read(0, [3,17,42,...]) 1K → 200
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

visitor captures:

DoubleRange

decoder\_.readWithVisitor(visitor < DoubleRange > )

visitor populates:

col4(StringColumnReader).read(0, [17,42,...]) 200 → 200
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

visitor captures:

AlwaysTrue

decoder\_.readWithVisitor(visitor < AlwaysTrue > )

visitor populates:

PHASE 2: GET VALUES (final 200 survivors passed to all)

col1.getValues([17,42,...])

compactScalarValues(): scan 1K rawValues\_, copy 200 matching

col3.getValues([17,42,...])

compactScalarValues(): scan 200 rawValues\_, copy 200 matching

col4.getValues([17,42,...])

compactScalarValues(): scan 200 rawValues\_, copy 200 matching

assemble RowVector
""""""""""""""""""

advanceToNextStripe() ↑

FILE LAYOUT

Stripe 0

Stripe 1

col1

col2

col3

col4

col5

StripeGroup 0

Stripe 2

Stripe 3

StripeGroup 1

Global Metadata

Footer

Postscript + Magic

Read from end: Postscript → Footer → Global Metadata → Stripes

BUILD COLUMN READER

col2, col5 skipped

SelectiveNimbleRowReader
""""""""""""""""""""""""

SelectiveStructColumnReader
"""""""""""""""""""""""""""

top-level ROW type — orchestrates filter order

vector < unique_ptr < SelectiveColumnReader > > children\_ (3 of 5 columns)

IntegerColumnReader
"""""""""""""""""""

col1 int32

scanSpec\_.filter\_: col1 > 50

(value)

Integer ChunkedDecoder
""""""""""""""""""""""

encoding\_ → FixedBitWidth unique_ptr < SeekableInputStream > input\_ shared_ptr < StreamIndex > streamIndex\_ rowPosition\_ (uint32) remainingValues\_ (int64) onChunkLoad\_ (callback)

FloatingPointColumnReader
"""""""""""""""""""""""""

col3 double

scanSpec\_.filter\_: col3 < 3.14

(value)

Floating Point ChunkedDecoder
"""""""""""""""""""""""""""""

encoding\_ → Trivial unique_ptr < SeekableInputStream > input\_ shared_ptr < StreamIndex > streamIndex\_ rowPosition\_ (uint32) remainingValues\_ (int64) onChunkLoad\_ (callback)

StringColumnReader
""""""""""""""""""

col4 string

scanSpec\_.filter\_: AlwaysTrue

(value)

DictionaryState dictState\_

String ChunkedDecoder
"""""""""""""""""""""

encoding\_ → Dictionary unique_ptr < SeekableInputStream > input\_ shared_ptr < StreamIndex > streamIndex\_ rowPosition\_ (uint32) remainingValues\_ (int64) onChunkLoad\_ → invalidates dictState\_

SelectiveNimbleReader.cpp · SelectiveNimbleRowReader (anonymous namespace)


col4(StringColumnReader).read(0, [17,42,...]) — read() tries the dictionary path before the flat path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For string columns with nimble_preserve_dictionary_encoding=true , readWithDictionary() intercepts the read above the decoder . Instead of the visitor applying the filter per-row during decode, it bulk-reads raw dictionary indices and applies the filter post-hoc on the alphabet via SIMD filterDictionaryIndices() .

Flat path (default)
"""""""""""""""""""

false

(valueHook active · zeroCopy off · preserveDict off · not dict-convertible at start or mid-read chunk boundary)

Filter

Dictionary path (optimization)
""""""""""""""""""""""""""""""

true

(so step 2 reads ALL rows, not filtered)

(side effect)

(so step 5 can use it)

set anyNulls\_ + returnReaderNulls\_ flags

so resultNulls() returns nullsInReadRange\_

(bitmap written by step 2, flags not set yet)


← unchanged

← filtered by 5

← from step 4

← set by step 5

filterDictionaryIndices() example

(a) Test & cache:

FAIL

PASS ✓

(b) Scan indices:

FAIL

PASS

FAIL

PASS

FAIL

PASS

(c) Compact:

⚡

Performance win:

filterCache

StringColumnReader.cpp:330-504 (readWithDictionary) · StringColumnReader.cpp:506-560 (read)

Part B

ChunkedDecoder

Chunk management, I/O, state machine — the bridge between column readers and encodings

1

ChunkedDecoder exposes multiple APIs — which path depends on the reader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Selective Reader (production)
"""""""""""""""""""""""""""""

APIs used:

readWithVisitor(visitor)

skip(n)

Batch Reader (non-selective)
""""""""""""""""""""""""""""

APIs used:

materialize(n, buffer)

skip(n)

No filter support:

2

encoding\_

Encoding*

rowPosition\_

uint32

remainingValues\_

int64

streamIndex\_

StreamIndex?

inputBuffer\_

Buffer

onChunkLoad\_

callback

🔄

onChunkLoad\_ callback

StringColumnReader

DictionaryVector

``decoder_.readWithVisitor(visitor<Filter>)``

Decodes values, applies filter via visitor. Loads chunks transparently.

while rows remaining

⊚ decoder\_.remainingValues\_ == 0?

yes → decoder\_.loadNextChunk()

⊚ decoder\_.computeNextRowIndex()

how many rows can current chunk serve?

⊚ encoding\_- > readWithVisitor(visitor)

→ visitor.process(value) per value → writes outputRows\_ / rawValues\_

⊚ decoder\_.advancePosition()

update rowPosition\_ & remainingValues\_

``decoder_.skip(n)``

Advances position without decoding values.

WITH StreamIndex — O(1)

⊚ streamIndex\_- > lookupChunk(targetRow)

→ {chunkOffset, rowOffset}

⊚ decoder\_.seekToChunk(offset)

→ jump + decoder\_.loadNextChunk()

⊚ encoding\_- > skip(rowOffset)

⊚ decoder\_.advancePosition(n)

WITHOUT StreamIndex — O(chunks)

⊚ decoder\_.remainingValues\_ == 0?

yes → decoder\_.loadNextChunk()

⊚ n < decoder\_.remainingValues\_?

yes → encoding\_- > skip(n) → done no → n -= remainingValues\_ → loop

⊚ decoder\_.advancePosition(n)

Internal: ``decoder_.loadNextChunk()``

Chunk byte layout example: nullable int32 (without chunking: entire stream = a single chunk):

Chunk Header:

4B (content length)

| 1B (Uncompressed)

Prefix:

1B=5 (Nullable)

| 1B Int32 (DataType)

| rows (rowCount, e.g. 10K total rows in chunk)

nulls (SparseBool)

Prefix:

1B=6 (SparseBool)

| 1B Bool (DataType)

| rows (rowCount, e.g. 10K same as parent)

1B sparseVal

Prefix:

1B=0 (Trivial)

| 1B UInt32 (DataType)

| rows (rowCount, e.g. 50 null positions)

(Compressed)

values (compressed)

data (FixedBitWidth)

Prefix:

1B=3 (FixedBitWidth)

| 1B Int32 (DataType)

| rows (rowCount, e.g. 9950 non-null values)

(Compressed)

packed bits (compressed)

encoding blob standard prefix (always readable) compressed data

Call steps:

ensureInput(5B)
"""""""""""""""

chunk header bytes


readChunkHeader()
"""""""""""""""""

(Compressed)


ensureInput(length)
"""""""""""""""""""

all chunk content bytes


``encoding_ = EncodingFactory::create(chunkData)``

↓ Section 3


``encoding_ = root Encoding``

remainingValues\_ = root rowCount (e.g. NullableEncoding's rows)

State updated:

encoding\_

remainingValues\_

onChunkLoad\_

inputData\_/inputSize\_

Part C

Encoding

Data layout, recursive encoding trees, decode algorithms, visitor protocol

3

encoding\_ is a unique_ptr < Encoding > on ChunkedDecoder. Virtual dispatch to concrete type. The encoding never sees the filter — it just decodes values and calls visitor.process(value) .

EncodingFactory::create(chunkData)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recursive descent parser. Reads prefix, switches on EncodingType, recursively builds tree. Leaf encodings decompress eagerly in their constructors. Example: nullable int32 column.

``EncodingFactory::create(chunkData)``

``NullableEncoding<int32>``

1. read prefix:

2. data:

``create() -> SparseBoolEncoding``

1. read prefix:

2. read metadata:

3. data:

``create() -> TrivialEncoding<uint32>``

1. read prefix:

2. read metadata:

3. data:

leaf — no more recursion

``create() -> FixedBitWidthEncoding<int32>``

1. read prefix:

2. read metadata:

3. data:

leaf — no more recursion

4

1. Visitor creation
^^^^^^^^^^^^^^^^^^^

Visitor construction (in ColumnReader::read())

// compile-time template

// e.g. col1 > 50

// [0..N] dense or [3,17,42,...] sparse

// references to column reader's buffers

Filter

// pass to ChunkedDecoder → encoding (filter baked into visitor type)

Dense vs Sparse
^^^^^^^^^^^^^^^

(isDense = true)

check: rows.back() == rows.size() - 1 (0-based within batch)

(isDense = false)

rows = [3, 17, 42, ...] — gaps between survivors subsequent filter columns get sparse from previous filter

``encoding_->readWithVisitor(visitor)``

ENCODING ↔ VISITOR INTERACTION

Encoding (decode)

dense: sequential decode sparse: random access or skip+read ↓ produces value

value →

Filter

— filter is a compile-time type parameter

visitor.process(value, atEnd)

holds:

(from scanSpec)

(dense or sparse)

(ref to column reader)

if filter\_.testValue(value): rawValues\_[numValues\_] = value outputRows\_[numValues\_] = currentRow numValues\_++ else: discard


Output

outputRows\_ rawValues\_ numValues\_

encoding never sees the filter · visitor never sees the data layout · they interact only through process(value)

5

Virtual dispatch to concrete encoding type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example: BlockBitPackingEncoding

.. code-block:: text

    template <typename V>
    void BlockBitPackingEncoding<T>::readWithVisitor(V& visitor, ReadWithVisitorParams& params) {
      // Compile-time check
      if constexpr (kIsSuitableWidth && ExtractToReader && kIsFluidCast) {
        // Runtime check
        if (useFastPath(visitor, nulls)) {
          detail::readWithVisitorFast(*this, visitor, params, nulls);  // → bulkScan
          return;
        }
      }
      // Fallthrough: slow path
      detail::readWithVisitorSlow(visitor, params, skip_lambda, decodeOne_lambda);
    }

Eligibility check
^^^^^^^^^^^^^^^^^

(if constexpr)

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Check
     - Variable
     - Condition
   * - Type width
     - kIsSuitableWidth
     - physicalType is 4 or 8 bytes (int32, int64, float, double)
   * - Extract type
     - —
     - V::Extract is velox::dwio::common::ExtractToReader (standard Velox extract)
   * - Cast compatibility
     - kIsFluidCast
     - sizeof(OutputType) > = sizeof(physicalType) and both are integral (no narrowing)

(useFastPath())

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Check
     - What it verifies
   * - Deterministic filter
     - Filter has no side effects, safe to apply post-decode
   * - No value hooks
     - No aggregation pushdown hooks active
   * - Compatible visitor state
     - Visitor is in a state where bulk processing is valid

Two read paths
^^^^^^^^^^^^^^

YES → readWithVisitorFast()

decode first, filter second

N rows

NO → readWithVisitorSlow()

decode and filter interleaved

2N function calls

for N rows

compute selectedRows

non-null row positions + scatterRows for null gaps

encoding.bulkScan(visitor, selectedRows)

dense: bulkGetWithBaseline32() — SIMD decode all values, then processFixedWidthRun() filters in bulk sparse: fixedBitArray\_.get(selectedRows[i]) + baseline\_ per value

processFixedWidthRun()

(generic utility, not FixedBitWidth-specific)

operates on decoded fixed-width values (int32, int64, double) in buffer: 1. filter: filter\_.testValue(value) per value 2. scatter: place at correct output positions (null gaps) 3. hook: forward to value hooks if present

per value

skip(gap)

advance to next selected row

decodeOne()

decode single value from encoding

visitor.process(value)

filter + store or discard

Used by: Dictionary, Delta, Varint, SparseBool, Constant, Prefix

Concrete examples
^^^^^^^^^^^^^^^^^

col1 (FixedBitWidth, dense, filter: > 50) — Fast path

(already decompressed in constructor)

Step 1: bulkScan → SIMD decode all 5 values into buffer

Step 2: processFixedWidthRun() → filter each

NO

YES

NO

YES

Result: numValues\_ = 3

col3 (Trivial, sparse rows=[1,3,4], filter: < 3.14) — Fast path

Step 1: bulkScan → random access selected rows

Step 2: filter each

YES

NO

Result: numValues\_ = 2

col4 (Dictionary, sparse rows=[1,3], filter: AlwaysTrue) — Slow path

Interleaved skip + decode + filter:

outputRows\_[0]=1, rawValues\_[0]="foo"

outputRows\_[1]=3, rawValues\_[1]="baz"

Result: numValues\_ = 2

6

Advances position without decoding values. Cost and decompression behavior depends on encoding type.

FixedBitWidth / Trivial

position\_ += n

ChunkedBitPacking

O(blocks) — walk headers

Varint

data already decompressed

Delta

data already decompressed

Dictionary

depends on index type

Nullable

depends on inner types

4

Why lazy loading exists
^^^^^^^^^^^^^^^^^^^^^^^

Column pruning (compile-time) removes columns not in the query. Lazy loading (runtime) defers reading columns that are in the query but might not be accessed for every batch. It is an inter-operator optimization — the TableScan does not know what downstream operators will do, so it defers non-filtered columns and lets the pipeline decide.

SELECT a, b FROM t WHERE a > 100 LIMIT 10

lazy

next() never called

SELECT sum(c) FROM t WHERE a > 100 AND expensive_udf(b) > 0

Pipeline: TableScan → Filter → Aggregate

c loaded

NEVER loaded

💡

speculative optimization

When columns become lazy
^^^^^^^^^^^^^^^^^^^^^^^^

During SelectiveStructColumnReaderBase::read() , each child column is checked against four conditions. If all hold, the column is not read during this call — it becomes a LazyVector .

// SelectiveStructColumnReader.cpp:454-458

hasFilter()

// → wrapped as LazyVector in getValues()

isTopLevel()
""""""""""""

Root-level column

``projectOut()``

Column is in SELECT list

``!hasFilter()``

No predicate on this column

``generateLazyChildren_``

Default: true

Decode path implications
^^^^^^^^^^^^^^^^^^^^^^^^

Lazy (no filter on column)
""""""""""""""""""""""""""

Load trigger:

RowSet:

[0, 1, ..., N-1]

bulkScan path:

V::dense = true

Benefit:

never loaded

The ColumnLoader renumbers rows to [0..N-1], so the original sparse positions from the filter column are lost

Eager (has filter on column)
""""""""""""""""""""""""""""

Load trigger:

read()

RowSet:

[3, 17, 42, ...]

bulkScan path:

V::dense = false

Scenario:

WHERE a=5 AND b > 100

Sparse bulkScan can skip unneeded rows/blocks

⚠

always take the dense bulkScan path

eager columns with filters

Concrete examples
^^^^^^^^^^^^^^^^^

SELECT a, b, c FROM t WHERE a = 5 — b, c are lazy

read() phase — StructColumnReader loops over children:

read eagerly

continue (LazyVector)

getValues() — build output RowVector:

Downstream access (Project operator touches b):

dense [0..49]

true

dense

SELECT a, b FROM t WHERE a = 5 AND b > 100 — b is eager, sparse

read() phase — StructColumnReader loops over children:

read eagerly

sparse

false

sparse

Part A: Orchestration

Part B: ChunkedDecoder

Part C: Encoding

col1 (int32)

col3 (double)

col4 (string)

Data flow

Fast / dict path

Slow / lazy

/visualize
