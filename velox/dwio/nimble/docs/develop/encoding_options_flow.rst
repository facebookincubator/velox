
Nimble Writer Configuration Flow
================================

Part 1: Encoding::Options Background

What the struct is, where it's constructed, and which encodings consume which field

1a

velox/dwio/nimble/encodings/common/Encoding.h

.. code-block:: text

   Encoding              useVarintRowCount  bufferPool  preserveDict  freqPartIdx  bbpBlockSize
   Trivial               ●                  ○
   RLE                   ●                  ○
   Dictionary            ●                  ○
   FixedBitWidth         ●                  ○
   BlockBitPacking       ●                  ○           ●
   FreqPartition         ●                  ○           ●             ○
   Delta, Varint,
   SparseBool,
   MainlyConst,
   Constant,
   Sentinel,
   Prefix,
   Nullable,
   ALP                   ●                  ○


useVarintRowCount

bufferPool

Encoding

1b

Write Path

Writer → file writing

WriterOptions fields
^^^^^^^^^^^^^^^^^^^^^^^^^

(from serde params)

Writer encode overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^

buildEncodingOptions()

bbpBlockSize=N

EncodingFactory::encode(options)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

options → select() + EncodingSelection

Serializer / Deserializer

Network/RPC vector transfer

SerializerOptions
^^^^^^^^^^^^^^^^^

useVarint

bbpBlockSize

StreamData
^^^^^^^^^^

useVarintRowCount

bufferPool

EncodingFactory(opts).create()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

options stored, passed to constructors

Selective Reader

ChunkedDecoder read path

SelectiveNimbleRowReader read path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

shared across all columns, default Options{}

NimbleParams::toFormatData read overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

decodingStats*

NimbleData read state
^^^^^^^^^^^^^^^^^^^^^

per-column: holds encodingFactory\_ + decodingStats\_ creates ChunkedDecoders for each stream

ChunkedDecoder::loadChunk()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

copies factory opts and overrides per-create

// copy, not reference

options

preserveDictionaryEncoding

options

decodingStats

``encodingFactory_->create(pool, data, factory, options)``

virtual dispatch

Part 2: Write Path Overview

Encoding::Options

2a

Path A: DWIO FileWriter API

TableService / Ingestion pipelines

Hive Metastore
^^^^^^^^^^^^^^

HiveTableMetadata.sd().serdeInfo().parameters()

FileWriterFactory::create()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

format = NIMBLE → copy serdeParams

NimbleWriterOptionBuilder from FileWriter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.withSerdeParams(schema, serdeParams).build()

optionOverrides.nimbleOverrides(options)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

optional lambda-based overrides from caller

dwio/api/FileWriter.cpp:484-544

Path B: Velox HiveDataSink

Presto / Spark write queries

HiveInsertTableHandle
^^^^^^^^^^^^^^^^^^^^^

serdeParameters\_ (from coordinator)

createWriterOptions()
^^^^^^^^^^^^^^^^^^^^^

options → serdeParameters = table serde params

processConfigs(connectorConfig, session)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

merges session & connector overrides via emplace()

NimbleWriterOptionBuilder from HiveDataSink
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.withSerdeParams(schema, serdeParams).build()

velox/connectors/hive/HiveDataSink.cpp

OVERRIDE PRIORITY (highest → lowest)

Table Serde Param

Session Property

Connector Config

Default

Enforced by std::map::emplace() — only inserts if key does not already exist


NimbleWriterOptionBuilder::withSerdeParams()

map < string,string >

WriterOptions

2b

WriterOptions in constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Encoding

encodingSelectionPolicyCreator

readFactors

(which encodings to try + weights)

compressionOptions

(codec, levels, accept ratio)

blockBitPackingBlockSize

buildEncodingOptions()

→ Encoding::Options

compressionOptions

(duplicate for replay path)

encodingLayoutTree

Flush / Chunking

(stripe size, memory thresholds)

Schema

flatMapColumns, dictionaryArrayColumns, deduplicatedMapColumns

Index

clusterIndexConfig

Memory / Parallelism

encodingExecutor, reclaimerFactory, spillConfig

Stats / Misc

enableStatsCollection, enableStreamDedup, metadata

2c

WriterOptions
^^^^^^^^^^^^^^^^^^

Writer constructor
^^^^^^^^^^^^^^^^^^^^^^^

distributes options to context, creates TabletWriter, invokes factories

Writer

enableChunking, chunk sizes stats collection schema columns encoding layout metadata

FieldWriterContext

buffer growth string buffers parallel encoding flat map nodes dict array nodes

TabletWriter

stream dedup chunk index metadata compression feature reorder

FlushPolicy

stripe raw size memory thresholds target storage sz compression factor

Encoding::Options

EncodingSelection

(policy)

blockBitPackingBlockSize

(options)

ClusterIndexWriter

index columns sort orders key constraints encoding layout

velox/dwio/nimble/writer/Writer.cpp · WriterOptions.h · FieldWriter.h · TabletWriter.h

2d

Writer encode implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

buildEncodingOptions()

(new)

EncodingFactory::encode(policy, data, buffer, options )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

options

(new)

Step 1

Estimation — pick encoding

select(values, stats, options )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

blockSize

options

(new)

picks best encoding → returns EncodingSelectionResult{BBP, comprFactory}

Step 2

Build selection — bundle result + options

EncodingSelection < T >
^^^^^^^^^^^^^^^^^^^^^^^

options

(new)

Step 3

Encode — write data

BBP::encode(selection, values, buffer, options )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

options

512

Estimation 512 = Encoding 512 ✓

Part 3: Read Path (Selective Reader)

Encoding::Options

3a

SelectiveNimbleRowReader option flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EncodingFactory()

legacy::EncodingFactory()

shared across all columns — default Options{}

velox/dwio/nimble/selective/SelectiveNimbleReader.cpp

3b

NimbleParams::toFormatData option extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

per-column

DecodingStats*

runtimeStatistics().decodingStatsSet

type- > id()

type- > type()- > kind()

NimbleData construction
^^^^^^^^^^^^^^^^^^^^^^^

per-column

encodingFactory\_

decodingStats\_

all streams for the same column share the same decodingStats\_

velox/dwio/nimble/selective/NimbleData.cpp

3c

ChunkedDecoder::loadNextChunk()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

copies factory ’ s base options, overrides per-create:

// Copy, not reference.

preserveDictionaryEncoding

decodingStats

options


4-arg

create()

virtual

zeroCopy

velox/dwio/nimble/selective/ChunkedDecoder.cpp

3d

``encodingFactory_->create(pool, data, sbf, options)``

virtual dispatch — depends on zeroCopy flag

nimble::EncodingFactory

zeroCopy = true

options

legacy::EncodingFactory

(prod default)

/*options*/

// drops options, delegates to 3-arg

options

Encoding tree (zeroCopy=true path only)

NullableEncoding

options

options\_

``nimble::``

options

// local, always base class


DictionaryEncoding

options

``nimble::``

options


TrivialEncoding

options

Compression::uncompress

options


zeroCopy=true

options

decodingStats

zeroCopy


All 12 production encodings use this pattern: ALP, Dictionary, Delta, Nullable, MainlyConstant, FreqPartition, ForEncoding, RLE, BBP, SubIntSplit, Trivial (string), DeltaEncoding.

3e

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Encoding
     - What it decompresses
     - Passes decompressCounter?
   * - Trivial < T >
     - values blob
     - ●
   * - Trivial < string >
     - data blob + bitmap
     - ●
   * - FixedBitWidth
     - packed bit array
     - ●
   * - BlockBitPacking
     - packed blocks
     - ●
   * - ForEncoding
     - packed data
     - ●


encodings/legacy/

Compression::uncompress()

/*decompressCounter=*/nullptr

decompressCounter

velox/dwio/nimble/compression/Compression.h · velox/dwio/common/Statistics.h

Ke Wang · 2026-06-25
