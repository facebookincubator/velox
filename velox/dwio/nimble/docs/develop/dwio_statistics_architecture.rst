
DWIO Statistics Architecture
============================

IoStatistics • ColumnReaderStatistics • Cache Stats • Writer Stats • Operator Stats • Full Aggregation Pipeline

1 Reader Stats
--------------

The Velox DWIO reader stack has two independent statistics pipelines that never reference each other. They join only at the operator level, where both contribute keys to the RuntimeMetric map.

Operator (TableScan)

OperatorStats::runtimeStats — map < string, RuntimeMetric >

↑ addIoStatsToRuntimeStats()

↑ toRuntimeMetricMap()

IoStatistics

velox/common/io/IoStatistics.h

Storage / cache layer — "how much did we read?"


Granularity: whole file (one per split)

RuntimeStatistics

velox/dwio/common/Statistics.h

Format reader layer — "how much CPU did decoding cost?"


ColumnReaderStatistics

flattenStringDictionaryValues, pageLoadTimeNs skippedStrides, processedStrides, numStripes

↓ optional (per column)

DecodingStatsSet → DecodingStats

typeKind, decodeCPUTimeNanos, decompressCPUTimeNanos

IoStatistics
^^^^^^^^^^^^

FileDataSource

velox/connectors/hive/FileDataSource.cpp:381

dataIoStats\_ = make_shared < IoStatistics > () — for column data IO metadataIoStats\_ = make_shared < IoStatistics > () — for footer/metadata IO


FileSplitReader → ReaderOptions::setDataIoStats(shared_ptr) → BufferedInput constructor

velox/common/io/IoStatistics.h:44

read\_

IoCounter

ramHit\_

IoCounter

ssdRead\_

IoCounter

prefetch\_

IoCounter

queryThreadIoLatencyUs\_

IoCounter

storageReadLatencyUs\_

IoCounter

ssdCacheReadLatencyUs\_

IoCounter

cacheWaitLatencyUs\_

IoCounter

rawBytesRead\_

atomic

rawOverreadBytes\_

atomic

map < string, OperationCounters >

OperationCounters

(per operation, e.g. "ws_pread")

Not in RuntimeMetric — ODS export only.


stored as shared_ptr in CachedBufferedInput / DirectBufferedInput

BufferedInput

IoStatistics*

enqueue()


CacheInputStream

(cached path)

read\_

ramHit\_

ssdRead\_

queryThreadIoLatencyUs\_


DirectInputStream

(no-cache path)

read\_

prefetch\_

queryThreadIoLatencyUs\_

CoalesceIoStats

— returned by coalesceIo() in groupRequests() , feeds back into IoStatistics:

gaps

readGap().merge()

duplicateRegions

incDuplicateRead()

extraBytes

incRawOverreadBytes()


FileDataSource::getRuntimeStats() calls addIoStatsToRuntimeStats()

OperatorStats::runtimeStats

map < string, RuntimeMetric >

.. list-table::
   :widths: auto
   :header-rows: 1

   * - IoStatistics counter
     - RuntimeMetric key
     - Unit
   * - queryThreadIoLatencyUs\_
     - ioWaitWallNanos
     - nanos (us*1000)
   * - read\_
     - storageReadBytes
     - bytes (sum/count/min/max)
   * - ramHit\_
     - numRamRead + ramReadBytes
     - count + bytes
   * - ssdRead\_
     - numLocalRead + localReadBytes
     - count + bytes
   * - prefetch\_
     - numPrefetch + prefetchBytes
     - count + bytes
   * - readGap\_
     - readGapBytes
     - bytes (sum/count/min/max)
   * - rawOverreadBytes\_
     - overreadBytes
     - bytes

metadata.

rawBytesRead\_

getCompletedBytes()

ColumnReaderStatistics
^^^^^^^^^^^^^^^^^^^^^^

Format RowReader

DwrfRowReader / SelectiveNimbleRowReader

owns ColumnReaderStatistics columnReaderStats\_ as value member


initColumnStatsCollection(schema, options) — if collectColumnCpuMetrics() is on

velox/dwio/common/Statistics.h:669

flattenStringDictionaryValues

int64

pageLoadTimeNs

IoCounter

optional < DecodingMetricsSet >

:596

folly::Synchronized < F14FastMap < nodeId, unique_ptr < DecodingMetrics > > >

DecodingMetrics

(per column, keyed by nodeId) :580

TypeKind

IoCounter


passed by reference: FormatParams(pool, columnReaderStats\_ & )

FormatParams

base class

(NimbleParams / DwrfParams)

runtimeStatistics() returns ColumnReaderStatistics &


columnMetricsSet- > getOrCreate(id)

SelectiveColumnReader

stores DecodingMetrics*


readWithTiming() wraps read()

FORMAT-AGNOSTIC

decodeCPUTimeNanos

DeltaCpuWallTimer callback


toFormatData() extracts IoCounter*

FormatData

NimbleData / DwrfData


& metrics- > decompressCPUTimeNanos

FORMAT-SPECIFIC

decompressCPUTimeNanos

encoding construction site


updateRuntimeStats() → stats.columnReaderStats.mergeFrom(columnReaderStats\_)

RuntimeStatistics

toRuntimeMetricMap()

OperatorStats::runtimeStats

Keys: column\_{nodeId}.{TypeName}.decodeCPUTimeNanos , ...decompressCPUTimeNanos

Decompress Timing: DWRF vs Nimble
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DWRF Path
^^^^^^^^^

StripeStreamsImpl

ColumnReaderStatistics*


getDecompressCounter(nodeId)

& metrics- > decompressCPUTimeNanos


PagedInputStream

decompressCounter\_


``withDecompressStats(counter, [ & ]{ /* zlib/zstd decompress */ })``

velox/dwio/dwrf/common/DecoderUtil.h • PagedInputStream.cpp:186

Nimble Path D108361387
^^^^^^^^^^^^^^^^^^^^^^

NimbleParams::toFormatData()

DecodingStats*

decodingStatsSet


NimbleData

decodingStats\_


makeValuesDecoder()

make*Decoder()

ChunkedDecoder


loadNextChunk()

options.recordDecompressNanos(decompressNanos)

velox/dwio/nimble/selective/NimbleData.cpp • ChunkedDecoder.cpp

2 Cache Stats
-------------

AsyncDataCache

velox/common/caching/AsyncDataCache.h — process-wide singleton

Aggregates from all CacheShard instances


each CacheShard increments its local counters, aggregated on stats() call

velox/common/caching/AsyncDataCache.h:543

Snapshot:

Cumulative:

shared_ptr < SsdCacheStats >

SsdCacheStats

SsdFile.h:142


reported periodically, NOT per-query

PeriodicStatsReporter

NOT exported to RuntimeMetric — completely separate from per-query stats

SimpleLRUCacheStats

SimpleLRUCache.h:30

Used by FileHandleFactory for file handle caching

ScanTracker

ScanTracker.h:87

Feeds adaptive prefetch decisions, not exported as metrics

3 Writer Stats
--------------

Writer

(Nimble)

DwrfWriter

(DWRF)

Owns stats as member; populated during write/flush/close


incremented during write(), flush(), close()

velox/dwio/nimble/writer/Writer.h:61 (Nimble)

Timing:

Sizes:

RuntimeMetric

Embedded: TabletWriter::Stats

duplicateStreamCount duplicateStreamBytes

RatioTracker

(DWRF only)

feeds flush policy decisions


Caller

writer.runtimeStats()

returned to TableWriter operator


Scuba

MetricsLog

StripeFlushMetrics, FileCloseMetrics
