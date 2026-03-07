=================
I/O and Buffering
=================

DWIO decouples logical data access from physical I/O through a layered
buffering system. This allows read coalescing, caching, and prefetching
without any changes to the column readers.

BufferedInput
=============

``BufferedInput`` (``dwio/common/BufferedInput.h``) is the basic I/O
scheduler:

1. Column readers call ``enqueue(region)`` to declare byte ranges they will
   need.
2. The caller invokes ``load()`` which coalesces nearby regions into larger
   reads. Two regions are merged if the gap between them is smaller than the
   *merge distance* (default ~1.25 MB). This reduces the number of I/O
   requests at the cost of reading a small amount of unused data.
3. After ``load()`` returns, column readers access the data through
   ``SeekableInputStream`` handles that serve bytes from the in-memory
   buffers.

CachedBufferedInput
===================

``CachedBufferedInput`` (``dwio/common/CachedBufferedInput.h``) extends
``BufferedInput`` with a two-level cache:

.. list-table::
   :widths: 15 50
   :header-rows: 1

   * - Level
     - Description
   * - DRAM
     - ``AsyncDataCache`` stores recently accessed regions in memory. Cache
       entries are reference-counted and evicted in LRU order.
   * - NVMe SSD
     - ``SsdCache`` acts as a second-level victim cache. Evicted DRAM entries
       are written to SSD and can be promoted back on a subsequent hit,
       avoiding a storage round-trip.

Background prefetch is executed on a ``folly::Executor`` thread pool so that
I/O can overlap with CPU-bound decoding.

``ScanTracker`` tracks per-column access frequency to inform prefetch
decisions and cache admission policies.

UnitLoader Abstraction
======================

A *unit* is a logically independent chunk of data that can be loaded and
unloaded independently -- typically a DWRF stripe or a Parquet row group.
The ``UnitLoader`` abstraction (``dwio/common/UnitLoader.h``) manages the
lifecycle:

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - Interface
     - Role
   * - ``LoadUnit``
     - Represents one unit. Provides ``load()`` and ``unload()`` methods.
       Also exposes the unit's row count and I/O size for scheduling decisions.
   * - ``UnitLoaderFactory``
     - Creates a ``UnitLoader`` from a list of ``LoadUnit`` objects.
   * - ``UnitLoader``
     - Orchestrates the loading strategy for a sequence of units.

OnDemandUnitLoader
------------------

``OnDemandUnitLoader`` (``dwio/common/OnDemandUnitLoader.h``) is the default
strategy. It loads each unit synchronously when the ``RowReader`` first
accesses it. No prefetching is performed. This is the simplest strategy and
is suitable when I/O latency is low (e.g. local SSD).

ParallelUnitLoader
------------------

``ParallelUnitLoader`` (``dwio/common/ParallelUnitLoader.h``) prefetches
*N* units ahead of the current read position on an I/O executor thread pool.
This hides I/O latency behind the CPU time spent decoding the current unit.
The look-ahead count is configurable.

Statistics and Metrics
======================

``RuntimeStatistics`` (``dwio/common/Statistics.h``) collects per-scan
metrics:

.. list-table::
   :widths: 25 40
   :header-rows: 1

   * - Metric
     - Description
   * - ``skippedStrides``
     - Number of row groups skipped by filter pushdown.
   * - ``processedStrides``
     - Number of row groups actually decoded.
   * - ``skippedSplits``
     - Entire splits skipped (e.g. when all row groups are filtered out).
   * - ``rawBytesRead``
     - Total bytes read from storage before decompression.

``ColumnReaderStatistics`` provides per-column counters such as the number of
null values decoded, dictionary hits, and flat-map key misses.
