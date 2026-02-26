/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <folly/Range.h>

namespace facebook::velox {

/// Velox metrics Registration.
void registerVeloxMetrics();

constexpr std::string_view kMetricHiveFileHandleGenerateLatencyMs{
    "velox.hive_file_handle_generate_latency_ms"};

constexpr std::string_view kMetricCacheShrinkCount{"velox.cache_shrink_count"};

constexpr std::string_view kMetricCacheShrinkTimeMs{"velox.cache_shrink_ms"};

constexpr std::string_view kMetricMaxSpillLevelExceededCount{
    "velox.spill_max_level_exceeded_count"};

constexpr std::string_view kMetricQueryMemoryReclaimTimeMs{
    "velox.query_memory_reclaim_time_ms"};

constexpr std::string_view kMetricQueryMemoryReclaimedBytes{
    "velox.query_memory_reclaim_bytes"};

constexpr std::string_view kMetricQueryMemoryReclaimCount{
    "velox.query_memory_reclaim_count"};

constexpr std::string_view kMetricTaskMemoryReclaimCount{
    "velox.task_memory_reclaim_count"};

constexpr std::string_view kMetricTaskMemoryReclaimWaitTimeMs{
    "velox.task_memory_reclaim_wait_ms"};

constexpr std::string_view kMetricTaskMemoryReclaimExecTimeMs{
    "velox.task_memory_reclaim_exec_ms"};

constexpr std::string_view kMetricTaskMemoryReclaimWaitTimeoutCount{
    "velox.task_memory_reclaim_wait_timeout_count"};

constexpr std::string_view kMetricTaskSplitsCount{"velox.task_splits_count"};

constexpr std::string_view kMetricOpMemoryReclaimTimeMs{
    "velox.op_memory_reclaim_time_ms"};

constexpr std::string_view kMetricOpMemoryReclaimedBytes{
    "velox.op_memory_reclaim_bytes"};

constexpr std::string_view kMetricOpMemoryReclaimCount{
    "velox.op_memory_reclaim_count"};

constexpr std::string_view kMetricMemoryNonReclaimableCount{
    "velox.memory_non_reclaimable_count"};

constexpr std::string_view kMetricMemoryPoolInitialCapacityBytes{
    "velox.memory_pool_initial_capacity_bytes"};

constexpr std::string_view kMetricMemoryPoolCapacityGrowCount{
    "velox.memory_pool_capacity_growth_count"};

constexpr std::string_view kMetricMemoryPoolUsageLeakBytes{
    "velox.memory_pool_usage_leak_bytes"};

constexpr std::string_view kMetricMemoryPoolReservationLeakBytes{
    "velox.memory_pool_reservation_leak_bytes"};

constexpr std::string_view kMetricMemoryAllocatorDoubleFreeCount{
    "velox.memory_allocator_double_free_count"};

constexpr std::string_view kMetricArbitratorLocalArbitrationCount{
    "velox.arbitrator_local_arbitration_count"};

constexpr std::string_view kMetricArbitratorGlobalArbitrationCount{
    "velox.arbitrator_global_arbitration_count"};

constexpr std::string_view kMetricArbitratorGlobalArbitrationNumReclaimVictims{
    "velox.arbitrator_global_arbitration_num_reclaim_victims"};

constexpr std::string_view kMetricArbitratorGlobalArbitrationFailedVictimCount{
    "velox.arbitrator_global_arbitration_failed_victim_count"};

constexpr std::string_view kMetricArbitratorGlobalArbitrationBytes{
    "velox.arbitrator_global_arbitration_bytes"};

constexpr std::string_view kMetricArbitratorGlobalArbitrationTimeMs{
    "velox.arbitrator_global_arbitration_time_ms"};

constexpr std::string_view kMetricArbitratorGlobalArbitrationWaitCount{
    "velox.arbitrator_global_arbitration_wait_count"};

constexpr std::string_view kMetricArbitratorGlobalArbitrationWaitTimeMs{
    "velox.arbitrator_global_arbitration_wait_time_ms"};

constexpr std::string_view kMetricArbitratorAbortedCount{
    "velox.arbitrator_aborted_count"};

constexpr std::string_view kMetricArbitratorFailuresCount{
    "velox.arbitrator_failures_count"};

constexpr std::string_view kMetricArbitratorOpExecTimeMs{
    "velox.arbitrator_op_exec_time_ms"};

constexpr std::string_view kMetricArbitratorFreeCapacityBytes{
    "velox.arbitrator_free_capacity_bytes"};

constexpr std::string_view kMetricArbitratorFreeReservedCapacityBytes{
    "velox.arbitrator_free_reserved_capacity_bytes"};

constexpr std::string_view kMetricDriverYieldCount{"velox.driver_yield_count"};

constexpr std::string_view kMetricDriverQueueTimeMs{
    "velox.driver_queue_time_ms"};

constexpr std::string_view kMetricDriverExecTimeMs{"velox.driver_exec_time_ms"};

constexpr std::string_view kMetricSpilledInputBytes{"velox.spill_input_bytes"};

constexpr std::string_view kMetricSpilledBytes{"velox.spill_bytes"};

constexpr std::string_view kMetricSpilledRowsCount{"velox.spill_rows_count"};

constexpr std::string_view kMetricSpilledFilesCount{"velox.spill_files_count"};

constexpr std::string_view kMetricSpillFillTimeMs{"velox.spill_fill_time_ms"};

constexpr std::string_view kMetricSpillSortTimeMs{"velox.spill_sort_time_ms"};

constexpr std::string_view kMetricSpillExtractVectorTimeMs{
    "velox.spill_extract_vector_time_ms"};

constexpr std::string_view kMetricSpillSerializationTimeMs{
    "velox.spill_serialization_time_ms"};

constexpr std::string_view kMetricSpillWritesCount{"velox.spill_writes_count"};

constexpr std::string_view kMetricSpillFlushTimeMs{"velox.spill_flush_time_ms"};

constexpr std::string_view kMetricSpillWriteTimeMs{"velox.spill_write_time_ms"};

constexpr std::string_view kMetricSpillMemoryBytes{"velox.spill_memory_bytes"};

constexpr std::string_view kMetricSpillPeakMemoryBytes{
    "velox.spill_peak_memory_bytes"};

constexpr std::string_view kMetricFileWriterEarlyFlushedRawBytes{
    "velox.file_writer_early_flushed_raw_bytes"};

constexpr std::string_view kMetricHiveSortWriterFinishTimeMs{
    "velox.hive_sort_writer_finish_time_ms"};

constexpr std::string_view kMetricArbitratorRequestsCount{
    "velox.arbitrator_requests_count"};

constexpr std::string_view kMetricMemoryAllocatorMappedBytes{
    "velox.memory_allocator_mapped_bytes"};

constexpr std::string_view kMetricMemoryAllocatorExternalMappedBytes{
    "velox.memory_allocator_external_mapped_bytes"};

constexpr std::string_view kMetricMemoryAllocatorAllocatedBytes{
    "velox.memory_allocator_allocated_bytes"};

constexpr std::string_view kMetricMemoryAllocatorTotalUsedBytes{
    "velox.memory_allocator_total_used_bytes"};

constexpr std::string_view kMetricMmapAllocatorDelegatedAllocatedBytes{
    "velox.mmap_allocator_delegated_allocated_bytes"};

constexpr std::string_view kMetricCacheMaxAgeSecs{"velox.cache_max_age_secs"};

constexpr std::string_view kMetricMemoryCacheNumTinyEntries{
    "velox.memory_cache_num_tiny_entries"};

constexpr std::string_view kMetricMemoryCacheNumLargeEntries{
    "velox.memory_cache_num_large_entries"};

constexpr std::string_view kMetricMemoryCacheNumEmptyEntries{
    "velox.memory_cache_num_empty_entries"};

constexpr std::string_view kMetricMemoryCacheNumSharedEntries{
    "velox.memory_cache_num_shared_entries"};

constexpr std::string_view kMetricMemoryCacheNumExclusiveEntries{
    "velox.memory_cache_num_exclusive_entries"};

constexpr std::string_view kMetricMemoryCacheNumPrefetchedEntries{
    "velox.memory_cache_num_prefetched_entries"};

constexpr std::string_view kMetricMemoryCacheTotalTinyBytes{
    "velox.memory_cache_total_tiny_bytes"};

constexpr std::string_view kMetricMemoryCacheTotalLargeBytes{
    "velox.memory_cache_total_large_bytes"};

constexpr std::string_view kMetricMemoryCacheTotalTinyPaddingBytes{
    "velox.memory_cache_total_tiny_padding_bytes"};

constexpr std::string_view kMetricMemoryCacheTotalLargePaddingBytes{
    "velox.memory_cache_total_large_padding_bytes"};

constexpr std::string_view kMetricMemoryCacheTotalPrefetchBytes{
    "velox.memory_cache_total_prefetched_bytes"};

constexpr std::string_view kMetricMemoryCacheSumEvictScore{
    "velox.memory_cache_sum_evict_score"};

constexpr std::string_view kMetricMemoryCacheNumHits{
    "velox.memory_cache_num_hits"};

constexpr std::string_view kMetricMemoryCacheHitBytes{
    "velox.memory_cache_hit_bytes"};

constexpr std::string_view kMetricMemoryCacheNumNew{
    "velox.memory_cache_num_new"};

constexpr std::string_view kMetricMemoryCacheNumEvicts{
    "velox.memory_cache_num_evicts"};

constexpr std::string_view kMetricMemoryCacheNumSavableEvicts{
    "velox.memory_cache_num_savable_evicts"};

constexpr std::string_view kMetricMemoryCacheNumEvictChecks{
    "velox.memory_cache_num_evict_checks"};

constexpr std::string_view kMetricMemoryCacheNumWaitExclusive{
    "velox.memory_cache_num_wait_exclusive"};

constexpr std::string_view kMetricMemoryCacheNumAllocClocks{
    "velox.memory_cache_num_alloc_clocks"};

constexpr std::string_view kMetricMemoryCacheNumShardMutexWaitClocks{
    "velox.memory_cache_num_shard_mutex_wait_clocks"};

/// Cumulative time in milliseconds spent waiting to acquire shard mutexes.
/// Converted from raw clocks assuming ~3GHz CPU frequency.
constexpr std::string_view kMetricMemoryCacheShardMutexWaitTimeMs{
    "velox.memory_cache_shard_mutex_wait_time_ms"};

constexpr std::string_view kMetricMemoryCacheNumAgedOutEntries{
    "velox.memory_cache_num_aged_out_entries"};

constexpr std::string_view kMetricMemoryCacheNumStaleEntries{
    "velox.memory_cache_num_stale_entries"};

constexpr std::string_view kMetricSsdCacheCachedRegions{
    "velox.ssd_cache_cached_regions"};

constexpr std::string_view kMetricSsdCacheCachedEntries{
    "velox.ssd_cache_cached_entries"};

constexpr std::string_view kMetricSsdCacheCachedBytes{
    "velox.ssd_cache_cached_bytes"};

constexpr std::string_view kMetricSsdCacheReadEntries{
    "velox.ssd_cache_read_entries"};

constexpr std::string_view kMetricSsdCacheReadBytes{
    "velox.ssd_cache_read_bytes"};

constexpr std::string_view kMetricSsdCacheWrittenEntries{
    "velox.ssd_cache_written_entries"};

constexpr std::string_view kMetricSsdCacheWrittenBytes{
    "velox.ssd_cache_written_bytes"};

constexpr std::string_view kMetricSsdCacheAgedOutEntries{
    "velox.ssd_cache_aged_out_entries"};

constexpr std::string_view kMetricSsdCacheAgedOutRegions{
    "velox.ssd_cache_aged_out_regions"};

constexpr std::string_view kMetricSsdCacheOpenSsdErrors{
    "velox.ssd_cache_open_ssd_errors"};

constexpr std::string_view kMetricSsdCacheOpenCheckpointErrors{
    "velox.ssd_cache_open_checkpoint_errors"};

constexpr std::string_view kMetricSsdCacheOpenLogErrors{
    "velox.ssd_cache_open_log_errors"};

constexpr std::string_view kMetricSsdCacheMetaFileDeleteErrors{
    "velox.ssd_cache_delete_meta_file_errors"};

constexpr std::string_view kMetricSsdCacheGrowFileErrors{
    "velox.ssd_cache_grow_file_errors"};

constexpr std::string_view kMetricSsdCacheWriteSsdErrors{
    "velox.ssd_cache_write_ssd_errors"};

constexpr std::string_view kMetricSsdCacheWriteNoSpaceErrors{
    "velox.ssd_cache_write_no_space_errors"};

constexpr std::string_view kMetricSsdCacheWriteSsdDropped{
    "velox.ssd_cache_write_ssd_dropped"};

constexpr std::string_view kMetricSsdCacheWriteExceedEntryLimit{
    "velox.ssd_cache_write_exceed_entry_limit"};

constexpr std::string_view kMetricSsdCacheWriteCheckpointErrors{
    "velox.ssd_cache_write_checkpoint_errors"};

constexpr std::string_view kMetricSsdCacheReadCorruptions{
    "velox.ssd_cache_read_corruptions"};

constexpr std::string_view kMetricSsdCacheReadSsdErrors{
    "velox.ssd_cache_read_ssd_errors"};

constexpr std::string_view kMetricSsdCacheReadCheckpointErrors{
    "velox.ssd_cache_read_checkpoint_errors"};

constexpr std::string_view kMetricSsdCacheReadWithoutChecksum{
    "velox.ssd_cache_read_without_checksum"};

constexpr std::string_view kMetricSsdCacheCheckpointsRead{
    "velox.ssd_cache_checkpoints_read"};

constexpr std::string_view kMetricSsdCacheCheckpointsWritten{
    "velox.ssd_cache_checkpoints_written"};

constexpr std::string_view kMetricSsdCacheRegionsEvicted{
    "velox.ssd_cache_regions_evicted"};

constexpr std::string_view kMetricSsdCacheRecoveredEntries{
    "velox.ssd_cache_recovered_entries"};

constexpr std::string_view kMetricExchangeTransactionCreateDelay{
    "velox.exchange.transaction_create_delay_ms"};

constexpr std::string_view kMetricExchangeDataTimeMs{
    "velox.exchange_data_time_ms"};

constexpr std::string_view kMetricExchangeDataBytes{
    "velox.exchange_data_bytes"};

constexpr std::string_view kMetricExchangeDataSize{"velox.exchange_data_size"};

constexpr std::string_view kMetricExchangeDataCount{
    "velox.exchange_data_count"};

constexpr std::string_view kMetricExchangeDataSizeTimeMs{
    "velox.exchange_data_size_time_ms"};

constexpr std::string_view kMetricExchangeDataSizeCount{
    "velox.exchange_data_size_count"};

constexpr std::string_view kMetricStorageThrottledDurationMs{
    "velox.storage_throttled_duration_ms"};

constexpr std::string_view kMetricStorageLocalThrottled{
    "velox.storage_local_throttled_count"};

constexpr std::string_view kMetricStorageGlobalThrottled{
    "velox.storage_global_throttled_count"};

constexpr std::string_view kMetricStorageNetworkThrottled{
    "velox.storage_network_throttled_count"};

constexpr std::string_view kMetricIndexLookupResultRawBytes{
    "velox.index_lookup_result_raw_bytes"};

constexpr std::string_view kMetricIndexLookupResultBytes{
    "velox.index_lookup_result_bytes"};

constexpr std::string_view kMetricIndexLookupTimeMs{
    "velox.index_lookup_time_ms"};

constexpr std::string_view kMetricIndexLookupWaitTimeMs{
    "velox.index_lookup_wait_time_ms"};

constexpr std::string_view kMetricIndexLookupBlockedWaitTimeMs{
    "velox.index_lookup_blocked_wait_time_ms"};

constexpr std::string_view kMetricIndexLookupErrorResultCount{
    "velox.index_lookup_error_result_count"};

constexpr std::string_view kMetricTableScanBatchProcessTimeMs{
    "velox.table_scan_batch_process_time_ms"};

constexpr std::string_view kMetricTableScanBatchBytes{
    "velox.table_scan_batch_bytes"};

constexpr std::string_view kMetricTaskBatchProcessTimeMs{
    "velox.task_batch_process_time_ms"};

constexpr std::string_view kMetricTaskBarrierProcessTimeMs{
    "velox.task_barrier_process_time_ms"};

} // namespace facebook::velox
