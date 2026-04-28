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

#include "velox/common/config/Config.h"
#include "velox/common/config/ConfigProperty.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::core {

/// Macros for defining query config properties.
///
/// To add a new property, add two lines (order doesn't matter, but
/// grouping related properties together helps readability):
///
/// 1. In QueryConfig.h (inside the class body):
///   VELOX_QUERY_CONFIG(kSpillEnabled, spillEnabled, "spill_enabled",
///       bool, false, "Global enable spilling flag.");
///
/// 2. In QueryConfig.cpp (inside registeredProperties()):
///   VELOX_REGISTER_QUERY_CONFIG(kSpillEnabled);
///
/// VELOX_QUERY_CONFIG generates:
///   - struct kSpillEnabledProperty {
///         using type = bool;
///         static constexpr const char* key = "spill_enabled";
///         static constexpr auto defaultValue = false;
///         static constexpr const char* description = "Global enable...";
///     };
///   - static constexpr const char* kSpillEnabled = "spill_enabled"
///   - bool spillEnabled() const {
///         return get<bool>(kSpillEnabled, false);
///     }
///
/// VELOX_QUERY_CONFIG_PROPERTY generates the same but without the accessor.
/// Used for legacy properties with custom accessor logic (capacity parsing,
/// validation, clamping). New properties should use VELOX_QUERY_CONFIG and
/// put validation in QueryConfig::validateConfig() and
/// QueryConfigProvider::normalize().
/// TODO: Unify validateConfig() and normalize() into a single validation path.
#define VELOX_QUERY_CONFIG_PROPERTY(                 \
    constName, keyStr, CppType, defaultVal, desc)    \
  struct constName##Property {                       \
    using type = CppType;                            \
    static constexpr const char* key = keyStr;       \
    static constexpr auto defaultValue = defaultVal; \
    static constexpr const char* description = desc; \
  };                                                 \
  static constexpr const char* constName = keyStr;

#define VELOX_QUERY_CONFIG(                                                 \
    constName, accessorName, keyStr, CppType, defaultVal, desc)             \
  VELOX_QUERY_CONFIG_PROPERTY(constName, keyStr, CppType, defaultVal, desc) \
  CppType accessorName() const {                                            \
    return get<CppType>(constName, defaultVal);                             \
  }

/// A simple wrapper around velox::IConfig. Defines constants for query
/// config properties and accessor methods.
/// Create per query context. Does not have a singleton instance.
/// Does not allow altering properties on the fly. Only at creation time.
class QueryConfig {
 public:
  explicit QueryConfig(std::unordered_map<std::string, std::string> values);

  // This is needed only to resolve correct ctor for cases like
  // QueryConfig{{}} or QueryConfig({}).
  struct ConfigTag {};

  explicit QueryConfig(
      ConfigTag /*tag*/,
      std::shared_ptr<const config::IConfig> config);

  /// Returns all registered query config properties.
  static const std::vector<config::ConfigProperty>& registeredProperties();

  /// Maximum memory that a query can use on a single host.
  VELOX_QUERY_CONFIG_PROPERTY(
      kQueryMaxMemoryPerNode,
      "query_max_memory_per_node",
      std::string,
      "0B",
      "Maximum memory that a query can use on a single host.")

  /// User provided session timezone. Stores a string with the actual timezone
  /// name, e.g: "America/Los_Angeles".
  VELOX_QUERY_CONFIG(
      kSessionTimezone,
      sessionTimezone,
      "session_timezone",
      std::string,
      "",
      "Session timezone name, e.g. 'America/Los_Angeles'.")

  /// Session start time in milliseconds since Unix epoch. This represents when
  /// the query session began execution. Used for functions that need to know
  /// the session start time (e.g., current_date, localtime).
  VELOX_QUERY_CONFIG(
      kSessionStartTime,
      sessionStartTimeMs,
      "start_time",
      int64_t,
      0,
      "Session start time in milliseconds since Unix epoch.")

  /// If true, timezone-less timestamp conversions (e.g. string to timestamp,
  /// when the string does not specify a timezone) will be adjusted to the user
  /// provided session timezone (if any).
  VELOX_QUERY_CONFIG(
      kAdjustTimestampToTimezone,
      adjustTimestampToTimezone,
      "adjust_timestamp_to_session_timezone",
      bool,
      false,
      "Adjust timezone-less timestamp conversions to session timezone.")

  /// Whether to use the simplified expression evaluation path. False by
  /// default.
  VELOX_QUERY_CONFIG(
      kExprEvalSimplified,
      exprEvalSimplified,
      "expression.eval_simplified",
      bool,
      false,
      "Use simplified expression evaluation path.")

  /// Whether to enable the FlatNoNulls fast path for expression evaluation.
  /// When enabled, expressions skip null checking and vector decoding when all
  /// inputs are flat-encoded with no nulls. True by default.
  VELOX_QUERY_CONFIG(
      kExprEvalFlatNoNulls,
      exprEvalFlatNoNulls,
      "expression.eval_flat_no_nulls",
      bool,
      true,
      "Enable FlatNoNulls fast path for expression evaluation.")

  /// Whether to track CPU usage for individual expressions (supported by call
  /// and cast expressions). False by default. Can be expensive when processing
  /// small batches, e.g. < 10K rows.
  VELOX_QUERY_CONFIG(
      kExprTrackCpuUsage,
      exprTrackCpuUsage,
      "expression.track_cpu_usage",
      bool,
      false,
      "Track CPU usage for individual expressions.")

  /// Takes a comma separated list of function names to track CPU usage for.
  /// Only applicable when kExprTrackCpuUsage is set to false. Is empty by
  /// default.
  VELOX_QUERY_CONFIG(
      kExprTrackCpuUsageForFunctions,
      exprTrackCpuUsageForFunctions,
      "expression.track_cpu_usage_for_functions",
      std::string,
      "",
      "Comma-separated function names to track CPU usage for.")

  /// Enables adaptive per-function CPU usage sampling. When enabled, each
  /// function is calibrated over the first 6 batches (1 warmup + 5
  /// calibration) to measure the overhead of CPU tracking (clock_gettime).
  VELOX_QUERY_CONFIG(
      kExprAdaptiveCpuSampling,
      exprAdaptiveCpuSampling,
      "expression.adaptive_cpu_sampling",
      bool,
      false,
      "Enable adaptive per-function CPU usage sampling.")

  /// Maximum acceptable overhead percentage for CPU tracking per function.
  /// Used with kExprAdaptiveCpuSampling.
  VELOX_QUERY_CONFIG(
      kExprAdaptiveCpuSamplingMaxOverheadPct,
      exprAdaptiveCpuSamplingMaxOverheadPct,
      "expression.adaptive_cpu_sampling_max_overhead_pct",
      double,
      1.0,
      "Maximum acceptable overhead percentage for CPU tracking per function.")

  /// Controls whether non-deterministic expressions are deduplicated during
  /// compilation. If set to false, non-deterministic functions (such as
  /// rand()) will not be deduplicated.
  VELOX_QUERY_CONFIG(
      kExprDedupNonDeterministic,
      exprDedupNonDeterministic,
      "expression.dedup_non_deterministic",
      bool,
      true,
      "Deduplicate non-deterministic expressions during compilation.")

  /// Whether to track CPU usage for stages of individual operators. True by
  /// default. Can be expensive when processing small batches, e.g. < 10K rows.
  VELOX_QUERY_CONFIG(
      kOperatorTrackCpuUsage,
      operatorTrackCpuUsage,
      "track_operator_cpu_usage",
      bool,
      true,
      "Track CPU usage for stages of individual operators.")

  /// Flags used to configure the CAST operator:
  VELOX_QUERY_CONFIG(
      kLegacyCast,
      isLegacyCast,
      "legacy_cast",
      bool,
      false,
      "Use legacy CAST behavior.")

  /// This flag makes the Row conversion to by applied in a way that the casting
  /// row field are matched by name instead of position.
  VELOX_QUERY_CONFIG(
      kCastMatchStructByName,
      isMatchStructByName,
      "cast_match_struct_by_name",
      bool,
      false,
      "Match Row fields by name instead of position in CAST.")

  /// Reduce() function will throw an error if encountered an array of size
  /// greater than this.
  VELOX_QUERY_CONFIG(
      kExprMaxArraySizeInReduce,
      exprMaxArraySizeInReduce,
      "expression.max_array_size_in_reduce",
      uint64_t,
      100'000,
      "Maximum array size allowed in reduce() function.")

  /// Controls maximum number of compiled regular expression patterns per
  /// function instance per thread of execution.
  VELOX_QUERY_CONFIG(
      kExprMaxCompiledRegexes,
      exprMaxCompiledRegexes,
      "expression.max_compiled_regexes",
      uint64_t,
      100,
      "Maximum compiled regex patterns per function instance per thread.")

  /// Used for backpressure to block local exchange producers when the local
  /// exchange buffer reaches or exceeds this size.
  VELOX_QUERY_CONFIG(
      kMaxLocalExchangeBufferSize,
      maxLocalExchangeBufferSize,
      "max_local_exchange_buffer_size",
      uint64_t,
      32UL << 20,
      "Max local exchange buffer size in bytes for backpressure.")

  /// Limits the number of partitions created by a local exchange.
  VELOX_QUERY_CONFIG(
      kMaxLocalExchangePartitionCount,
      maxLocalExchangePartitionCount,
      "max_local_exchange_partition_count",
      uint32_t,
      std::numeric_limits<uint32_t>::max(),
      "Maximum number of partitions in local exchange.")

  /// Minimum number of local exchange output partitions to use buffered
  /// partitioning.
  VELOX_QUERY_CONFIG(
      kMinLocalExchangePartitionCountToUsePartitionBuffer,
      minLocalExchangePartitionCountToUsePartitionBuffer,
      "min_local_exchange_partition_count_to_use_partition_buffer",
      uint32_t,
      33,
      "Minimum output partitions to use buffered partitioning.")

  /// Maximum size in bytes to accumulate for a single partition of a local
  /// exchange before flushing.
  VELOX_QUERY_CONFIG(
      kMaxLocalExchangePartitionBufferSize,
      maxLocalExchangePartitionBufferSize,
      "max_local_exchange_partition_buffer_size",
      uint64_t,
      64UL * 1024,
      "Max bytes to accumulate per local exchange partition.")

  /// Try to preserve the encoding of the input vector when copying it to the
  /// buffer.
  VELOX_QUERY_CONFIG(
      kLocalExchangePartitionBufferPreserveEncoding,
      localExchangePartitionBufferPreserveEncoding,
      "local_exchange_partition_buffer_preserve_encoding",
      bool,
      false,
      "Preserve encoding of input vector in partition buffer.")

  /// Maximum number of vectors buffered in each local merge source before
  /// blocking to wait for consumers.
  VELOX_QUERY_CONFIG(
      kLocalMergeSourceQueueSize,
      localMergeSourceQueueSize,
      "local_merge_source_queue_size",
      uint32_t,
      2,
      "Maximum vectors buffered in each local merge source.")

  /// Maximum size in bytes to accumulate in ExchangeQueue. Enforced
  /// approximately, not strictly.
  VELOX_QUERY_CONFIG(
      kMaxExchangeBufferSize,
      maxExchangeBufferSize,
      "exchange.max_buffer_size",
      uint64_t,
      32UL << 20,
      "Maximum bytes to accumulate in ExchangeQueue.")

  /// Maximum size in bytes to accumulate among all sources of the merge
  /// exchange. Enforced approximately, not strictly.
  VELOX_QUERY_CONFIG(
      kMaxMergeExchangeBufferSize,
      maxMergeExchangeBufferSize,
      "merge_exchange.max_buffer_size",
      uint64_t,
      128UL << 20,
      "Maximum bytes to accumulate among all merge exchange sources.")

  /// The minimum number of bytes to accumulate in the ExchangeQueue
  /// before unblocking a consumer.
  VELOX_QUERY_CONFIG(
      kMinExchangeOutputBatchBytes,
      minExchangeOutputBatchBytes,
      "min_exchange_output_batch_bytes",
      uint64_t,
      2UL << 20,
      "Minimum bytes to accumulate before unblocking an exchange consumer.")

  VELOX_QUERY_CONFIG(
      kMaxPartialAggregationMemory,
      maxPartialAggregationMemoryUsage,
      "max_partial_aggregation_memory",
      uint64_t,
      1L << 24,
      "Maximum memory for partial aggregation.")

  VELOX_QUERY_CONFIG(
      kMaxExtendedPartialAggregationMemory,
      maxExtendedPartialAggregationMemoryUsage,
      "max_extended_partial_aggregation_memory",
      uint64_t,
      1L << 26,
      "Maximum memory for extended partial aggregation.")

  VELOX_QUERY_CONFIG(
      kAbandonPartialAggregationMinRows,
      abandonPartialAggregationMinRows,
      "abandon_partial_aggregation_min_rows",
      int32_t,
      100'000,
      "Minimum input rows before checking whether to abandon partial aggregation.")

  VELOX_QUERY_CONFIG(
      kAbandonPartialAggregationMinPct,
      abandonPartialAggregationMinPct,
      "abandon_partial_aggregation_min_pct",
      int32_t,
      80,
      "Abandon partial aggregation if reduction percentage exceeds this.")

  /// Memory threshold in bytes for triggering string compaction during
  /// global aggregation. Disabled by default (0).
  VELOX_QUERY_CONFIG(
      kAggregationCompactionBytesThreshold,
      aggregationCompactionBytesThreshold,
      "aggregation_compaction_bytes_threshold",
      uint64_t,
      0,
      "Memory threshold in bytes for triggering string compaction during aggregation.")

  /// Ratio of unused (evicted) bytes to total bytes that triggers compaction.
  VELOX_QUERY_CONFIG(
      kAggregationCompactionUnusedMemoryRatio,
      aggregationCompactionUnusedMemoryRatio,
      "aggregation_compaction_unused_memory_ratio",
      double,
      0.25,
      "Ratio of unused bytes to total bytes that triggers compaction.")

  /// If true, enables lightweight memory compaction before spilling during
  /// memory reclaim in aggregation.
  VELOX_QUERY_CONFIG(
      kAggregationMemoryCompactionReclaimEnabled,
      aggregationMemoryCompactionReclaimEnabled,
      "aggregation_memory_compaction_reclaim_enabled",
      bool,
      false,
      "Enable memory compaction before spilling during aggregation reclaim.")

  VELOX_QUERY_CONFIG(
      kAbandonPartialTopNRowNumberMinRows,
      abandonPartialTopNRowNumberMinRows,
      "abandon_partial_topn_row_number_min_rows",
      int32_t,
      100'000,
      "Minimum rows before checking whether to abandon partial TopN row number.")

  VELOX_QUERY_CONFIG(
      kAbandonPartialTopNRowNumberMinPct,
      abandonPartialTopNRowNumberMinPct,
      "abandon_partial_topn_row_number_min_pct",
      int32_t,
      80,
      "Abandon partial TopN row number if reduction percentage exceeds this.")

  /// Number of input rows to receive before starting to check whether to
  /// abandon building a HashTable without duplicates in HashBuild.
  VELOX_QUERY_CONFIG(
      kAbandonDedupHashMapMinRows,
      abandonHashBuildDedupMinRows,
      "abandon_dedup_hashmap_min_rows",
      int32_t,
      100'000,
      "Minimum rows before checking whether to abandon dedup hash map.")

  /// Abandons building a HashTable without duplicates in HashBuild for left
  /// semi/anti join if the percentage of distinct keys exceeds this threshold.
  VELOX_QUERY_CONFIG(
      kAbandonDedupHashMapMinPct,
      abandonHashBuildDedupMinPct,
      "abandon_dedup_hashmap_min_pct",
      int32_t,
      0,
      "Abandon dedup hash map if distinct key percentage exceeds this. 0 disables.")

  VELOX_QUERY_CONFIG(
      kMaxElementsSizeInRepeatAndSequence,
      maxElementsSizeInRepeatAndSequence,
      "max_elements_size_in_repeat_and_sequence",
      int32_t,
      10'000,
      "Maximum elements size in repeat and sequence functions.")

  /// If true, the PartitionedOutput operator will flush rows eagerly.
  VELOX_QUERY_CONFIG(
      kPartitionedOutputEagerFlush,
      partitionedOutputEagerFlush,
      "partitioned_output_eager_flush",
      bool,
      false,
      "Flush PartitionedOutput rows eagerly without buffering.")

  /// The maximum number of bytes to buffer in PartitionedOutput operator to
  /// avoid creating tiny SerializedPages.
  VELOX_QUERY_CONFIG(
      kMaxPartitionedOutputBufferSize,
      maxPartitionedOutputBufferSize,
      "max_page_partitioning_buffer_size",
      uint64_t,
      32UL << 20,
      "Maximum bytes to buffer in PartitionedOutput operator.")

  /// The maximum size in bytes for the task's buffered output.
  VELOX_QUERY_CONFIG(
      kMaxOutputBufferSize,
      maxOutputBufferSize,
      "max_output_buffer_size",
      uint64_t,
      32UL << 20,
      "Maximum size in bytes for the task's buffered output.")

  /// Preferred size of batches in bytes to be returned by operators from
  /// Operator::getOutput.
  VELOX_QUERY_CONFIG(
      kPreferredOutputBatchBytes,
      preferredOutputBatchBytes,
      "preferred_output_batch_bytes",
      uint64_t,
      10UL << 20,
      "Preferred size of output batches in bytes.")

  /// Preferred number of rows to be returned by operators from
  /// Operator::getOutput. Used when average row size is not known.
  VELOX_QUERY_CONFIG_PROPERTY(
      kPreferredOutputBatchRows,
      "preferred_output_batch_rows",
      uint32_t,
      1024,
      "Preferred number of rows in output batches.")

  /// Max number of rows that could be return by operators from
  /// Operator::getOutput.
  VELOX_QUERY_CONFIG_PROPERTY(
      kMaxOutputBatchRows,
      "max_output_batch_rows",
      uint32_t,
      10000,
      "Maximum number of rows in output batches.")

  /// Initial output batch size in rows for MergeJoin operator.
  VELOX_QUERY_CONFIG_PROPERTY(
      kMergeJoinOutputBatchStartSize,
      "merge_join_output_batch_start_size",
      uint32_t,
      0,
      "Initial output batch size in rows for MergeJoin. 0 disables dynamic adjustment.")

  /// TableScan operator will exit getOutput() method after this many
  /// milliseconds even if it has no data to return yet. Zero means 'no time
  /// limit'.
  VELOX_QUERY_CONFIG(
      kTableScanGetOutputTimeLimitMs,
      tableScanGetOutputTimeLimitMs,
      "table_scan_getoutput_time_limit_ms",
      uint32_t,
      5'000,
      "Time limit in ms for TableScan getOutput(). 0 means no limit.")

  /// If non-zero, overrides the number of rows in each output batch produced
  /// by the TableScan operator. Zero means 'no override'.
  VELOX_QUERY_CONFIG(
      kTableScanOutputBatchRowsOverride,
      tableScanOutputBatchRowsOverride,
      "table_scan_output_batch_rows_override",
      uint32_t,
      0,
      "Override number of rows in TableScan output batches. 0 means no override.")

  /// If false, the 'group by' code is forced to use generic hash mode
  /// hashtable.
  VELOX_QUERY_CONFIG(
      kHashAdaptivityEnabled,
      hashAdaptivityEnabled,
      "hash_adaptivity_enabled",
      bool,
      true,
      "Enable hash adaptivity for group-by hash mode selection.")

  /// If true, the conjunction expression can reorder inputs based on the time
  /// taken to calculate them.
  VELOX_QUERY_CONFIG(
      kAdaptiveFilterReorderingEnabled,
      adaptiveFilterReorderingEnabled,
      "adaptive_filter_reordering_enabled",
      bool,
      true,
      "Allow conjunction to reorder inputs based on evaluation time.")

  /// If true, allow hash probe drivers to generate build-side rows in parallel.
  VELOX_QUERY_CONFIG(
      kParallelOutputJoinBuildRowsEnabled,
      parallelOutputJoinBuildRowsEnabled,
      "parallel_output_join_build_rows_enabled",
      bool,
      false,
      "Allow hash probe drivers to generate build-side rows in parallel.")

  /// Global enable spilling flag.
  VELOX_QUERY_CONFIG(
      kSpillEnabled,
      spillEnabled,
      "spill_enabled",
      bool,
      false,
      "Global enable spilling flag.")

  /// Aggregation spilling flag, only applies if "spill_enabled" flag is set.
  VELOX_QUERY_CONFIG(
      kAggregationSpillEnabled,
      aggregationSpillEnabled,
      "aggregation_spill_enabled",
      bool,
      true,
      "Enable aggregation spilling. Requires spill_enabled.")

  /// Join spilling flag, only applies if "spill_enabled" flag is set.
  VELOX_QUERY_CONFIG(
      kJoinSpillEnabled,
      joinSpillEnabled,
      "join_spill_enabled",
      bool,
      true,
      "Enable join spilling. Requires spill_enabled.")

  /// Config to enable hash join spill for mixed grouped execution mode.
  VELOX_QUERY_CONFIG(
      kMixedGroupedModeHashJoinSpillEnabled,
      mixedGroupedModeHashJoinSpillEnabled,
      "mixed_grouped_mode_hash_join_spill_enabled",
      bool,
      false,
      "Enable hash join spill for mixed grouped execution mode.")

  /// OrderBy spilling flag, only applies if "spill_enabled" flag is set.
  VELOX_QUERY_CONFIG(
      kOrderBySpillEnabled,
      orderBySpillEnabled,
      "order_by_spill_enabled",
      bool,
      true,
      "Enable order-by spilling. Requires spill_enabled.")

  /// Window spilling flag, only applies if "spill_enabled" flag is set.
  VELOX_QUERY_CONFIG(
      kWindowSpillEnabled,
      windowSpillEnabled,
      "window_spill_enabled",
      bool,
      true,
      "Enable window spilling. Requires spill_enabled.")

  /// When processing spilled window data, read batches of whole partitions
  /// having at least that many rows.
  VELOX_QUERY_CONFIG(
      kWindowSpillMinReadBatchRows,
      windowSpillMinReadBatchRows,
      "window_spill_min_read_batch_rows",
      uint32_t,
      1'000,
      "Minimum rows to read per batch when processing spilled window data.")

  /// If true, the memory arbitrator will reclaim memory from table writer by
  /// flushing its buffered data to disk. Requires spill_enabled.
  VELOX_QUERY_CONFIG(
      kWriterSpillEnabled,
      writerSpillEnabled,
      "writer_spill_enabled",
      bool,
      true,
      "Enable writer spilling by flushing buffered data to disk.")

  /// RowNumber spilling flag, only applies if "spill_enabled" flag is set.
  VELOX_QUERY_CONFIG(
      kRowNumberSpillEnabled,
      rowNumberSpillEnabled,
      "row_number_spill_enabled",
      bool,
      true,
      "Enable RowNumber spilling. Requires spill_enabled.")

  /// MarkDistinct spilling flag, only applies if "spill_enabled" flag is set.
  VELOX_QUERY_CONFIG(
      kMarkDistinctSpillEnabled,
      markDistinctSpillEnabled,
      "mark_distinct_spill_enabled",
      bool,
      false,
      "Enable MarkDistinct spilling. Requires spill_enabled.")

  /// TopNRowNumber spilling flag, only applies if "spill_enabled" flag is set.
  VELOX_QUERY_CONFIG(
      kTopNRowNumberSpillEnabled,
      topNRowNumberSpillEnabled,
      "topn_row_number_spill_enabled",
      bool,
      true,
      "Enable TopNRowNumber spilling. Requires spill_enabled.")

  /// LocalMerge spilling flag, only applies if "spill_enabled" flag is set.
  VELOX_QUERY_CONFIG(
      kLocalMergeSpillEnabled,
      localMergeSpillEnabled,
      "local_merge_spill_enabled",
      bool,
      false,
      "Enable LocalMerge spilling. Requires spill_enabled.")

  /// Specify the max number of local sources to merge at a time.
  /// Uses std::numeric_limits<uint32_t>::max() as default.
  static constexpr const char* kLocalMergeMaxNumMergeSources =
      "local_merge_max_num_merge_sources";

  /// The max row numbers to fill and spill for each spill run. Default is
  /// 12 million rows which uses ~128MB memory.
  VELOX_QUERY_CONFIG(
      kMaxSpillRunRows,
      maxSpillRunRows,
      "max_spill_run_rows",
      uint64_t,
      12UL << 20,
      "Maximum rows to fill and spill per spill run. 0 means no limit.")

  /// The max spill bytes limit set for each query. Default is 100 GB.
  VELOX_QUERY_CONFIG(
      kMaxSpillBytes,
      maxSpillBytes,
      "max_spill_bytes",
      uint64_t,
      100UL << 30,
      "Maximum total spill bytes per query. 0 means no limit.")

  /// The max allowed spilling level with zero being the initial spilling level.
  VELOX_QUERY_CONFIG(
      kMaxSpillLevel,
      maxSpillLevel,
      "max_spill_level",
      int32_t,
      1,
      "Maximum allowed spilling level. -1 means no limit.")

  /// The max allowed spill file size. If it is zero, then there is no limit.
  VELOX_QUERY_CONFIG(
      kMaxSpillFileSize,
      maxSpillFileSize,
      "max_spill_file_size",
      uint64_t,
      0,
      "Maximum spill file size. 0 means no limit.")

  VELOX_QUERY_CONFIG(
      kSpillCompressionKind,
      spillCompressionKind,
      "spill_compression_codec",
      std::string,
      "none",
      "Compression codec for spill data.")

  /// The max number of files to merge at a time when merging sorted spilled
  /// files. 0 means unlimited.
  VELOX_QUERY_CONFIG(
      kSpillNumMaxMergeFiles,
      spillNumMaxMergeFiles,
      "spill_num_max_merge_files",
      uint32_t,
      0,
      "Maximum files to merge at a time when merging sorted spill files. 0 means unlimited.")

  /// Enable the prefix sort or fallback to timsort in spill.
  VELOX_QUERY_CONFIG(
      kSpillPrefixSortEnabled,
      spillPrefixSortEnabled,
      "spill_prefixsort_enabled",
      bool,
      false,
      "Enable prefix sort in spill instead of timsort.")

  /// Specifies spill write buffer size in bytes. 0 disables buffering.
  VELOX_QUERY_CONFIG(
      kSpillWriteBufferSize,
      spillWriteBufferSize,
      "spill_write_buffer_size",
      uint64_t,
      1L << 20,
      "Spill write buffer size in bytes. 0 disables buffering.")

  /// Specifies the buffer size in bytes to read from one spilled file.
  VELOX_QUERY_CONFIG(
      kSpillReadBufferSize,
      spillReadBufferSize,
      "spill_read_buffer_size",
      uint64_t,
      1L << 20,
      "Buffer size in bytes to read from one spilled file.")

  /// Config used to create spill files.
  VELOX_QUERY_CONFIG(
      kSpillFileCreateConfig,
      spillFileCreateConfig,
      "spill_file_create_config",
      std::string,
      "",
      "Config for creating spill files, passed to underlying file system.")

  /// Config used to create aggregation spill files.
  VELOX_QUERY_CONFIG(
      kAggregationSpillFileCreateConfig,
      aggregationSpillFileCreateConfig,
      "aggregation_spill_file_create_config",
      std::string,
      "",
      "Config for creating aggregation spill files.")

  /// Config used to create hash join spill files.
  VELOX_QUERY_CONFIG(
      kHashJoinSpillFileCreateConfig,
      hashJoinSpillFileCreateConfig,
      "hash_join_spill_file_create_config",
      std::string,
      "",
      "Config for creating hash join spill files.")

  /// Config used to create row number spill files.
  VELOX_QUERY_CONFIG(
      kRowNumberSpillFileCreateConfig,
      rowNumberSpillFileCreateConfig,
      "row_number_spill_file_create_config",
      std::string,
      "",
      "Config for creating row number spill files.")

  /// Default offset spill start partition bit.
  VELOX_QUERY_CONFIG(
      kSpillStartPartitionBit,
      spillStartPartitionBit,
      "spiller_start_partition_bit",
      uint8_t,
      48,
      "Start partition bit offset for spilling.")

  /// Default number of spill partition bits. Max 3 (8-way partitioning).
  VELOX_QUERY_CONFIG_PROPERTY(
      kSpillNumPartitionBits,
      "spiller_num_partition_bits",
      uint8_t,
      3,
      "Number of bits for spill partition calculation.")

  /// The minimal available spillable memory reservation in percentage of the
  /// current memory usage.
  VELOX_QUERY_CONFIG(
      kMinSpillableReservationPct,
      minSpillableReservationPct,
      "min_spillable_reservation_pct",
      int32_t,
      5,
      "Minimum available spillable memory reservation as percentage of usage.")

  /// The spillable memory reservation growth percentage.
  VELOX_QUERY_CONFIG(
      kSpillableReservationGrowthPct,
      spillableReservationGrowthPct,
      "spillable_reservation_growth_pct",
      int32_t,
      10,
      "Spillable memory reservation growth percentage.")

  /// Minimum memory footprint size required to reclaim memory from a file
  /// writer by flushing its buffered data to disk.
  VELOX_QUERY_CONFIG(
      kWriterFlushThresholdBytes,
      writerFlushThresholdBytes,
      "writer_flush_threshold_bytes",
      uint64_t,
      96L << 20,
      "Minimum memory footprint to reclaim from a file writer by flushing.")

  /// If true, array_agg() aggregation function will ignore nulls in the input.
  VELOX_QUERY_CONFIG(
      kPrestoArrayAggIgnoreNulls,
      prestoArrayAggIgnoreNulls,
      "presto.array_agg.ignore_nulls",
      bool,
      false,
      "If true, array_agg() ignores nulls in the input.")

  /// If true, Spark function's behavior is ANSI-compliant.
  VELOX_QUERY_CONFIG(
      kSparkAnsiEnabled,
      sparkAnsiEnabled,
      "spark.ansi_enabled",
      bool,
      false,
      "Enable ANSI-compliant behavior for Spark functions.")

  /// The default number of expected items for the bloomfilter.
  VELOX_QUERY_CONFIG(
      kSparkBloomFilterExpectedNumItems,
      sparkBloomFilterExpectedNumItems,
      "spark.bloom_filter.expected_num_items",
      int64_t,
      1'000'000L,
      "Default number of expected items for the Spark bloom filter.")

  /// The default number of bits to use for the bloom filter.
  VELOX_QUERY_CONFIG(
      kSparkBloomFilterNumBits,
      sparkBloomFilterNumBits,
      "spark.bloom_filter.num_bits",
      int64_t,
      8'388'608L,
      "Default number of bits for the Spark bloom filter.")

  /// The max number of bits to use for the bloom filter.
  VELOX_QUERY_CONFIG(
      kSparkBloomFilterMaxNumBits,
      sparkBloomFilterMaxNumBits,
      "spark.bloom_filter.max_num_bits",
      int64_t,
      67'108'864,
      "Maximum number of bits for the Spark bloom filter.")

  /// The max number of items to use for the bloom filter.
  VELOX_QUERY_CONFIG(
      kSparkBloomFilterMaxNumItems,
      sparkBloomFilterMaxNumItems,
      "spark.bloom_filter.max_num_items",
      int64_t,
      4'000'000L,
      "Maximum number of items for the Spark bloom filter.")

  /// The current spark partition id. No default (throws if not set).
  static constexpr const char* kSparkPartitionId = "spark.partition_id";

  /// If true, simple date formatter is used for time formatting and parsing.
  VELOX_QUERY_CONFIG(
      kSparkLegacyDateFormatter,
      sparkLegacyDateFormatter,
      "spark.legacy_date_formatter",
      bool,
      false,
      "Use simple date formatter instead of Joda for Spark.")

  /// If true, Spark statistical aggregation functions return NaN instead of
  /// NULL when dividing by zero.
  VELOX_QUERY_CONFIG(
      kSparkLegacyStatisticalAggregate,
      sparkLegacyStatisticalAggregate,
      "spark.legacy_statistical_aggregate",
      bool,
      false,
      "Return NaN instead of NULL for Spark statistical aggregation on divide-by-zero.")

  /// If true, ignore null fields when generating JSON string.
  VELOX_QUERY_CONFIG(
      kSparkJsonIgnoreNullFields,
      sparkJsonIgnoreNullFields,
      "spark.json_ignore_null_fields",
      bool,
      true,
      "Ignore null fields when generating JSON string in Spark.")

  /// If true, collect_list aggregate function will ignore nulls in the input.
  VELOX_QUERY_CONFIG(
      kSparkCollectListIgnoreNulls,
      sparkCollectListIgnoreNulls,
      "spark.collect_list.ignore_nulls",
      bool,
      true,
      "If true, Spark collect_list() ignores nulls in the input.")

  /// The number of local parallel table writer operators per task.
  VELOX_QUERY_CONFIG(
      kTaskWriterCount,
      taskWriterCount,
      "task_writer_count",
      uint32_t,
      4,
      "Number of local parallel table writer operators per task.")

  /// The number of local parallel table writer operators per task for
  /// partitioned writes. If not set, use "task_writer_count". No default.
  static constexpr const char* kTaskPartitionedWriterCount =
      "task_partitioned_writer_count";

  /// If true, finish the hash probe on an empty build table for a specific set
  /// of hash joins.
  VELOX_QUERY_CONFIG(
      kHashProbeFinishEarlyOnEmptyBuild,
      hashProbeFinishEarlyOnEmptyBuild,
      "hash_probe_finish_early_on_empty_build",
      bool,
      false,
      "Finish hash probe early on empty build table.")

  /// Whether hash probe can generate any dynamic filter and push down to
  /// upstream operators.
  VELOX_QUERY_CONFIG(
      kHashProbeDynamicFilterPushdownEnabled,
      hashProbeDynamicFilterPushdownEnabled,
      "hash_probe_dynamic_filter_pushdown_enabled",
      bool,
      true,
      "Enable dynamic filter generation and pushdown from hash probe.")

  /// Whether hash probe can generate dynamic filter for string types.
  VELOX_QUERY_CONFIG(
      kHashProbeStringDynamicFilterPushdownEnabled,
      hashProbeStringDynamicFilterPushdownEnabled,
      "hash_probe_string_dynamic_filter_pushdown_enabled",
      bool,
      false,
      "Enable dynamic filter pushdown for string types from hash probe.")

  /// The maximum byte size of Bloom filter from hash probe. 0 = disabled.
  VELOX_QUERY_CONFIG(
      kHashProbeBloomFilterPushdownMaxSize,
      hashProbeBloomFilterPushdownMaxSize,
      "hash_probe_bloom_filter_pushdown_max_size",
      uint64_t,
      0,
      "Maximum byte size of Bloom filter from hash probe. 0 disables.")

  /// The minimum number of table rows that can trigger the parallel hash join
  /// table build.
  VELOX_QUERY_CONFIG(
      kMinTableRowsForParallelJoinBuild,
      minTableRowsForParallelJoinBuild,
      "min_table_rows_for_parallel_join_build",
      uint32_t,
      1'000,
      "Minimum table rows to trigger parallel hash join table build.")

  /// If set to true, validate output vectors of every operator for consistency.
  VELOX_QUERY_CONFIG(
      kValidateOutputFromOperators,
      validateOutputFromOperators,
      "debug.validate_output_from_operators",
      bool,
      false,
      "Validate output vectors from every operator for consistency.")

  /// If true, enable caches in expression evaluation for performance.
  VELOX_QUERY_CONFIG(
      kEnableExpressionEvaluationCache,
      isExpressionEvaluationCacheEnabled,
      "enable_expression_evaluation_cache",
      bool,
      true,
      "Enable caches in expression evaluation for performance.")

  /// For a given shared subexpression, the maximum distinct sets of inputs we
  /// cache results for.
  VELOX_QUERY_CONFIG(
      kMaxSharedSubexprResultsCached,
      maxSharedSubexprResultsCached,
      "max_shared_subexpr_results_cached",
      uint32_t,
      10,
      "Maximum distinct input sets to cache for shared subexpressions.")

  /// Maximum number of splits to preload. Set to 0 to disable preloading.
  VELOX_QUERY_CONFIG(
      kMaxSplitPreloadPerDriver,
      maxSplitPreloadPerDriver,
      "max_split_preload_per_driver",
      int32_t,
      2,
      "Maximum number of splits to preload. 0 disables preloading.")

  /// If not zero, specifies the cpu time slice limit in ms that a driver thread
  /// can continuously run without yielding.
  VELOX_QUERY_CONFIG(
      kDriverCpuTimeSliceLimitMs,
      driverCpuTimeSliceLimitMs,
      "driver_cpu_time_slice_limit_ms",
      uint32_t,
      0,
      "CPU time slice limit in ms before yielding. 0 means no limit.")

  /// Window operator sub-partition count per thread.
  VELOX_QUERY_CONFIG(
      kWindowNumSubPartitions,
      windowNumSubPartitions,
      "window_num_sub_partitions",
      uint32_t,
      1,
      "Number of sub-partitions per thread in Window operator.")

  /// Maximum number of bytes to use for the normalized key in prefix-sort.
  VELOX_QUERY_CONFIG(
      kPrefixSortNormalizedKeyMaxBytes,
      prefixSortNormalizedKeyMaxBytes,
      "prefixsort_normalized_key_max_bytes",
      uint32_t,
      128,
      "Maximum bytes for normalized key in prefix-sort. 0 disables.")

  /// Minimum number of rows to use prefix-sort.
  VELOX_QUERY_CONFIG(
      kPrefixSortMinRows,
      prefixSortMinRows,
      "prefixsort_min_rows",
      uint32_t,
      128,
      "Minimum rows to use prefix-sort.")

  /// Maximum number of bytes to be stored in prefix-sort buffer for a string
  /// key.
  VELOX_QUERY_CONFIG(
      kPrefixSortMaxStringPrefixLength,
      prefixSortMaxStringPrefixLength,
      "prefixsort_max_string_prefix_length",
      uint32_t,
      16,
      "Maximum bytes stored in prefix-sort buffer for a string key.")

  /// Enable query tracing flag.
  VELOX_QUERY_CONFIG(
      kQueryTraceEnabled,
      queryTraceEnabled,
      "query_trace_enabled",
      bool,
      false,
      "Enable query tracing.")

  /// Base dir of a query to store tracing data.
  VELOX_QUERY_CONFIG(
      kQueryTraceDir,
      queryTraceDir,
      "query_trace_dir",
      std::string,
      "",
      "Base directory for storing query trace data.")

  /// The plan node id whose input data will be traced.
  VELOX_QUERY_CONFIG(
      kQueryTraceNodeId,
      queryTraceNodeId,
      "query_trace_node_id",
      std::string,
      "",
      "Plan node id whose input data will be traced.")

  /// The max trace bytes limit. Tracing is disabled if zero.
  VELOX_QUERY_CONFIG(
      kQueryTraceMaxBytes,
      queryTraceMaxBytes,
      "query_trace_max_bytes",
      uint64_t,
      0,
      "Maximum trace bytes. Tracing disabled if zero.")

  /// The regexp of traced task id.
  VELOX_QUERY_CONFIG(
      kQueryTraceTaskRegExp,
      queryTraceTaskRegExp,
      "query_trace_task_reg_exp",
      std::string,
      "",
      "Regexp for task ids to enable tracing on.")

  /// If true, only collect input trace without actual execution.
  VELOX_QUERY_CONFIG(
      kQueryTraceDryRun,
      queryTraceDryRun,
      "query_trace_dry_run",
      bool,
      false,
      "Collect input trace without actual execution.")

  /// Config used to create operator trace directory.
  VELOX_QUERY_CONFIG(
      kOpTraceDirectoryCreateConfig,
      opTraceDirectoryCreateConfig,
      "op_trace_directory_create_config",
      std::string,
      "",
      "Config for creating operator trace directory.")

  /// Disable optimization in expression evaluation to peel common dictionary
  /// layer from inputs.
  VELOX_QUERY_CONFIG(
      kDebugDisableExpressionWithPeeling,
      debugDisableExpressionsWithPeeling,
      "debug_disable_expression_with_peeling",
      bool,
      false,
      "Disable dictionary peeling optimization in expression evaluation.")

  /// Minimum number of rows in the selectivity vector for peeling to be
  /// applied during expression evaluation. For small batches, the overhead of
  /// peeling can outweigh the benefits.
  VELOX_QUERY_CONFIG(
      kMinRowsForPeeling,
      minRowsForPeeling,
      "expression.min_rows_for_peeling",
      int32_t,
      0,
      "Minimum number of rows to process for peeling optimization in expression evaluation to be active.");

  /// Disable optimization in expression evaluation to re-use cached results for
  /// common sub-expressions.
  VELOX_QUERY_CONFIG(
      kDebugDisableCommonSubExpressions,
      debugDisableCommonSubExpressions,
      "debug_disable_common_sub_expressions",
      bool,
      false,
      "Disable common sub-expression caching in expression evaluation.")

  /// Disable optimization in expression evaluation to re-use cached results
  /// between subsequent dictionary-encoded input batches.
  VELOX_QUERY_CONFIG(
      kDebugDisableExpressionWithMemoization,
      debugDisableExpressionsWithMemoization,
      "debug_disable_expression_with_memoization",
      bool,
      false,
      "Disable dictionary memoization in expression evaluation.")

  /// Disable optimization in expression evaluation to delay loading of lazy
  /// inputs unless required.
  VELOX_QUERY_CONFIG(
      kDebugDisableExpressionWithLazyInputs,
      debugDisableExpressionsWithLazyInputs,
      "debug_disable_expression_with_lazy_inputs",
      bool,
      false,
      "Disable lazy input loading optimization in expression evaluation.")

  /// Fix the random seed used in approx_percentile. No default (optional).
  static constexpr const char*
      kDebugAggregationApproxPercentileFixedRandomSeed =
          "debug_aggregation_approx_percentile_fixed_random_seed";

  /// Regex to match memory pools for allocation callsite tracking.
  VELOX_QUERY_CONFIG(
      kDebugMemoryPoolNameRegex,
      debugMemoryPoolNameRegex,
      "debug_memory_pool_name_regex",
      std::string,
      "",
      "Regex to match memory pools for allocation callsite tracking.")

  /// Warning threshold in bytes for debug memory pools. Uses toCapacity
  /// for parsing.
  VELOX_QUERY_CONFIG_PROPERTY(
      kDebugMemoryPoolWarnThresholdBytes,
      "debug_memory_pool_warn_threshold_bytes",
      std::string,
      "0B",
      "Warning threshold in bytes for debug memory pools.")

  /// Batch size for lambda function evaluation over arrays and maps.
  VELOX_QUERY_CONFIG(
      kDebugLambdaFunctionEvaluationBatchSize,
      debugLambdaFunctionEvaluationBatchSize,
      "debug_lambda_function_evaluation_batch_size",
      int32_t,
      10'000,
      "Batch size for lambda function evaluation over arrays and maps.")

  /// Max zoom level difference for bing_tile_children UDF.
  VELOX_QUERY_CONFIG(
      kDebugBingTileChildrenMaxZoomShift,
      debugBingTileChildrenMaxZoomShift,
      "debug_bing_tile_children_max_zoom_shift",
      uint8_t,
      7,
      "Maximum zoom level difference for bing_tile_children.")

  /// Temporary flag to control whether selective Nimble reader should be used.
  VELOX_QUERY_CONFIG(
      kSelectiveNimbleReaderEnabled,
      selectiveNimbleReaderEnabled,
      "selective_nimble_reader_enabled",
      bool,
      true,
      "Enable selective Nimble reader.")

  /// The max ratio of query memory usage to max capacity for scale writer
  /// exchange to stop scaling.
  VELOX_QUERY_CONFIG(
      kScaleWriterRebalanceMaxMemoryUsageRatio,
      scaleWriterRebalanceMaxMemoryUsageRatio,
      "scaled_writer_rebalance_max_memory_usage_ratio",
      double,
      0.7,
      "Max memory usage ratio before scale writer exchange stops scaling.")

  /// The max number of logical table partitions per writer thread.
  VELOX_QUERY_CONFIG(
      kScaleWriterMaxPartitionsPerWriter,
      scaleWriterMaxPartitionsPerWriter,
      "scaled_writer_max_partitions_per_writer",
      uint32_t,
      128,
      "Max logical table partitions per table writer thread.")

  /// Minimum data processed per partition to trigger writer scaling.
  VELOX_QUERY_CONFIG(
      kScaleWriterMinPartitionProcessedBytesRebalanceThreshold,
      scaleWriterMinPartitionProcessedBytesRebalanceThreshold,
      "scaled_writer_min_partition_processed_bytes_rebalance_threshold",
      uint64_t,
      128 << 20,
      "Minimum bytes processed per partition to trigger writer scaling.")

  /// Minimum total data processed to trigger skewed partition rebalancing.
  VELOX_QUERY_CONFIG(
      kScaleWriterMinProcessedBytesRebalanceThreshold,
      scaleWriterMinProcessedBytesRebalanceThreshold,
      "scaled_writer_min_processed_bytes_rebalance_threshold",
      uint64_t,
      256 << 20,
      "Minimum total bytes processed to trigger skewed partition rebalancing.")

  /// If true, enables the scaled table scan processing.
  VELOX_QUERY_CONFIG(
      kTableScanScaledProcessingEnabled,
      tableScanScaledProcessingEnabled,
      "table_scan_scaled_processing_enabled",
      bool,
      false,
      "Enable scaled table scan processing based on memory usage.")

  /// The query memory usage ratio for scan controller to decide if it can
  /// increase running scan threads.
  VELOX_QUERY_CONFIG(
      kTableScanScaleUpMemoryUsageRatio,
      tableScanScaleUpMemoryUsageRatio,
      "table_scan_scale_up_memory_usage_ratio",
      double,
      0.7,
      "Memory usage ratio threshold for scaling up scan threads.")

  /// Specifies the shuffle compression kind.
  VELOX_QUERY_CONFIG(
      kShuffleCompressionKind,
      shuffleCompressionKind,
      "shuffle_compression_codec",
      std::string,
      "none",
      "Compression codec for shuffle data.")

  /// Minimum serialized page size in bytes to attempt shuffle compression.
  VELOX_QUERY_CONFIG(
      kMinShuffleCompressionPageSizeBytes,
      minShuffleCompressionPageSizeBytes,
      "min_shuffle_compression_page_size_bytes",
      int32_t,
      0,
      "Minimum serialized page size in bytes to attempt shuffle compression.")

  /// When true, throw exception on duplicate map key.
  VELOX_QUERY_CONFIG(
      kThrowExceptionOnDuplicateMapKeys,
      throwExceptionOnDuplicateMapKeys,
      "throw_exception_on_duplicate_map_keys",
      bool,
      false,
      "Throw exception on duplicate map keys.")

  /// Max number of input batches to prefetch for index lookup. 0 = disabled.
  VELOX_QUERY_CONFIG(
      kIndexLookupJoinMaxPrefetchBatches,
      indexLookupJoinMaxPrefetchBatches,
      "index_lookup_join_max_prefetch_batches",
      uint32_t,
      0,
      "Maximum input batches to prefetch for index lookup. 0 disables.")

  /// If true, index join operator may split output per input batch.
  VELOX_QUERY_CONFIG(
      kIndexLookupJoinSplitOutput,
      indexLookupJoinSplitOutput,
      "index_lookup_join_split_output",
      bool,
      true,
      "Allow index join operator to split output based on batch size.")

  // Max wait time for exchange request in seconds.
  VELOX_QUERY_CONFIG(
      kRequestDataSizesMaxWaitSec,
      requestDataSizesMaxWaitSec,
      "request_data_sizes_max_wait_sec",
      int32_t,
      10,
      "Maximum wait time for exchange request in seconds.")

  /// In streaming aggregation, min output rows before producing a batch.
  VELOX_QUERY_CONFIG(
      kStreamingAggregationMinOutputBatchRows,
      streamingAggregationMinOutputBatchRows,
      "streaming_aggregation_min_output_batch_rows",
      int32_t,
      0,
      "Minimum rows to accumulate before producing streaming aggregation output. 0 uses default.")

  /// TODO: Remove after dependencies are cleaned up.
  VELOX_QUERY_CONFIG(
      kStreamingAggregationEagerFlush,
      streamingAggregationEagerFlush,
      "streaming_aggregation_eager_flush",
      bool,
      false,
      "Enable eager flush for streaming aggregation.")

  // If true, skip request data size if there is only single source.
  VELOX_QUERY_CONFIG(
      kSkipRequestDataSizeWithSingleSourceEnabled,
      singleSourceExchangeOptimizationEnabled,
      "skip_request_data_size_with_single_source_enabled",
      bool,
      false,
      "Skip request data size check with single exchange source.")

  /// If true, exchange clients defer data fetching until next() is called.
  VELOX_QUERY_CONFIG(
      kExchangeLazyFetchingEnabled,
      exchangeLazyFetchingEnabled,
      "exchange_lazy_fetching_enabled",
      bool,
      false,
      "Defer exchange data fetching until next() is called.")

  /// If true, use struct field names as JSON element names when casting row to
  /// json.
  VELOX_QUERY_CONFIG(
      kFieldNamesInJsonCastEnabled,
      isFieldNamesInJsonCastEnabled,
      "field_names_in_json_cast_enabled",
      bool,
      false,
      "Use struct field names as JSON element names in CAST to JSON.")

  /// If true, operators track stats for non-special-form expressions.
  VELOX_QUERY_CONFIG(
      kOperatorTrackExpressionStats,
      operatorTrackExpressionStats,
      "operator_track_expression_stats",
      bool,
      false,
      "Track expression stats in operators that evaluate expressions.")

  /// If true, enable operator input/output batch size stats collection.
  VELOX_QUERY_CONFIG(
      kEnableOperatorBatchSizeStats,
      enableOperatorBatchSizeStats,
      "enable_operator_batch_size_stats",
      bool,
      true,
      "Enable input/output batch size stats collection in driver execution.")

  /// If true, the unnest operator may split output per input batch.
  VELOX_QUERY_CONFIG(
      kUnnestSplitOutput,
      unnestSplitOutput,
      "unnest_split_output",
      bool,
      true,
      "Allow unnest operator to split output based on batch size.")

  /// Priority of the query in the memory pool reclaimer. Lower means higher
  /// priority.
  VELOX_QUERY_CONFIG(
      kQueryMemoryReclaimerPriority,
      queryMemoryReclaimerPriority,
      "query_memory_reclaimer_priority",
      int32_t,
      std::numeric_limits<int32_t>::max(),
      "Query priority in memory pool reclaimer. Lower means higher priority.")

  /// The max number of input splits to listen to by SplitListener.
  VELOX_QUERY_CONFIG(
      kMaxNumSplitsListenedTo,
      maxNumSplitsListenedTo,
      "max_num_splits_listened_to",
      int32_t,
      0,
      "Max input splits to listen to by SplitListener per scan node per worker.")

  /// Source of the query.
  VELOX_QUERY_CONFIG(
      kSource,
      source,
      "source",
      std::string,
      "",
      "Source of the query.")

  /// Client tags of the query.
  VELOX_QUERY_CONFIG(
      kClientTags,
      clientTags,
      "client_tags",
      std::string,
      "",
      "Client tags of the query.")

  /// Enable (reader) row size tracker. Uses enum stored as int32_t.
  VELOX_QUERY_CONFIG_PROPERTY(
      kRowSizeTrackingMode,
      "row_size_tracking_mode",
      int32_t,
      2,
      "Row size tracking mode: 0=disabled, 1=exclude delta splits, 2=enabled for all.")

  /// Maximum number of distinct values to keep when merging vector hashers in
  /// join HashBuild.
  VELOX_QUERY_CONFIG(
      kJoinBuildVectorHasherMaxNumDistinct,
      joinBuildVectorHasherMaxNumDistinct,
      "join_build_vector_hasher_max_num_distinct",
      uint32_t,
      1'000'000,
      "Max distinct values to keep when merging vector hashers in join HashBuild.")

  /// Batch size threshold for zero-copy optimization in MarkSorted operator.
  VELOX_QUERY_CONFIG(
      kMarkSortedZeroCopyThreshold,
      markSortedZeroCopyThreshold,
      "mark_sorted_zero_copy_threshold",
      int32_t,
      1000,
      "Batch size threshold for zero-copy in MarkSorted operator.")

  // --- Hand-written accessors for properties that need custom logic ---

  // Generated by VELOX_QUERY_CONFIG for simple properties above.

  uint64_t queryMaxMemoryPerNode() const {
    return config::toCapacity(
        get<std::string>(kQueryMaxMemoryPerNode, "0B"),
        config::CapacityUnit::BYTE);
  }

  vector_size_t preferredOutputBatchRows() const {
    const uint32_t batchRows = get<uint32_t>(kPreferredOutputBatchRows, 1024);
    VELOX_USER_CHECK_LE(batchRows, std::numeric_limits<vector_size_t>::max());
    return batchRows;
  }

  vector_size_t maxOutputBatchRows() const {
    const uint32_t maxBatchRows = get<uint32_t>(kMaxOutputBatchRows, 10'000);
    VELOX_USER_CHECK_LE(
        maxBatchRows, std::numeric_limits<vector_size_t>::max());
    return maxBatchRows;
  }

  vector_size_t mergeJoinOutputBatchStartSize() const {
    const uint32_t batchRows = get<uint32_t>(kMergeJoinOutputBatchStartSize, 0);
    VELOX_USER_CHECK_LE(batchRows, std::numeric_limits<vector_size_t>::max());
    return batchRows;
  }

  uint32_t localMergeMaxNumMergeSources() const {
    const auto maxNumMergeSources = get<uint32_t>(
        kLocalMergeMaxNumMergeSources, std::numeric_limits<uint32_t>::max());
    VELOX_CHECK_GT(maxNumMergeSources, 0);
    return maxNumMergeSources;
  }

  uint8_t spillNumPartitionBits() const {
    constexpr uint8_t kDefaultBits = 3;
    constexpr uint8_t kMaxBits = 3;
    return std::min(
        kMaxBits, get<uint8_t>(kSpillNumPartitionBits, kDefaultBits));
  }

  int32_t sparkPartitionId() const {
    auto id = get<int32_t>(kSparkPartitionId);
    VELOX_CHECK(id.has_value(), "Spark partition id is not set.");
    auto value = id.value();
    VELOX_CHECK_GE(value, 0, "Invalid Spark partition id.");
    return value;
  }

  uint32_t taskPartitionedWriterCount() const {
    return get<uint32_t>(kTaskPartitionedWriterCount)
        .value_or(taskWriterCount());
  }

  std::optional<uint32_t> debugAggregationApproxPercentileFixedRandomSeed()
      const {
    return get<uint32_t>(kDebugAggregationApproxPercentileFixedRandomSeed);
  }

  uint64_t debugMemoryPoolWarnThresholdBytes() const {
    return config::toCapacity(
        get<std::string>(kDebugMemoryPoolWarnThresholdBytes, "0B"),
        config::CapacityUnit::BYTE);
  }

  enum class RowSizeTrackingMode {
    DISABLED = 0,
    EXCLUDE_DELTA_SPLITS = 1,
    ENABLED_FOR_ALL = 2,
  };

  RowSizeTrackingMode rowSizeTrackingMode() const {
    return get<RowSizeTrackingMode>(
        kRowSizeTrackingMode, RowSizeTrackingMode::ENABLED_FOR_ALL);
  }

  template <typename T>
  T get(const std::string& key, const T& defaultValue) const {
    return config_->get<T>(key, defaultValue);
  }

  template <typename T>
  std::optional<T> get(const std::string& key) const {
    return config_->get<T>(key);
  }

  const std::shared_ptr<const config::IConfig>& config() const {
    return config_;
  }

  /// Test-only method to override the current query config properties.
  /// It is not thread safe.
  void testingOverrideConfigUnsafe(
      std::unordered_map<std::string, std::string>&& values);

  std::unordered_map<std::string, std::string> rawConfigsCopy() const;

 private:
  void validateConfig();

  std::shared_ptr<const config::IConfig> config_;
};

#undef VELOX_QUERY_CONFIG
#undef VELOX_QUERY_CONFIG_PROPERTY

} // namespace facebook::velox::core
