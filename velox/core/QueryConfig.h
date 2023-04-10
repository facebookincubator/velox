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

#include "velox/core/Context.h"

namespace facebook::velox::core {

/// A simple wrapper around BaseConfigManager. Defines constants for query
/// config properties and accessor methods.
class QueryConfig {
 public:
  explicit QueryConfig(BaseConfigManager* configManager)
      : configManager_{configManager} {}

  static constexpr const char* kCodegenEnabled = "driver.codegen.enabled";

  static constexpr const char* kCodegenConfigurationFilePath =
      "driver.codegen.configuration_file_path";

  static constexpr const char* kCodegenLazyLoading =
      "driver.codegen.lazy_loading";

  // User provided session timezone. Stores a string with the actual timezone
  // name, e.g: "America/Los_Angeles".
  static constexpr const char* kSessionTimezone = "driver.session.timezone";

  // If true, timezone-less timestamp conversions (e.g. string to timestamp,
  // when the string does not specify a timezone) will be adjusted to the user
  // provided session timezone (if any).
  //
  // For instance:
  //
  //  if this option is true and user supplied "America/Los_Angeles",
  //  "1970-01-01" will be converted to -28800 instead of 0.
  //
  // False by default.
  static constexpr const char* kAdjustTimestampToTimezone =
      "driver.session.adjust_timestamp_to_timezone";

  // Whether to use the simplified expression evaluation path. False by default.
  static constexpr const char* kExprEvalSimplified =
      "expression.eval_simplified";

  // Whether to track CPU usage for individual expressions (supported by call
  // and cast expressions). False by default. Can be expensive when processing
  // small batches, e.g. < 10K rows.
  static constexpr const char* kExprTrackCpuUsage =
      "expression.track_cpu_usage";

  // Whether to track CPU usage for stages of individual operators. True by
  // default. Can be expensive when processing small batches, e.g. < 10K rows.
  static constexpr const char* kOperatorTrackCpuUsage =
      "driver.track_operator_cpu_usage";

  // Flags used to configure the CAST operator:

  // This flag makes the Row conversion to by applied
  // in a way that the casting row field are matched by
  // name instead of position
  static constexpr const char* kCastMatchStructByName =
      "driver.cast.match_struct_by_name";

  // This flags forces the cast from float/double to integer to be performed by
  // truncating the decimal part instead of rounding.
  static constexpr const char* kCastIntByTruncate =
      "driver.cast.int_by_truncate";

  static constexpr const char* kMaxLocalExchangeBufferSize =
      "max_local_exchange_buffer_size";

  static constexpr const char* kMaxPartialAggregationMemory =
      "max_partial_aggregation_memory";

  static constexpr const char* kMaxExtendedPartialAggregationMemory =
      "max_extended_partial_aggregation_memory";

  static constexpr const char* kMaxPartitionedOutputBufferSize =
      "driver.max-page-partitioning-buffer-size";

  /// Preffered number of rows to be returned by operators from
  /// Operator::getOutput.
  static constexpr const char* kPreferredOutputBatchSize =
      "preferred_output_batch_size";

  static constexpr const char* kHashAdaptivityEnabled =
      "driver.hash_adaptivity_enabled";

  static constexpr const char* kAdaptiveFilterReorderingEnabled =
      "driver.adaptive_filter_reordering_enabled";

  static constexpr const char* kCreateEmptyFiles = "driver.create_empty_files";

  /// Global enable spilling flag.
  static constexpr const char* kSpillEnabled = "spill_enabled";

  /// Spill path. "/tmp" by default.
  static constexpr const char* kSpillPath = "spiller-spill-path";

  /// Aggregation spilling flag, only applies if "spill_enabled" flag is set.
  static constexpr const char* kAggregationSpillEnabled =
      "aggregation_spill_enabled";

  /// Join spilling flag, only applies if "spill_enabled" flag is set.
  static constexpr const char* kJoinSpillEnabled = "join_spill_enabled";

  /// OrderBy spilling flag, only applies if "spill_enabled" flag is set.
  static constexpr const char* kOrderBySpillEnabled = "order_by_spill_enabled";

  /// The max memory that a final aggregation can use before spilling. If it 0,
  /// then there is no limit.
  static constexpr const char* kAggregationSpillMemoryThreshold =
      "aggregation_spill_memory_threshold";

  /// The max memory that a hash join can use before spilling. If it 0, then
  /// there is no limit.
  static constexpr const char* kJoinSpillMemoryThreshold =
      "join_spill_memory_threshold";

  /// The max memory that an order by can use before spilling. If it 0, then
  /// there is no limit.
  static constexpr const char* kOrderBySpillMemoryThreshold =
      "order_by_spill_memory_threshold";

  static constexpr const char* kTestingSpillPct = "testing.spill-pct";

  /// The max allowed spilling level with zero being the initial spilling level.
  /// This only applies for hash build spilling which might trigger recursive
  /// spilling when the build table is too big. If it is set to -1, then there
  /// is no limit and then some extreme large query might run out of spilling
  /// partition bits (see kSpillPartitionBits) at the end. The max spill level
  /// is used in production to prevent some bad user queries from using too much
  /// io and cpu resources.
  static constexpr const char* kMaxSpillLevel = "max-spill-level";

  /// The max allowed spill file size. If it is zero, then there is no limit.
  static constexpr const char* kMaxSpillFileSize = "max-spill-file-size";

  /// The min spill run size limit used to select partitions for spilling. The
  /// spiller tries to spill a previously spilled partitions if its data size
  /// exceeds this limit, otherwise it spills the partition with most data.
  /// If the limit is zero, then the spiller always spill a previously spilled
  /// partition if it has any data. This is to avoid spill from a partition with
  /// a small amount of data which might result in generating too many small
  /// spilled files.
  static constexpr const char* kMinSpillRunSize = "min-spill-run-size";

  static constexpr const char* kSpillStartPartitionBit =
      "spiller-start-partition-bit";

  static constexpr const char* kSpillPartitionBits = "spiller-partition-bits";

  static constexpr const char* kSpillableReservationGrowthPct =
      "spillable-reservation-growth-pct";

  uint64_t maxPartialAggregationMemoryUsage() const {
    static constexpr uint64_t kDefault = 1L << 24;
    return get<uint64_t>(kMaxPartialAggregationMemory, kDefault);
  }

  uint64_t maxExtendedPartialAggregationMemoryUsage() const {
    static constexpr uint64_t kDefault = 1L << 26;
    return get<uint64_t>(kMaxExtendedPartialAggregationMemory, kDefault);
  }

  uint64_t aggregationSpillMemoryThreshold() const {
    static constexpr uint64_t kDefault = 0;
    return get<uint64_t>(kAggregationSpillMemoryThreshold, kDefault);
  }

  uint64_t joinSpillMemoryThreshold() const {
    static constexpr uint64_t kDefault = 0;
    return get<uint64_t>(kJoinSpillMemoryThreshold, kDefault);
  }

  uint64_t orderBySpillMemoryThreshold() const {
    static constexpr uint64_t kDefault = 0;
    return get<uint64_t>(kOrderBySpillMemoryThreshold, kDefault);
  }

  // Returns the target size for a Task's buffered output. The
  // producer Drivers are blocked when the buffered size exceeds
  // this. The Drivers are resumed when the buffered size goes below
  // PartitionedOutputBufferManager::kContinuePct % of this.
  uint64_t maxPartitionedOutputBufferSize() const {
    static constexpr uint64_t kDefault = 32UL << 20;
    return get<uint64_t>(kMaxPartitionedOutputBufferSize, kDefault);
  }

  uint64_t maxLocalExchangeBufferSize() const {
    static constexpr uint64_t kDefault = 32UL << 20;
    return get<uint64_t>(kMaxLocalExchangeBufferSize, kDefault);
  }

  uint32_t preferredOutputBatchSize() const {
    return get<uint32_t>(kPreferredOutputBatchSize, 1024);
  }

  bool hashAdaptivityEnabled() const {
    return get<bool>(kHashAdaptivityEnabled, true);
  }

  uint32_t writeStrideSize() const {
    static constexpr uint32_t kDefault = 100'000;
    return kDefault;
  }

  bool flushPerBatch() const {
    static constexpr bool kDefault = true;
    return kDefault;
  }

  bool adaptiveFilterReorderingEnabled() const {
    return get<bool>(kAdaptiveFilterReorderingEnabled, true);
  }

  bool isMatchStructByName() const {
    return get<bool>(kCastMatchStructByName, false);
  }

  bool isCastIntByTruncate() const {
    return get<bool>(kCastIntByTruncate, false);
  }

  bool codegenEnabled() const {
    return get<bool>(kCodegenEnabled, false);
  }

  std::string codegenConfigurationFilePath() const {
    return get<std::string>(kCodegenConfigurationFilePath, "");
  }

  bool codegenLazyLoading() const {
    return get<bool>(kCodegenLazyLoading, true);
  }

  bool createEmptyFiles() const {
    return get<bool>(kCreateEmptyFiles, false);
  }

  bool adjustTimestampToTimezone() const {
    return get<bool>(kAdjustTimestampToTimezone, false);
  }

  std::string sessionTimezone() const {
    return get<std::string>(kSessionTimezone, "");
  }

  bool exprEvalSimplified() const {
    return get<bool>(kExprEvalSimplified, false);
  }

  /// Returns true if spilling is enabled.
  bool spillEnabled() const {
    return get<bool>(kSpillEnabled, false);
  }

  /// Returns the path for writing spill files. The path should be
  /// interpretable by filesystems::getFileSystem and may refer to any writable
  /// location. Actual file names are composed by appending '/' and a filename
  /// composed of Task id and serial numbers. The files are automatically
  /// deleted when no longer needed. Files may be left behind after crashes but
  /// are identifiable based on the Task id in the name.
  /// TODO(spershin): This method and kSpillPath will be removed.
  std::optional<std::string> spillPath() const {
    return get<std::string>(kSpillPath);
  }

  /// Returns 'is aggregation spilling enabled' flag. Must also check the
  /// spillEnabled()!
  bool aggregationSpillEnabled() const {
    return get<bool>(kAggregationSpillEnabled, true);
  }

  /// Returns 'is join spilling enabled' flag. Must also check the
  /// spillEnabled()!
  bool joinSpillEnabled() const {
    return get<bool>(kJoinSpillEnabled, true);
  }

  /// Returns 'is orderby spilling enabled' flag. Must also check the
  /// spillEnabled()!
  bool orderBySpillEnabled() const {
    return get<bool>(kOrderBySpillEnabled, true);
  }

  // Returns a percentage of aggregation or join input batches that
  // will be forced to spill for testing. 0 means no extra spilling.
  int32_t testingSpillPct() const {
    return get<int32_t>(kTestingSpillPct, 0);
  }

  int32_t maxSpillLevel() const {
    return get<int32_t>(kMaxSpillLevel, 4);
  }

  /// Returns the start partition bit which is used with 'kSpillPartitionBits'
  /// together to calculate the spilling partition number.
  uint8_t spillStartPartitionBit() const {
    constexpr uint8_t kDefaultStartBit = 29;
    return get<uint8_t>(kSpillStartPartitionBit, kDefaultStartBit);
  }

  /// Returns the number of bits used to calculate the spilling partition
  /// number. The number of spilling partitions will be power of two.
  ///
  /// NOTE: as for now, we only support up to 8-way spill partitioning.
  int32_t spillPartitionBits() const {
    constexpr int32_t kDefaultBits = 2;
    constexpr int32_t kMaxBits = 3;
    return std::min(kMaxBits, get<int32_t>(kSpillPartitionBits, kDefaultBits));
  }

  uint64_t maxSpillFileSize() const {
    constexpr uint64_t kDefaultMaxFileSize = 0;
    return get<uint64_t>(kMaxSpillFileSize, kDefaultMaxFileSize);
  }

  uint64_t minSpillRunSize() const {
    constexpr uint64_t kDefaultMinSpillRunSize = 256 << 20; // 256MB.
    return get<uint64_t>(kMinSpillRunSize, kDefaultMinSpillRunSize);
  }

  /// Returns the spillable memory reservation growth percentage of the previous
  /// memory reservation size. 25 means exponential growth along a series of
  /// integer powers of 5/4. The reservation grows by this much until it no
  /// longer can, after which it starts spilling.
  int32_t spillableReservationGrowthPct() const {
    constexpr int32_t kDefaultPct = 25;
    return get<double>(kSpillableReservationGrowthPct, kDefaultPct);
  }

  bool exprTrackCpuUsage() const {
    return get<bool>(kExprTrackCpuUsage, false);
  }

  bool operatorTrackCpuUsage() const {
    return get<bool>(kOperatorTrackCpuUsage, true);
  }

  template <typename T>
  T get(const std::string& key, const T& defaultValue) const {
    return configManager_->get<T>(key, defaultValue);
  }
  template <typename T>
  std::optional<T> get(const std::string& key) const {
    return std::optional<T>(configManager_->get<T>(key));
  }

 private:
  BaseConfigManager* FOLLY_NONNULL configManager_;
};
} // namespace facebook::velox::core
