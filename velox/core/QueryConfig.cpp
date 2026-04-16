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

#include <re2/re2.h>

#include "velox/common/config/Config.h"
#include "velox/core/QueryConfig.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::core {

const std::vector<config::ConfigProperty>& QueryConfig::registeredProperties() {
  static const std::vector<config::ConfigProperty> kProperties = [] {
    std::vector<config::ConfigProperty> properties;
#define VELOX_REGISTER_QUERY_CONFIG(constName) \
  config::registerConfigProperty<QueryConfig::constName##Property>(properties)

    // Memory.
    VELOX_REGISTER_QUERY_CONFIG(kQueryMaxMemoryPerNode);

    // Session.
    VELOX_REGISTER_QUERY_CONFIG(kSessionTimezone);
    VELOX_REGISTER_QUERY_CONFIG(kSessionStartTime);
    VELOX_REGISTER_QUERY_CONFIG(kAdjustTimestampToTimezone);

    // Expression evaluation.
    VELOX_REGISTER_QUERY_CONFIG(kExprEvalSimplified);
    VELOX_REGISTER_QUERY_CONFIG(kExprEvalFlatNoNulls);
    VELOX_REGISTER_QUERY_CONFIG(kExprTrackCpuUsage);
    VELOX_REGISTER_QUERY_CONFIG(kExprTrackCpuUsageForFunctions);
    VELOX_REGISTER_QUERY_CONFIG(kExprAdaptiveCpuSampling);
    VELOX_REGISTER_QUERY_CONFIG(kExprAdaptiveCpuSamplingMaxOverheadPct);
    VELOX_REGISTER_QUERY_CONFIG(kExprDedupNonDeterministic);
    VELOX_REGISTER_QUERY_CONFIG(kExprMaxArraySizeInReduce);
    VELOX_REGISTER_QUERY_CONFIG(kExprMaxCompiledRegexes);

    // Operator.
    VELOX_REGISTER_QUERY_CONFIG(kOperatorTrackCpuUsage);

    // Cast.
    VELOX_REGISTER_QUERY_CONFIG(kLegacyCast);
    VELOX_REGISTER_QUERY_CONFIG(kCastMatchStructByName);

    // Local exchange.
    VELOX_REGISTER_QUERY_CONFIG(kMaxLocalExchangeBufferSize);
    VELOX_REGISTER_QUERY_CONFIG(kMaxLocalExchangePartitionCount);
    VELOX_REGISTER_QUERY_CONFIG(
        kMinLocalExchangePartitionCountToUsePartitionBuffer);
    VELOX_REGISTER_QUERY_CONFIG(kMaxLocalExchangePartitionBufferSize);
    VELOX_REGISTER_QUERY_CONFIG(kLocalExchangePartitionBufferPreserveEncoding);
    VELOX_REGISTER_QUERY_CONFIG(kLocalMergeSourceQueueSize);

    // Exchange.
    VELOX_REGISTER_QUERY_CONFIG(kMaxExchangeBufferSize);
    VELOX_REGISTER_QUERY_CONFIG(kMaxMergeExchangeBufferSize);
    VELOX_REGISTER_QUERY_CONFIG(kMinExchangeOutputBatchBytes);

    // Aggregation.
    VELOX_REGISTER_QUERY_CONFIG(kMaxPartialAggregationMemory);
    VELOX_REGISTER_QUERY_CONFIG(kMaxExtendedPartialAggregationMemory);
    VELOX_REGISTER_QUERY_CONFIG(kAbandonPartialAggregationMinRows);
    VELOX_REGISTER_QUERY_CONFIG(kAbandonPartialAggregationMinPct);
    VELOX_REGISTER_QUERY_CONFIG(kAggregationCompactionBytesThreshold);
    VELOX_REGISTER_QUERY_CONFIG(kAggregationCompactionUnusedMemoryRatio);
    VELOX_REGISTER_QUERY_CONFIG(kAggregationMemoryCompactionReclaimEnabled);

    // TopN row number.
    VELOX_REGISTER_QUERY_CONFIG(kAbandonPartialTopNRowNumberMinRows);
    VELOX_REGISTER_QUERY_CONFIG(kAbandonPartialTopNRowNumberMinPct);

    // Hash build dedup.
    VELOX_REGISTER_QUERY_CONFIG(kAbandonDedupHashMapMinRows);
    VELOX_REGISTER_QUERY_CONFIG(kAbandonDedupHashMapMinPct);

    // Miscellaneous.
    VELOX_REGISTER_QUERY_CONFIG(kMaxElementsSizeInRepeatAndSequence);

    // Partitioned output.
    VELOX_REGISTER_QUERY_CONFIG(kPartitionedOutputEagerFlush);
    VELOX_REGISTER_QUERY_CONFIG(kMaxPartitionedOutputBufferSize);
    VELOX_REGISTER_QUERY_CONFIG(kMaxOutputBufferSize);

    // Output batch.
    VELOX_REGISTER_QUERY_CONFIG(kPreferredOutputBatchBytes);
    VELOX_REGISTER_QUERY_CONFIG(kPreferredOutputBatchRows);
    VELOX_REGISTER_QUERY_CONFIG(kMaxOutputBatchRows);
    VELOX_REGISTER_QUERY_CONFIG(kMergeJoinOutputBatchStartSize);

    // Table scan.
    VELOX_REGISTER_QUERY_CONFIG(kTableScanGetOutputTimeLimitMs);
    VELOX_REGISTER_QUERY_CONFIG(kTableScanOutputBatchRowsOverride);

    // Hash table.
    VELOX_REGISTER_QUERY_CONFIG(kHashAdaptivityEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kAdaptiveFilterReorderingEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kParallelOutputJoinBuildRowsEnabled);

    // Spill.
    VELOX_REGISTER_QUERY_CONFIG(kSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kAggregationSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kJoinSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kMixedGroupedModeHashJoinSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kOrderBySpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kWindowSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kWindowSpillMinReadBatchRows);
    VELOX_REGISTER_QUERY_CONFIG(kWriterSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kRowNumberSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kMarkDistinctSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kTopNRowNumberSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kLocalMergeSpillEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kMaxSpillRunRows);
    VELOX_REGISTER_QUERY_CONFIG(kMaxSpillBytes);
    VELOX_REGISTER_QUERY_CONFIG(kMaxSpillLevel);
    VELOX_REGISTER_QUERY_CONFIG(kMaxSpillFileSize);
    VELOX_REGISTER_QUERY_CONFIG(kSpillCompressionKind);
    VELOX_REGISTER_QUERY_CONFIG(kSpillNumMaxMergeFiles);
    VELOX_REGISTER_QUERY_CONFIG(kSpillPrefixSortEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kSpillWriteBufferSize);
    VELOX_REGISTER_QUERY_CONFIG(kSpillReadBufferSize);
    VELOX_REGISTER_QUERY_CONFIG(kSpillFileCreateConfig);
    VELOX_REGISTER_QUERY_CONFIG(kAggregationSpillFileCreateConfig);
    VELOX_REGISTER_QUERY_CONFIG(kHashJoinSpillFileCreateConfig);
    VELOX_REGISTER_QUERY_CONFIG(kRowNumberSpillFileCreateConfig);
    VELOX_REGISTER_QUERY_CONFIG(kSpillStartPartitionBit);
    VELOX_REGISTER_QUERY_CONFIG(kSpillNumPartitionBits);
    VELOX_REGISTER_QUERY_CONFIG(kMinSpillableReservationPct);
    VELOX_REGISTER_QUERY_CONFIG(kSpillableReservationGrowthPct);

    // Writer.
    VELOX_REGISTER_QUERY_CONFIG(kWriterFlushThresholdBytes);

    // Presto-specific.
    VELOX_REGISTER_QUERY_CONFIG(kPrestoArrayAggIgnoreNulls);

    // Spark-specific.
    VELOX_REGISTER_QUERY_CONFIG(kSparkAnsiEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kSparkBloomFilterExpectedNumItems);
    VELOX_REGISTER_QUERY_CONFIG(kSparkBloomFilterNumBits);
    VELOX_REGISTER_QUERY_CONFIG(kSparkBloomFilterMaxNumBits);
    VELOX_REGISTER_QUERY_CONFIG(kSparkBloomFilterMaxNumItems);
    VELOX_REGISTER_QUERY_CONFIG(kSparkLegacyDateFormatter);
    VELOX_REGISTER_QUERY_CONFIG(kSparkLegacyStatisticalAggregate);
    VELOX_REGISTER_QUERY_CONFIG(kSparkJsonIgnoreNullFields);
    VELOX_REGISTER_QUERY_CONFIG(kSparkCollectListIgnoreNulls);

    // Task writer.
    VELOX_REGISTER_QUERY_CONFIG(kTaskWriterCount);

    // Hash probe.
    VELOX_REGISTER_QUERY_CONFIG(kHashProbeFinishEarlyOnEmptyBuild);
    VELOX_REGISTER_QUERY_CONFIG(kHashProbeDynamicFilterPushdownEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kHashProbeStringDynamicFilterPushdownEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kHashProbeBloomFilterPushdownMaxSize);
    VELOX_REGISTER_QUERY_CONFIG(kMinTableRowsForParallelJoinBuild);

    // Debug and validation.
    VELOX_REGISTER_QUERY_CONFIG(kValidateOutputFromOperators);
    VELOX_REGISTER_QUERY_CONFIG(kEnableExpressionEvaluationCache);
    VELOX_REGISTER_QUERY_CONFIG(kMaxSharedSubexprResultsCached);

    // Split preload.
    VELOX_REGISTER_QUERY_CONFIG(kMaxSplitPreloadPerDriver);

    // Driver.
    VELOX_REGISTER_QUERY_CONFIG(kDriverCpuTimeSliceLimitMs);

    // Window.
    VELOX_REGISTER_QUERY_CONFIG(kWindowNumSubPartitions);

    // Prefix sort.
    VELOX_REGISTER_QUERY_CONFIG(kPrefixSortNormalizedKeyMaxBytes);
    VELOX_REGISTER_QUERY_CONFIG(kPrefixSortMinRows);
    VELOX_REGISTER_QUERY_CONFIG(kPrefixSortMaxStringPrefixLength);

    // Query trace.
    VELOX_REGISTER_QUERY_CONFIG(kQueryTraceEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kQueryTraceDir);
    VELOX_REGISTER_QUERY_CONFIG(kQueryTraceNodeId);
    VELOX_REGISTER_QUERY_CONFIG(kQueryTraceMaxBytes);
    VELOX_REGISTER_QUERY_CONFIG(kQueryTraceTaskRegExp);
    VELOX_REGISTER_QUERY_CONFIG(kQueryTraceDryRun);
    VELOX_REGISTER_QUERY_CONFIG(kOpTraceDirectoryCreateConfig);

    // Debug expression.
    VELOX_REGISTER_QUERY_CONFIG(kDebugDisableExpressionWithPeeling);
    VELOX_REGISTER_QUERY_CONFIG(kDebugDisableCommonSubExpressions);
    VELOX_REGISTER_QUERY_CONFIG(kDebugDisableExpressionWithMemoization);
    VELOX_REGISTER_QUERY_CONFIG(kDebugDisableExpressionWithLazyInputs);
    VELOX_REGISTER_QUERY_CONFIG(kDebugMemoryPoolNameRegex);
    VELOX_REGISTER_QUERY_CONFIG(kDebugMemoryPoolWarnThresholdBytes);
    VELOX_REGISTER_QUERY_CONFIG(kDebugLambdaFunctionEvaluationBatchSize);
    VELOX_REGISTER_QUERY_CONFIG(kDebugBingTileChildrenMaxZoomShift);

    // Nimble.
    VELOX_REGISTER_QUERY_CONFIG(kSelectiveNimbleReaderEnabled);

    // Scale writer.
    VELOX_REGISTER_QUERY_CONFIG(kScaleWriterRebalanceMaxMemoryUsageRatio);
    VELOX_REGISTER_QUERY_CONFIG(kScaleWriterMaxPartitionsPerWriter);
    VELOX_REGISTER_QUERY_CONFIG(
        kScaleWriterMinPartitionProcessedBytesRebalanceThreshold);
    VELOX_REGISTER_QUERY_CONFIG(
        kScaleWriterMinProcessedBytesRebalanceThreshold);

    // Table scan scaling.
    VELOX_REGISTER_QUERY_CONFIG(kTableScanScaledProcessingEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kTableScanScaleUpMemoryUsageRatio);

    // Shuffle.
    VELOX_REGISTER_QUERY_CONFIG(kShuffleCompressionKind);

    // Map.
    VELOX_REGISTER_QUERY_CONFIG(kThrowExceptionOnDuplicateMapKeys);

    // Index lookup join.
    VELOX_REGISTER_QUERY_CONFIG(kIndexLookupJoinMaxPrefetchBatches);
    VELOX_REGISTER_QUERY_CONFIG(kIndexLookupJoinSplitOutput);

    // Exchange request.
    VELOX_REGISTER_QUERY_CONFIG(kRequestDataSizesMaxWaitSec);

    // Streaming aggregation.
    VELOX_REGISTER_QUERY_CONFIG(kStreamingAggregationMinOutputBatchRows);
    VELOX_REGISTER_QUERY_CONFIG(kStreamingAggregationEagerFlush);

    // Exchange optimization.
    VELOX_REGISTER_QUERY_CONFIG(kSkipRequestDataSizeWithSingleSourceEnabled);
    VELOX_REGISTER_QUERY_CONFIG(kExchangeLazyFetchingEnabled);

    // JSON cast.
    VELOX_REGISTER_QUERY_CONFIG(kFieldNamesInJsonCastEnabled);

    // Operator stats.
    VELOX_REGISTER_QUERY_CONFIG(kOperatorTrackExpressionStats);
    VELOX_REGISTER_QUERY_CONFIG(kEnableOperatorBatchSizeStats);

    // Unnest.
    VELOX_REGISTER_QUERY_CONFIG(kUnnestSplitOutput);

    // Memory reclaimer.
    VELOX_REGISTER_QUERY_CONFIG(kQueryMemoryReclaimerPriority);

    // Splits.
    VELOX_REGISTER_QUERY_CONFIG(kMaxNumSplitsListenedTo);

    // Source and tags.
    VELOX_REGISTER_QUERY_CONFIG(kSource);
    VELOX_REGISTER_QUERY_CONFIG(kClientTags);

    // Row size tracking.
    VELOX_REGISTER_QUERY_CONFIG(kRowSizeTrackingMode);

    // Join build.
    VELOX_REGISTER_QUERY_CONFIG(kJoinBuildVectorHasherMaxNumDistinct);

    // Mark sorted.
    VELOX_REGISTER_QUERY_CONFIG(kMarkSortedZeroCopyThreshold);

#undef VELOX_REGISTER_QUERY_CONFIG

    return properties;
  }();
  return kProperties;
}

QueryConfig::QueryConfig(std::unordered_map<std::string, std::string> values)
    : QueryConfig{
          ConfigTag{},
          std::make_shared<config::ConfigBase>(std::move(values))} {}

QueryConfig::QueryConfig(
    ConfigTag /*tag*/,
    std::shared_ptr<const config::IConfig> config)
    : config_{std::move(config)} {
  validateConfig();
}

void QueryConfig::validateConfig() {
  // Validate if timezone name can be recognized.
  if (auto tz = config_->get<std::string>(QueryConfig::kSessionTimezone)) {
    VELOX_USER_CHECK(
        tz::getTimeZoneID(*tz, false) != -1,
        "session '{}' set with invalid value '{}'",
        QueryConfig::kSessionTimezone,
        *tz);
  }
}

void QueryConfig::testingOverrideConfigUnsafe(
    std::unordered_map<std::string, std::string>&& values) {
  config_ = std::make_unique<config::ConfigBase>(std::move(values));
}

std::unordered_map<std::string, std::string> QueryConfig::rawConfigsCopy()
    const {
  return config_->rawConfigsCopy();
}

} // namespace facebook::velox::core
