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

#include <folly/Executor.h>
#include <folly/container/F14Map.h>
#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/IndexReader.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::connector::hive {

class HiveConfig;

/// Provides index lookup for Hive-metastore-backed tables.
///
/// Owns format-agnostic orchestration: remaining filter evaluation, output
/// projection, and partition routing. Delegates format-specific reads to
/// either the built-in FileIndexReader (for Nimble) or external
/// SplitIndexReader implementations registered via IndexReaderFactoryRegistry.
/// Handles partitioned tables by routing probe rows to the correct file based
/// on partition column values from split metadata.
class HiveIndexSource : public IndexSource,
                        public std::enable_shared_from_this<HiveIndexSource> {
 public:
  HiveIndexSource(
      const RowTypePtr& requestType,
      const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions,
      const RowTypePtr& outputType,
      HiveTableHandlePtr tableHandle,
      const ColumnHandleMap& columnHandles,
      FileHandleFactory* fileHandleFactory,
      ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<HiveConfig>& hiveConfig,
      folly::Executor* executor);

  ~HiveIndexSource() override = default;

  void addSplits(std::vector<std::shared_ptr<ConnectorSplit>> splits) override;

  std::shared_ptr<ResultIterator> lookup(const Request& request) override;

  std::unordered_map<std::string, RuntimeMetric> runtimeStats() override;

 private:
  friend class HiveLookupIterator;

  memory::MemoryPool* pool() const {
    return pool_;
  }

  const RowTypePtr& outputType() const {
    return outputType_;
  }

  const RowVectorPtr& getEmptyOutput() {
    if (!emptyOutput_) {
      emptyOutput_ = RowVector::createEmpty(outputType_, pool_);
    }
    return emptyOutput_;
  }

  // Evaluates remainingFilter on the specified vector. Returns number of rows
  // passed. Populates filterEvalCtx_.selectedIndices and selectedBits if only
  // some rows passed the filter. If none or all rows passed
  // filterEvalCtx_.selectedIndices and selectedBits are not updated.
  vector_size_t evaluateRemainingFilter(RowVectorPtr& rowVector);

  // Applies non-index equi-join conditions as post-read equality filters.
  // Compares request (probe) values against reader output column values
  // for each non-index join condition. Returns the number of rows that pass
  // all conditions. Populates 'passingIndices' with the indices of rows that
  // pass when not all rows pass.
  vector_size_t applyNonIndexConditions(
      const RowVectorPtr& request,
      const RowVectorPtr& output,
      const BufferPtr& inputHits,
      BufferPtr& passingIndices);

  // Projects output from reader output to the final output type. Wraps child
  // vectors with remaining filter indices if provided. Partition
  // columns are emitted by the reader directly via setConstantValue() on
  // the scan spec, so they are projected like any other column here.
  RowVectorPtr projectOutput(
      vector_size_t numRows,
      const BufferPtr& remainingIndices,
      const RowVectorPtr& rowVector);

  // Builds a fresh scan spec from the configured reader output type,
  // projected subfields, filters, and data columns. When partitionValues
  // is provided, sets partition column constants on the spec via
  // setPartitionValue().
  std::shared_ptr<common::ScanSpec> buildScanSpec(
      const std::unordered_map<std::string, std::optional<std::string>>*
          partitionValues = nullptr);

  // Sets a constant partition value on the scanSpec for a partition column.
  void setPartitionValue(
      common::ScanSpec* spec,
      const std::string& partitionKey,
      const std::optional<std::string>& value) const;

  void init(
      const ColumnHandleMap& assignments,
      const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions);

  // Initializes all join conditions into three buckets:
  // 1. Index conditions stored in indexLookupConditions_, pushed to the reader.
  // 2. Partition conditions stored in partitionIndexConditions_.
  // 3. Non-index conditions stored in nonIndexConditions_ for post-read
  //    equality filtering. Their columns are added to
  //    readColumnNames/readColumnTypes if not already present.
  void initConditions(
      const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions,
      const ColumnHandleMap& assignments,
      const folly::F14FastMap<std::string_view, const HiveColumnHandle*>&
          columnHandles,
      std::vector<std::string>& readColumnNames,
      std::vector<TypePtr>& readColumnTypes);

  // Processes index columns in order, converting filters to index lookup
  // conditions where possible. Populates indexLookupConditions_ and returns
  // the set of column names consumed as index conditions.
  folly::F14FastSet<std::string> initIndexConditions(
      const folly::F14FastMap<std::string, core::IndexLookupConditionPtr>&
          conditionMap);

  // Processes remaining non-index conditions from conditionMap, splitting
  // them into partitionIndexConditions_ or nonIndexConditions_. Adds
  // required columns to readColumnNames/readColumnTypes.
  void initNonIndexConditions(
      const folly::F14FastMap<std::string, core::IndexLookupConditionPtr>&
          conditionMap,
      const folly::F14FastSet<std::string>& indexConditionColumns,
      const folly::F14FastMap<std::string_view, const HiveColumnHandle*>&
          columnHandles,
      std::vector<std::string>& readColumnNames,
      std::vector<TypePtr>& readColumnTypes);

  // Initializes the remaining filter:
  // - Compiles the remaining filter expression.
  // - Identifies columns to eagerly materialize.
  // - Adds columns referenced by remaining filter but not projected.
  // - Extracts and processes subfields from the remaining filter.
  void initRemainingFilter(
      std::vector<std::string>& readColumnNames,
      std::vector<TypePtr>& readColumnTypes);

  // Creates a lookup iterator over the given readers: a single
  // HiveLookupIterator if there's only one reader, otherwise a
  // UnionResultIterator that k-way-merges per-reader results.
  std::shared_ptr<ResultIterator> createLookupIterator(
      const Request& request,
      const std::vector<SplitIndexReader*>& readers,
      const SplitIndexReader::Options& options);

  // Creates a SplitIndexReader using a registered IndexReaderFactory.
  void createCustomIndexReader(
      const IndexReaderFactory& factory,
      std::shared_ptr<const HiveConnectorSplit> split);

  // Creates a FileIndexReader for a single split.
  void createFileIndexReader(
      std::shared_ptr<const HiveConnectorSplit> split,
      const std::shared_ptr<common::ScanSpec>& scanSpec);

  // Creates readers from splits into readers_. Uses a registered
  // IndexReaderFactory when available, otherwise falls back to
  // FileIndexReader.
  void createReadersFromSplits(
      std::vector<std::shared_ptr<const HiveConnectorSplit>> splits,
      const std::shared_ptr<common::ScanSpec>& scanSpec);

  // Groups splits by their routing-column partition values, then builds
  // one PartitionGroup per distinct partition with its own scan
  // spec (partition columns set as constants) and routing constants for
  // runtime probe matching. Populates partitionGroups_. Called from
  // addSplits() when partitionIndexConditions_ is non-empty.
  void buildPartitionGroups(
      std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits);

  // Groups splits by their partition column values.
  folly::F14FastMap<
      std::string,
      std::vector<std::shared_ptr<const HiveConnectorSplit>>>
  groupSplitsByPartitions(
      std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits) const;

  // Encodes a split's partition values into an opaque grouping key.
  std::string makePartitionKey(const HiveConnectorSplit& split) const;

  // Splits sharing the same partition values, grouped during addSplits().
  // One group per distinct set of partition column values. Each group owns
  // its own scan spec with partition-column constants set, so the underlying
  // reader emits partition values directly.
  struct PartitionGroup {
    // Raw pointers into readers_; ownership stays with readers_.
    std::vector<SplitIndexReader*> readers;
    // Typed constants aligned with partitionIndexConditions_, used by
    // findPartitionGroup() to match against probe row values.
    std::vector<VectorPtr> partitionValues;
  };

  // Finds the partition group whose partition values match the given probe
  // row's partition column values. Returns nullptr if no group matches.
  PartitionGroup* findPartitionGroup(
      const RowVectorPtr& probeInput,
      vector_size_t row);

  // Creates a partitioned lookup iterator when probe rows target different
  // partition groups. Takes a pre-built row-to-group mapping, dispatches
  // each sub-batch to the matching group's readers, and merges results.
  std::shared_ptr<ResultIterator> createPartitionLookupIterator(
      const Request& request,
      folly::F14FastMap<PartitionGroup*, std::vector<vector_size_t>>
          partitionRowMap,
      const SplitIndexReader::Options& options);

  // Per-iteration timing breakdown for index lookups.
  struct IterationStats {
    uint64_t setupWallNs{0};
    uint64_t setupCpuNs{0};
    uint64_t readWallNs{0};
    uint64_t readCpuNs{0};
    uint64_t outputWallNs{0};
    uint64_t outputCpuNs{0};
    uint64_t filterWallNs{0};
    uint64_t filterCpuNs{0};

    static constexpr std::string_view kConnectorLookupWallNanos{
        "connectorLookupWallNanos"};
    static constexpr std::string_view kConnectorResultPrepareCpuNanos{
        "connectorResultPrepareCpuNanos"};
    static constexpr std::string_view kIndexSetupWallNanos{
        "connectorIndexSetupWallNanos"};
    static constexpr std::string_view kIndexSetupCpuNanos{
        "connectorIndexSetupCpuNanos"};
    static constexpr std::string_view kIndexReadWallNanos{
        "connectorIndexReadWallNanos"};
    static constexpr std::string_view kIndexReadCpuNanos{
        "connectorIndexReadCpuNanos"};
    static constexpr std::string_view kPostFilterWallNanos{
        "connectorPostFilterWallNanos"};
    static constexpr std::string_view kPostFilterCpuNanos{
        "connectorPostFilterCpuNanos"};
  };

  // Records per-iteration timing breakdown using addOperatorRuntimeStats to
  // preserve per-call count/min/max granularity.
  void recordIterationStats(const IterationStats& iterationStats);

  FileHandleFactory* const fileHandleFactory_;
  ConnectorQueryCtx* const connectorQueryCtx_;
  const std::shared_ptr<HiveConfig> hiveConfig_;
  memory::MemoryPool* const pool_;
  core::ExpressionEvaluator* const expressionEvaluator_;
  const uint32_t maxRowsPerIndexRequest_;

  const HiveTableHandlePtr tableHandle_;
  const RowTypePtr requestType_;
  const RowTypePtr outputType_;
  folly::Executor* const executor_;

  // All index lookup conditions including equal lookup keys converted to
  // EqualIndexLookupConditions and original non-filter index lookup conditions.
  // This is passed to FileIndexReader.
  std::vector<core::IndexLookupConditionPtr> indexLookupConditions_;

  // Non-index equi-join conditions: join keys that are not index columns
  // and not partition columns (e.g., bucket columns used for colocated
  // joins). Applied as post-read equality filters during lookup.
  // Partition column conditions are stored separately in
  // partitionIndexConditions_.
  struct NonIndexCondition {
    column_index_t outputColumnIndex;
    column_index_t requestColumnIndex;
  };
  std::vector<NonIndexCondition> nonIndexConditions_;

  // Partition index condition: join condition on a partition column.
  // Used to dispatch probe rows to the correct partition group's readers.
  struct PartitionIndexCondition {
    // Table-side column name (handle->name()), used to look up the partition
    // value from HiveConnectorSplit::partitionKeys.
    std::string partitionColumnName;
    // Index of the corresponding probe column in requestType_.
    column_index_t requestColumnIndex;
    // True when the partition value string is encoded as days-since-epoch
    // (alternative to a YYYY-MM-DD date string). Mirrors the same flag on
    // HiveColumnHandle and is consumed by newConstantFromString.
    bool isPartitionDateValueDaysSinceEpoch{false};
  };
  std::vector<PartitionIndexCondition> partitionIndexConditions_;

  // Partition column handles, keyed by table column name (handle->name()).
  // Populated from kPartitionKey assignments in init(). Used to:
  // - classify a join condition as a routing condition
  // - feed partition handles into makeScanSpec()
  // - look up dataType() and isPartitionDateValueDaysSinceEpoch() when
  //   parsing partition value strings into typed constants
  std::unordered_map<std::string, FileColumnHandlePtr> partitionKeyHandles_;

  // Filters for pushdown. Includes subfield filters from table handle and
  // filters converted from constant index lookup conditions.
  common::SubfieldFilters filters_;

  // Remaining filter expression set after filter pushdown.
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;
  // Subfields referenced by the remaining filter.
  std::vector<common::Subfield> remainingFilterSubfields_;
  // Total time spent on evaluating remaining filter in nanoseconds.
  uint64_t remainingFilterTimeNs_{0};
  // Field indices referenced in both remaining filter and output type. These
  // columns need to be materialized eagerly to avoid missing values in output.
  std::vector<column_index_t> remainingEagerlyLoadFields_;

  // Filter evaluation state for remaining filter.
  SelectivityVector remainingFilterRows_;
  VectorPtr remainingFilterResult_;
  DecodedVector remainingFilterLazyDecoded_;
  SelectivityVector remainingFilterLazyBaseRows_;
  exec::FilterEvalCtx remainingFilterEvalCtx_;

  // Points to subfields from both output columns' required subfields and
  // remainingFilterSubfields_. Used to tell the reader which nested fields to
  // project out.
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      projectedSubfields_;

  // Output type for the index reader. Includes partition columns when they
  // appear in outputType_; the per-split reader emits their values as
  // constants via the scan spec's setConstantValue().
  RowTypePtr readerOutputType_;
  // All index readers (both built-in and external). Owns every reader
  // regardless of partitioned vs non-partitioned path. Created by addSplits().
  std::vector<std::unique_ptr<SplitIndexReader>> readers_;
  // Raw pointers into readers_ for the non-partitioned path. Built once
  // in addSplits() to avoid per-lookup allocation.
  std::vector<SplitIndexReader*> defaultReaders_;
  // Partition groups built during addSplits() when partition index conditions
  // are present. One group per distinct set of partition column values.
  std::vector<PartitionGroup> partitionGroups_;

  // Cached empty output vector.
  RowVectorPtr emptyOutput_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;

  // Per-call timing stats accumulated via addOperatorRuntimeStats().
  std::unordered_map<std::string, RuntimeMetric> runtimeStats_;
};

} // namespace facebook::velox::connector::hive
