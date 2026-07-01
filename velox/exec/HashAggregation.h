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

#include <string_view>

#include "velox/exec/GroupingSet.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

class HashAggregation : public Operator {
 public:
  /// Runtime stat keys for hash aggregation.
  /// Number of rows flushed in partial aggregation output.
  static constexpr std::string_view kFlushRowCount = "flushRowCount";
  /// Number of partial aggregation flush operations.
  static constexpr std::string_view kFlushTimes = "flushTimes";
  /// Ratio of output to input rows in partial aggregation as a percentage.
  static constexpr std::string_view kPartialAggregationPct =
      "partialAggregationPct";
  /// Number of rows emitted after partial aggregation was abandoned.
  static constexpr std::string_view kAbandonedPartialAggregationRows =
      "abandonedPartialAggregationRows";

  HashAggregation(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::AggregationNode>& aggregationNode);

  void initialize() override;

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !noMoreInput_ && !partialFull_;
  }

  bool startDrain() override;

  void finishDrain() override;

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  /// HashAggregation can reclaim memory via lightweight compaction, disk spill,
  /// or relocation to a memory tier; any of these being enabled makes it
  /// reclaimable.
  bool canReclaim() const override {
    return (memoryCompactionEnabled_ && hasCompactableAggregates_) ||
        canSpill() || relocationPool_ != nullptr;
  }

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

  void close() override;

 private:
  // Returns the memory tier pool to relocate the payload into, or nullptr when
  // no tier is configured or the aggregation shape forbids a byte-copy
  // relocation.
  memory::MemoryPool* relocationPoolForAggregation(
      const std::vector<std::unique_ptr<VectorHasher>>& hashers,
      const std::vector<AggregateInfo>& aggregateInfos) const;

  void updateRuntimeStats();

  void prepareOutput(vector_size_t size);

  // Invoked to reset partial aggregation state if it was full and has been
  // flushed.
  void resetPartialOutputIfNeed();

  // Invoked on partial output flush to try to bump up the partial aggregation
  // memory usage if it needs. 'aggregationPct' is the ratio between the number
  // of output rows and the number of input rows as a percentage. It is a
  // measure of the effectiveness of the partial aggregation.
  void maybeIncreasePartialAggregationMemoryUsage(double aggregationPct);

  // True if we have enough rows and not enough reduction, i.e. more than
  // 'abandonPartialAggregationMinRows_' rows and more than
  // 'abandonPartialAggregationMinPct_' % of rows are unique.
  bool abandonPartialAggregationEarly(int64_t numOutput) const;

  RowVectorPtr getDistinctOutput();

  // Setups the projections for accessing grouping keys stored in grouping
  // set.
  // For 'groupingKeyInputChannels', the index is the key column index from
  // the grouping set, and the value is the key column channel from the input.
  // For 'outputChannelProjections', the index is the key column channel from
  // the output, and the value is the key column index from the grouping set.
  void setupGroupingKeyChannelProjections(
      std::vector<column_index_t>& groupingKeyInputChannels,
      std::vector<column_index_t>& groupingKeyOutputChannels) const;

  void updateEstimatedOutputRowSize();

  // Returns whether this driver should emit the default global grouping set
  // rows.
  bool shouldEmitDefaultGlobalGroupingSetRows();

  // Barrier across peers. True only on the elected last driver when input is
  // globally empty; false when parked on 'future_' or input is non-empty.
  bool electDefaultGlobalGroupingSetDriver();

  // Returns the default global grouping set rows for the () set.
  RowVectorPtr getDefaultGlobalGroupingSetOutput();

  std::shared_ptr<const core::AggregationNode> aggregationNode_;

  const bool isPartialOutput_;
  const bool isGlobal_;
  const bool isDistinct_;
  // True for raw-input steps (kSingle/kPartial).
  const bool isRawInput_;
  // True when the aggregation has a global grouping set: the empty () set that
  // yields one grand-total row over all input.
  const bool hasGlobalGroupingSets_;
  const bool memoryCompactionEnabled_;
  const int64_t maxExtendedPartialAggregationMemoryUsage_;
  // Minimum number of rows to see before deciding to give up on partial
  // aggregation.
  const int32_t abandonPartialAggregationMinRows_;
  // Min unique rows pct for partial aggregation. If more than this many rows
  // are unique, the partial aggregation is not worthwhile.
  const int32_t abandonPartialAggregationMinPct_;

  int64_t maxPartialAggregationMemoryUsage_;
  std::unique_ptr<GroupingSet> groupingSet_;

  // Cached from groupingSet_->hasCompactableAggregates() during initialize().
  // Stored separately to allow safe access from the arbitration thread without
  // dereferencing groupingSet_.
  bool hasCompactableAggregates_{false};

  // The memory tier pool that reclaim() relocates the payload into instead of
  // disk spilling, or nullptr when no tier is configured or the aggregation
  // shape forbids a byte-copy relocation. Set during initialize().
  memory::MemoryPool* relocationPool_{nullptr};

  // Size of a single output row estimated using
  // 'groupingSet_->estimateRowSize()'. If spilling, this value is set to max
  // 'groupingSet_->estimateRowSize()' across all accumulated data set.
  std::optional<int64_t> estimatedOutputRowSize_;

  bool partialFull_ = false;
  bool newDistincts_ = false;
  bool finished_ = false;
  // True if partial aggregation has been found to be non-reducing.
  bool abandonedPartialAggregation_{false};

  RowContainerIterator resultIterator_;
  bool pushdownChecked_ = false;
  bool mayPushdown_ = false;

  // Count the number of input rows. It is reset on partial aggregation output
  // flush.
  int64_t numInputRows_ = 0;
  // Count the number of output rows. It is reset on partial aggregation output
  // flush.
  int64_t numOutputRows_ = 0;

  // Possibly reusable output vector.
  RowVectorPtr output_;

  // Set in noMoreInput() to park a non-last driver for the peer election;
  // consumed by the next isBlocked().
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  // Set only on the single elected driver when the input is globally empty;
  // that driver emits the default () grouping-set rows.
  bool emitDefaultGlobalGroupingSetRows_{false};
};

} // namespace facebook::velox::exec
