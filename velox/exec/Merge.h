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

#include "velox/exec/Exchange.h"
#include "velox/exec/MergeSource.h"
#include "velox/exec/Spill.h"
#include "velox/exec/Spiller.h"
#include "velox/exec/TreeOfLosers.h"

namespace facebook::velox::exec {

class SourceStream;
class MergeBuffer;

// Merge operator Implementation: This implementation uses priority queue
// to perform a k-way merge of its inputs. It stops merging if any one of
// its inputs is blocked.
class Merge : public SourceOperator {
 public:
  Merge(
      int32_t operatorId,
      DriverCtx* driverCtx,
      RowTypePtr outputType,
      const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
          sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      const std::string& planNodeId,
      const std::string& operatorType,
      const std::optional<common::SpillConfig>& spillConfig = std::nullopt);

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  RowVectorPtr getOutput() override;

  void close() override;

  const RowTypePtr& outputType() const {
    return outputType_;
  }

 protected:
  virtual BlockingReason addMergeSources(ContinueFuture* future) = 0;

  std::vector<std::shared_ptr<MergeSource>> sources_;
  size_t numStartedSources_{0};

 private:
  void startSources();

  void initializeTreeOfLosers();

  void spill();

  /// Maximum number of rows in the output batch.
  const vector_size_t outputBatchSize_;

  std::vector<SpillSortKey> sortingKeys_;

  /// A list of cursors over batches of ordered source data. One per source.
  /// Aligned with 'sources'.
  std::vector<SourceStream*> streams_;

  /// Used to merge data from two or more sources.
  std::unique_ptr<TreeOfLosers<SourceStream>> treeOfLosers_;

  RowVectorPtr output_;

  /// Number of rows accumulated in 'output_' so far.
  vector_size_t outputSize_{0};

  bool finished_{false};

  /// A list of blocking futures for sources. These are populates when a given
  /// source is blocked waiting for the next batch of data.
  std::vector<ContinueFuture> sourceBlockingFutures_;

  std::unique_ptr<MergeBuffer> mergeBuffer_;

  std::unique_ptr<MergeSpiller> mergeSpiller_;

  // SpillReadFiles group for all partial merge run.
  std::vector<std::vector<std::unique_ptr<SpillReadFile>>> spillReadFilesGroup_;

  uint32_t maxMergeSources_;
};

class MergeBuffer {
 public:
  MergeBuffer(const RowTypePtr& type, velox::memory::MemoryPool* pool)
      : type_(type), pool_(pool) {}

  RowVectorPtr getOutputFromSource(
      vector_size_t maxOutputRows,
      std::vector<ContinueFuture>& sourceBlockingFutures,
      bool& needFinish);

  RowVectorPtr getOutputFromSpill(vector_size_t maxOutputRows);

  /// No more data to spill if sourceStreamMerger_ is null.
  bool needsSpill() const {
    return sourceStreamMerger_ != nullptr;
  }

  /// Start sources for this partial merge run.
  void maybeStartMoreSources(
      size_t& numStartedSources,
      std::vector<ContinueFuture>& sourceBlockingFutures,
      uint32_t maxMergeSources,
      vector_size_t outputBatchSize,
      const std::vector<std::shared_ptr<MergeSource>>& sources,
      const std::vector<SpillSortKey>& sortingKeys);

  void createSpillMerger(
      std::vector<std::vector<std::unique_ptr<SpillReadFile>>>
          spillReadFilesGroup);

  void addSpillRowsNum(uint64_t numRows) {
    numSpillRows_ += numRows;
  }

 private:
  const RowTypePtr type_;
  velox::memory::MemoryPool* const pool_;

  std::unique_ptr<TreeOfLosers<SourceStream>> sourceStreamMerger_;
  std::vector<SourceStream*> sourceStreams_;
  // Used to merge the sorted runs from in-memory rows and spilled rows on disk.
  std::unique_ptr<TreeOfLosers<SpillMergeStream>> spillMerger_;
  // Records the source rows to copy to 'output_' in order.
  std::vector<const RowVector*> spillSources_;
  std::vector<vector_size_t> spillSourceRows_;
  // Reusable output vector.
  RowVectorPtr output_;
  // The number of received input rows.
  uint64_t numSpillRows_{0};
  // The number of rows that has been returned.
  uint64_t numOutputRows_{0};
};

class SourceStream final : public MergeStream {
 public:
  SourceStream(
      MergeSource* source,
      const std::vector<SpillSortKey>& sortingKeys,
      uint32_t outputBatchSize)
      : source_{source},
        sortingKeys_{sortingKeys},
        outputRows_(outputBatchSize, false),
        sourceRows_(outputBatchSize) {
    keyColumns_.reserve(sortingKeys.size());
  }

  /// Returns true and appends a future to 'futures' if needs to wait for the
  /// source to produce data.
  bool isBlocked(std::vector<ContinueFuture>& futures) {
    if (needData_) {
      return fetchMoreData(futures);
    }
    return false;
  }

  bool hasData() const override {
    return !atEnd_;
  }

  /// Returns true if current source row is less then current source row in
  /// 'other'.
  bool operator<(const MergeStream& other) const override;

  /// Advances to the next row. Returns true and appends a future to 'futures'
  /// if runs out of rows in the current batch and needs to wait for the
  /// source to produce the next batch. The return flag has the meaning of
  /// 'is-blocked'.
  bool pop(std::vector<ContinueFuture>& futures);

  /// Records the output row number for the current row. Returns true if
  /// current row is the last row in the current batch, in which case the
  /// caller must call 'copyToOutput' before calling pop(). The caller must
  /// call 'setOutputRow' before calling 'pop'. The output rows must
  /// monotonically increase in between calls to 'copyToOutput'.
  bool setOutputRow(vector_size_t row) {
    outputRows_.setValid(row, true);
    return currentSourceRow_ == data_->size() - 1;
  }

  /// Called if either current row is the last row in the current batch or the
  /// caller accumulated enough output rows across all sources to produce an
  /// output batch.
  void copyToOutput(RowVectorPtr& output);

 private:
  bool fetchMoreData(std::vector<ContinueFuture>& futures);

  MergeSource* source_;

  const std::vector<SpillSortKey>& sortingKeys_;

  /// Ordered source rows.
  RowVectorPtr data_;

  /// Raw pointers to vectors corresponding to sorting key columns in the same
  /// order as 'sortingKeys_'.
  std::vector<BaseVector*> keyColumns_;

  /// Index of the current row.
  vector_size_t currentSourceRow_{0};

  /// True if source has been exhausted.
  bool atEnd_{false};

  /// True if ran out of rows in 'data_' and needs to wait for the future
  /// returned by 'source_->next()'.
  bool needData_{true};

  /// First source row that hasn't been copied out yet.
  vector_size_t firstSourceRow_{0};

  /// Output row numbers for source rows that haven't been copied out yet.
  SelectivityVector outputRows_;

  /// Reusable memory.
  std::vector<vector_size_t> sourceRows_;
};

// LocalMerge merges its source's output into a single stream of
// sorted rows. It runs single threaded. The sources may run multi-threaded and
// in the same task.
class LocalMerge : public Merge {
 public:
  LocalMerge(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::LocalMergeNode>& localMergeNode);

 protected:
  BlockingReason addMergeSources(ContinueFuture* future) override;
};

// MergeExchange merges its sources' outputs into a single stream of
// sorted rows similar to local merge. However, the sources are splits
// and may be generated by a different task.
class MergeExchange : public Merge {
 public:
  MergeExchange(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::MergeExchangeNode>& orderByNode);

  VectorSerde* serde() const {
    return serde_;
  }

  VectorSerde::Options* serdeOptions() const {
    return serdeOptions_.get();
  }

  void close() override;

 protected:
  BlockingReason addMergeSources(ContinueFuture* future) override;

 private:
  VectorSerde* const serde_;
  const std::unique_ptr<VectorSerde::Options> serdeOptions_;
  bool noMoreSplits_ = false;
  // Task Ids from all the splits we took to process so far.
  std::vector<std::string> remoteSourceTaskIds_;
};

} // namespace facebook::velox::exec
