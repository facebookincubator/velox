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

#include <memory>
#include "velox/exec/Exchange.h"
#include "velox/exec/MergeSource.h"
#include "velox/exec/TreeOfLosers.h"

namespace facebook::velox::exec {

// Merge operator Implementation: This implementation uses priority queue
// to perform a k-way merge of its inputs. It stops merging if any one of
// its inputs is blocked.
class Merge : public SourceOperator {
 public:
  Merge(
      int32_t operatorId,
      DriverCtx* ctx,
      RowTypePtr outputType,
      const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
          sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      const std::string& planNodeId,
      const std::string& operatorType);

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  RowVectorPtr getOutput() override;

  const RowTypePtr& outputType() const {
    return outputType_;
  }

  memory::MappedMemory* mappedMemory() const {
    return operatorCtx_->mappedMemory();
  }

 protected:
  virtual BlockingReason addMergeSources(ContinueFuture* future) = 0;

  std::vector<std::shared_ptr<MergeSource>> sources_;

 private:
  struct SourceRow {
    RowVector* vector;
    vector_size_t index;
  };

  class Comparator {
   public:
    Comparator(
        const RowTypePtr& outputType,
        const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
            sortingKeys,
        const std::vector<core::SortOrder>& sortingOrders);

    Comparator(const Comparator&) = delete;

    // Returns true if lhs > rhs, false otherwise.
    bool operator()(const SourceRow& lhs, const SourceRow& rhs) const {
      for (const auto& key : keyInfo_) {
        if (auto result = lhs.vector->childAt(key.first)->compare(
                rhs.vector->childAt(key.first).get(),
                lhs.index,
                rhs.index,
                key.second)) {
          return result > 0;
        }
      }
      return false;
    }

   private:
    std::vector<std::pair<ChannelIndex, CompareFlags>> keyInfo_;
  };

  /// A cursor over an ordered batch of source rows copied into 'rowContainer_'.
  struct SourceCursor {
    SourceCursor(
        MergeSource* source,
        std::vector<ContinueFuture>& blockingFutures)
        : source_{source}, blockingFutures_{blockingFutures} {
      fetchMoreData();
    }

    void isReady() {
      if (needData_) {
        fetchMoreData();
      }
    }

    bool atEnd() const {
      return atEnd_;
    }

    SourceRow next() {
      SourceRow row = {data_.get(), index_};

      ++index_;
      if (index_ == data_->size()) {
        prevData_ = std::move(data_);
        fetchMoreData();
      }

      return row;
    }

   private:
    void fetchMoreData() {
      ContinueFuture future{false};
      auto reason = source_->next(data_, &future);
      if (reason != BlockingReason::kNotBlocked) {
        blockingFutures_.push_back(std::move(future));
        needData_ = true;
      } else {
        atEnd_ = !data_;
        needData_ = false;
        index_ = 0;

        if (data_) {
          for (auto& child : data_->children()) {
            child = BaseVector::loadedVectorShared(child);
          }
        }
      }
    }

    /// Ordered source rows.
    RowVectorPtr data_;
    RowVectorPtr prevData_;
    /// Index of the next row.
    vector_size_t index_{0};
    /// True if source has been exhausted.
    bool atEnd_{false};
    bool needData_{false};
    MergeSource* source_;
    std::vector<ContinueFuture>& blockingFutures_;
  };

  /// A list of cursors over batches of ordered source data. One per source.
  /// Aligned with 'sources'.
  std::vector<SourceCursor*> sourceCursors_;

  /// STL-compatible comparator to compare rows by sorting keys.
  Comparator comparator_;

  std::unique_ptr<TreeOfLosers<SourceRow, SourceCursor>> treeOfLoosers_;

  vector_size_t outputSize_{0};

  bool finished_{false};

  /// A list of blocking futures for sources. These are populates when a given
  /// source is blocked waiting for the next batch of data.
  std::vector<ContinueFuture> sourceBlockingFutures_;
  size_t numSourcesAdded_ = 0;
  size_t currentSourcePos_ = 0;
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

 protected:
  BlockingReason addMergeSources(ContinueFuture* future) override;

 private:
  bool noMoreSplits_ = false;
  size_t numSplits_{0}; // Number of splits we took to process so far.
};

} // namespace facebook::velox::exec
