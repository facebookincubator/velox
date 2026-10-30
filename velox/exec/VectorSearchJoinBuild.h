/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <folly/Function.h>
#include <folly/container/F14Map.h>

#include "velox/common/vectorindex/Embeddings.h"
#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

/// The build side of a vector search join: the build rows that can match,
/// each with its decoded local index, looked up by the join keys. A build side
/// holds few rows, one per local index, so a map from the hash of the keys to
/// rows serves the lookups.
class VectorSearchJoinTable {
 public:
  using Index = std::unique_ptr<common::vectorindex::LocalVectorIndex>;

  /// 'rows' holds the build rows: their 'numKeys' keys, then the build
  /// columns of the join's output. 'indexes' holds each row's index. No key
  /// is null. 'packedColumns' hold the bytes the indexes refer to. Throws a
  /// user error if the indexes have different dimensions.
  VectorSearchJoinTable(
      RowVectorPtr rows,
      column_index_t numKeys,
      std::vector<Index> indexes,
      std::vector<VectorPtr> packedColumns);

  const RowVectorPtr& rows() const {
    return rows_;
  }

  /// Calls 'onMatch' with each build row whose keys equal the keys of
  /// probe 'row' in 'probeKeys'. A null probe key matches nothing.
  void forEachMatch(
      const std::vector<const BaseVector*>& probeKeys,
      vector_size_t row,
      folly::FunctionRef<void(vector_size_t buildRow)> onMatch) const;

  /// The embedding dimension of the indexes, which all have the same one. 0
  /// if there are no build rows.
  int32_t dimension() const {
    return dimension_;
  }

  /// The number of results per query of searching build 'row's index for the
  /// 'k' nearest embeddings: 'k', capped at the size of the index.
  int64_t numResults(vector_size_t row, int64_t k) const;

  /// Searches build 'row's index for the 'numResults' embeddings nearest to
  /// each of 'queries', writing them to 'ids' and 'scores' as
  /// LocalVectorIndex::search does.
  void search(
      vector_size_t row,
      const common::vectorindex::Embeddings& queries,
      int64_t numResults,
      std::span<int64_t> ids,
      std::span<float> scores) const;

 private:
  const RowVectorPtr rows_;
  const column_index_t numKeys_;
  // Holds the bytes 'indexes_' refer to. Declared before 'indexes_' so that
  // it outlives them.
  const std::vector<VectorPtr> packedColumns_;
  const std::vector<Index> indexes_;
  // The key columns of 'rows_'.
  std::vector<const BaseVector*> keys_;
  int32_t dimension_{0};
  // The rows of 'rows_' by the hash of their keys.
  folly::F14FastMap<uint64_t, std::vector<vector_size_t>> rowsByHash_;
};

/// Decodes the vector index of each build row of a vector search join and,
/// in the last build driver, hands the build rows of all drivers to the probe
/// side as a VectorSearchJoinTable.
class VectorSearchJoinBuild : public Operator {
 public:
  VectorSearchJoinBuild(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void close() override;

 private:
  // Concatenates 'rows_' into one vector.
  RowVectorPtr mergeRows() const;

  const common::vectorindex::VectorIndexMetric metric_;
  const common::vectorindex::LocalVectorIndexOptions searchOptions_;
  // Channels of the keys, then of the build columns of the output.
  std::vector<column_index_t> channels_;
  // The type of the input projected on 'channels_'.
  RowTypePtr rowType_;
  column_index_t numKeys_;
  column_index_t indexChannel_;

  // The input rows that can match, those with no null key and a non-null
  // index, projected on 'channels_'.
  std::vector<RowVectorPtr> rows_;
  // The input columns holding the bytes 'indexes_' refer to.
  std::vector<VectorPtr> packedColumns_;
  // The decoded index of each row in 'rows_', in order.
  std::vector<VectorSearchJoinTable::Index> indexes_;

  // Future for synchronizing with other Drivers of the same pipeline. All build
  // Drivers must be completed before making data available for the probe side.
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

} // namespace facebook::velox::exec
