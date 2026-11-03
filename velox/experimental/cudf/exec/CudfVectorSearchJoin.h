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

#include "velox/experimental/cudf/exec/CudfJoinBuild.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/VectorSearchJoinBuild.h"

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudaEvent;

/// The build side of a vector search join on the GPU: the build rows that can
/// match, with their keys hashed for the probe side's join, and the device
/// copy of each row's local index. It reaches the probe side through the
/// task's VectorSearchJoinBridge.
class CudfVectorSearchJoinTable : public exec::VectorSearchJoinTable {
 public:
  /// 'rows' holds the build rows, 'numKeys' keys then the build columns of
  /// the join's output, materialized on 'stream'. 'indexes' holds each row's
  /// index, each with a device copy, and 'packedIndexes' the host bytes they
  /// refer to. Throws a user error if the indexes have different dimensions.
  CudfVectorSearchJoinTable(
      std::unique_ptr<cudf::table> rows,
      column_index_t numKeys,
      std::vector<Index> indexes,
      VectorPtr packedIndexes,
      cuda::stream_ref stream);

  ~CudfVectorSearchJoinTable() override;

  cudf::table_view deviceRows() const {
    return rows_->view();
  }

  cudf::size_type numRows() const {
    return rows_->num_rows();
  }

  /// Hashes the keys of the rows. Null if there are no keys or no rows.
  const cudf::hash_join* hashJoin() const {
    return hashJoin_.get();
  }

  /// The embedding dimension of the indexes. 0 if there are no rows.
  int32_t embeddingDimension() const {
    return dimension_;
  }

  const common::vectorindex::LocalVectorIndex& index(
      cudf::size_type row) const {
    return *indexes_[row];
  }

  /// Makes 'stream' wait until the rows are ready.
  void waitUntilReady(cuda::stream_ref stream) const;

  /// Makes the stream that frees the rows wait for the reads on 'stream'.
  void recordRead(cuda::stream_ref stream) const;

 private:
  const cuda::stream_ref stream_;
  std::unique_ptr<cudf::table> rows_;
  std::unique_ptr<cudf::hash_join> hashJoin_;
  // Holds the bytes 'indexes_' refer to. Declared before 'indexes_' so that
  // it outlives them.
  const VectorPtr packedIndexes_;
  const std::vector<Index> indexes_;
  int32_t dimension_{0};
  std::shared_ptr<CudaEvent> readyEvent_;
};

/// Gathers the build rows of a vector search join on the GPU and loads each
/// row's local index onto the device, then hands them to the probe side as a
/// CudfVectorSearchJoinTable. The packed indexes are read on the host, the one
/// place FAISS can parse them; the rows and the searches stay on the device.
class CudfVectorSearchJoinBuild : public CudfJoinBuild {
 public:
  CudfVectorSearchJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::VectorSearchJoinNode> joinNode);

 protected:
  void buildAndPublish(std::vector<CudfVectorPtr> inputs) override;

 private:
  const std::shared_ptr<const core::VectorSearchJoinNode> joinNode_;
  const common::vectorindex::LocalVectorIndexOptions searchOptions_;
  // Channels of the keys, then of the build columns of the output.
  std::vector<cudf::size_type> channels_;
  cudf::size_type numKeys_;
  cudf::size_type indexChannel_;
};

/// Joins each probe row with the 'k' embeddings nearest its query embedding
/// in each build row's local index whose keys match, on the GPU. A cuDF hash
/// join pairs probe rows with build rows, the queries are gathered and
/// normalized in device memory, FAISS GPU and cuVS search each index with
/// them in place, and the output is gathered from the device columns. Inner
/// and left joins.
class CudfVectorSearchJoinProbe : public CudfOperatorBase {
 public:
  CudfVectorSearchJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::VectorSearchJoinNode> joinNode);

  static bool isSupportedJoinType(core::JoinType joinType) {
    return joinType == core::JoinType::kInner ||
        joinType == core::JoinType::kLeft;
  }

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  // The results of joining 'probe': for each, the probe row, the build row
  // (-1 for none), the id and the score.
  std::unique_ptr<cudf::table> search(
      cudf::table_view probe,
      cuda::stream_ref stream);

  // The (probe row, build row) pairs of 'probe' to search, ordered by build
  // row, leaving out probe rows with a null embedding.
  std::unique_ptr<cudf::table> pairsToSearch(
      cudf::table_view probe,
      cuda::stream_ref stream);

  const bool isLeftJoin_;
  const common::vectorindex::VectorIndexMetric metric_;
  const int64_t k_;
  std::vector<cudf::size_type> keyChannels_;
  cudf::size_type embeddingChannel_;
  // (input channel, output channel) of the probe and build columns of the
  // output. Build input channels are columns of the table's rows.
  std::vector<std::pair<cudf::size_type, column_index_t>> probeProjections_;
  std::vector<std::pair<cudf::size_type, column_index_t>> buildProjections_;
  std::optional<column_index_t> idChannel_;
  std::optional<column_index_t> scoreChannel_;

  std::shared_ptr<const CudfVectorSearchJoinTable> table_;
};

} // namespace facebook::velox::cudf_velox
