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

#include <span>

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/VectorSearchJoinBuild.h"

namespace facebook::velox::exec {

/// Searches, for each probe row of a vector search join, the vector indexes of
/// the build rows it matches, and outputs the embeddings found a batch at a
/// time. A query is a probe row's embedding, normalized for the metric. A
/// search looks up the queries of some probe rows in one local index.
class VectorSearchJoinProbe : public Operator {
 public:
  VectorSearchJoinProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode);

  bool needsInput() const override {
    return input_ == nullptr;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

  void close() override;

 private:
  // Estimates the size of an output row: its probe and build columns, the id
  // and the score.
  uint64_t estimateOutputRowSize() const;

  // Fills 'searches_' with the local indexes the rows of 'input_' search, and
  // prepares the queries of the rows that search any.
  void prepareSearches();

  // Copies the embedding at 'index' in 'arrays', whose elements are
  // 'elements', into 'query', normalized for the metric.
  void prepareQuery(
      const ArrayVector& arrays,
      vector_size_t index,
      const FlatVector<float>& elements,
      std::span<float> query) const;

  // Searches build 'row's index for the embeddings of 'probeRows', adding the
  // embeddings found to 'results_'.
  void search(vector_size_t buildRow, std::span<const vector_size_t> probeRows);

  // Refills 'results_' with the results of the next searches, up to an output
  // batch, then, for a left join, with the probe rows that found no
  // embeddings. Returns false when 'input_' has no more results.
  bool fillResults();

  // Makes an output batch of 'numResults' results from 'start' in 'results_'.
  RowVectorPtr makeOutput(size_t start, vector_size_t numResults);

  const bool isLeftJoin_;
  const common::vectorindex::VectorIndexMetric metric_;
  const int64_t k_;
  std::vector<column_index_t> keyChannels_;
  column_index_t embeddingChannel_;
  // For each probe and build column of the output, its channel in the input
  // or in the table's rows.
  std::vector<IdentityProjection> probeProjections_;
  std::vector<IdentityProjection> buildProjections_;
  // The output channels of the id and the score, if the output includes them.
  std::optional<column_index_t> idChannel_;
  std::optional<column_index_t> scoreChannel_;

  std::shared_ptr<const VectorSearchJoinTable> table_;
  // The embedding dimension of the table's indexes.
  size_t dimension_{0};

  // A local index 'input_' searches, by its row in 'table_', and the probe rows
  // to search it with.
  struct Search {
    vector_size_t buildRow;
    std::vector<vector_size_t> probeRows;
  };

  // The local indexes 'input_' searches, each with the probe rows to search it
  // with, in the order they are first matched.
  std::vector<Search> searches_;
  // The next search, and the next of its probe rows.
  size_t nextSearch_{0};
  size_t nextProbeRow_{0};
  // For a left join, a bit per probe row, set if the row found embeddings, and
  // the next row to check.
  BufferPtr hasResults_;
  vector_size_t nextLeftJoinRow_{0};

  // A probe row with an embedding found in a build row's index, or with no
  // build row for a left join's probe row that finds no embeddings.
  struct Result {
    vector_size_t probeRow;
    std::optional<vector_size_t> buildRow;
    int64_t id;
    float score;
  };

  // The number of rows of an output batch of 'input_'.
  size_t maxOutputRows_{0};
  std::vector<Result, memory::StlAllocator<Result>> results_;
  // The number of 'results_' already output.
  size_t numOutputResults_{0};
  // The normalized embeddings of the probe rows that match build rows, one
  // per probe row at the row's position.
  BufferPtr queries_;
  // The embeddings of the probe rows one search takes, one after another.
  BufferPtr searchQueries_;
  // The ids and scores one search writes, 'k' per query.
  BufferPtr searchIds_;
  BufferPtr searchScores_;
};

} // namespace facebook::velox::exec
