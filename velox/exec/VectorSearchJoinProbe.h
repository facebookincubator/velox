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

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/VectorSearchJoinTable.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

/// The common part of the probe operators of a vector search join: gets the
/// build side's table, lists the build rows each probe row matches, searches
/// local indexes with queries, and makes the output a batch at a time. A query
/// is a query row's embedding, normalized for the metric. A search looks up the
/// queries of some query rows in one local index. A join that drops the probe
/// rows matching nothing, all but a left join, pushes filters on the keys, when
/// the table has their values, into the probe side, and reads no probe rows if
/// no build row can match.
///
/// VectorSearchJoinQueryProbe probes index rows with streaming query rows, and
/// VectorSearchJoinIndexProbe probes query rows with streaming index rows.
/// create() makes the one for the join's index side.
class BaseVectorSearchJoinProbe : public Operator {
 public:
  /// Makes the probe operator for the side 'joinNode' has its index rows on.
  static std::unique_ptr<Operator> create(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode);

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  void close() override;

 protected:
  BaseVectorSearchJoinProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode);

  /// Marks no row.
  static constexpr vector_size_t kNoRow = -1;

  /// Called once 'table_' is set.
  virtual void setTable() {}

  /// Adds results to the output batch, which is empty, until it is full or
  /// the results run out. Returns false if there are none until the next
  /// input.
  virtual bool fillResults() = 0;

  /// Sets 'maxOutputRows_' for the output of 'input_', or of build rows
  /// alone if there is no input.
  void setMaxOutputRows();

  /// Returns a buffer for 'numQueries' queries of 'dimension' to search.
  float* searchQueries(size_t numQueries, size_t dimension);

  /// Returns how many queries a search of 'index' can take with room in the
  /// output batch for all their results: 0 if the batch has no room for the
  /// results of one query.
  size_t maxQueries(const common::vectorindex::LocalVectorIndex& index);

  /// Searches 'index' for the 'k' nearest embeddings of the 'numQueries'
  /// queries in the buffer searchQueries() returned, and adds each embedding
  /// found to the output batch. 'resultRows' is called once for each query
  /// that finds embeddings, and returns the probe row and the build row of its
  /// results. maxQueries() must allow 'numQueries'.
  void search(
      const common::vectorindex::LocalVectorIndex& index,
      size_t numQueries,
      folly::FunctionRef<std::pair<vector_size_t, char*>(size_t query)>
          resultRows);

  /// Adds a left join's result for 'probeRow' of 'input_', which finds no
  /// embeddings, to the output batch.
  void addMiss(vector_size_t probeRow);

  /// Returns the room in the output batch for build rows that find no
  /// embeddings, with no probe row. addMisses() adds the first 'numRows'.
  std::span<char*> missRows();
  void addMisses(vector_size_t numRows);

  /// The number of results in the output batch.
  vector_size_t numResults() const {
    return numResults_;
  }

  bool isOutputFull() const {
    return ids_ != nullptr && numResults_ == outputCapacity_;
  }

  /// Looks up the keys of 'rows' of 'input_' in the table. Returns false if
  /// no row can match.
  bool lookup(SelectivityVector& rows);

  /// Lists up to 'maxMatches' next matches of the last lookup in
  /// 'matchProbeRows_' and 'matchBuildRows_', by probe row. Returns how many,
  /// 0 when there are no more.
  int32_t nextMatches(int32_t maxMatches);

  /// Sets the container the build rows of the output batch are in and its
  /// first build column of the output. A null container makes the build columns
  /// null.
  void setResultRows(const RowContainer* rows, column_index_t firstColumn) {
    resultRows_ = rows;
    firstResultColumn_ = firstColumn;
  }

  const bool isOuterJoin_;
  std::shared_ptr<const VectorSearchJoinTable> table_;
  size_t maxOutputRows_{0};
  raw_vector<vector_size_t> matchProbeRows_;
  raw_vector<char*> matchBuildRows_;

 private:
  // nextMatches() for a join without keys: every probe row of the last lookup
  // matches every build row.
  int32_t nextCrossMatches(int32_t maxMatches);

  // The number of results a search of 'index' finds per query, at most.
  vector_size_t searchK(
      const common::vectorindex::LocalVectorIndex& index) const;

  // Allocates the output batch if there is none, or if the batch, which must
  // be empty then, has room for fewer than 'maxSearchK_' results.
  void ensureOutput();

  // Returns the output batch and starts a new one.
  RowVectorPtr makeOutput();

  // Pushes a filter on each key, passing the keys of the build rows, into the
  // operators upstream that accept one, such as a table scan.
  void pushdownDynamicFilters();

  const int64_t k_;
  // Whether a probe row that matches nothing is output: true for a left join.
  const bool preservesProbeRows_;
  // Hash the keys of the input, at their channels, for the lookup.
  std::vector<std::unique_ptr<VectorHasher>> hashers_;
  // For each probe column of the output, its channel in the input; for each
  // build column, its position among the build columns.
  std::vector<IdentityProjection> probeProjections_;
  std::vector<IdentityProjection> buildProjections_;
  // The output channels of the id and the score, if the output includes them.
  std::optional<column_index_t> idChannel_;
  std::optional<column_index_t> scoreChannel_;

  // The last lookup for a join with keys.
  std::unique_ptr<HashLookup> lookup_;
  std::unique_ptr<BaseHashTable::JoinResultIterator> joinResults_;
  // The last lookup for a join without keys: the probe rows, and the position
  // of the next match, a probe row, a container of build rows and a row in
  // it.
  std::vector<vector_size_t> crossProbeRows_;
  size_t nextCrossProbeRow_{0};
  size_t nextCrossContainer_{0};
  RowContainerIterator nextCrossRow_;
  const RowContainer* resultRows_{nullptr};
  column_index_t firstResultColumn_{0};
  // The estimated bytes of the build columns of an output row, set with
  // 'table_'.
  uint64_t buildRowSize_{0};
  // The queries one search takes, one after another.
  BufferPtr searchQueries_;
  // The most results per query of the indexes searched so far, at least 1.
  vector_size_t maxSearchK_{1};
  // The output batch: its capacity, at least 'maxSearchK_', and for each
  // result, its probe row, its build row,
  // and the id and score of the embedding found, null if none. A search
  // writes its ids and scores in place and then drops the padding.
  vector_size_t outputCapacity_{0};
  vector_size_t numResults_{0};
  BufferPtr probeRows_;
  raw_vector<char*> buildRows_;
  FlatVectorPtr<int64_t> ids_;
  FlatVectorPtr<float> scores_;
};

/// Probes a build side of index rows with streaming query rows: searches each
/// batch of query rows in the indexes they match.
class VectorSearchJoinQueryProbe : public BaseVectorSearchJoinProbe {
 public:
  VectorSearchJoinQueryProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode);

  void addInput(RowVectorPtr input) override;

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

  void close() override;

 private:
  void setTable() override;

  bool fillResults() override;

  // Lists the matches of the rows of 'input_' by index row, and prepares the
  // queries of the rows that match any.
  void prepareSearches();

  const common::vectorindex::VectorIndexMetric metric_;
  const column_index_t embeddingChannel_;
  const VectorSearchJoinIndexTable* indexTable_{nullptr};
  // The embedding dimension of the table's indexes.
  size_t dimension_{0};

  // The matches of 'input_' as index rows and probe rows, by index row, and
  // the next to search.
  std::vector<
      std::pair<char*, vector_size_t>,
      memory::StlAllocator<std::pair<char*, vector_size_t>>>
      matches_;
  size_t nextMatch_{0};
  // For a left join, a bit per probe row, set if the row found embeddings, and
  // the next row to check.
  BufferPtr hasResults_;
  vector_size_t nextLeftJoinRow_{0};
  // The normalized embeddings of the probe rows that match index rows, one
  // per probe row at the row's position.
  BufferPtr queries_;
};

/// Probes a build side of query rows with streaming index rows: searches the
/// index of each probe row with the query rows it matches. For a right join,
/// marks the query rows that find embeddings, and the last probe driver
/// outputs the others.
class VectorSearchJoinIndexProbe : public BaseVectorSearchJoinProbe {
 public:
  VectorSearchJoinIndexProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode);

  void addInput(RowVectorPtr input) override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void close() override;

 private:
  void setTable() override;

  bool fillResults() override;

  // Returns the index of 'probeRow' of 'input_'.
  const common::vectorindex::LocalVectorIndex& probeIndex(
      vector_size_t probeRow);

  // Searches 'index', of 'probeRow', with the query rows 'buildRows'.
  void search(
      const common::vectorindex::LocalVectorIndex& index,
      vector_size_t probeRow,
      RowSpan buildRows);

  // For a right join, waits for the other probe drivers to finish their input.
  // Returns true in the last driver.
  bool lastProbe();

  // Fills the output batch with the query rows no driver found embeddings
  // for. Returns false when there are no more.
  bool fillUnmatchedResults();

  // Reads the probe indexes with the join's metric and search settings.
  std::unique_ptr<common::vectorindex::LocalVectorIndexReader> reader_;
  const column_index_t indexChannel_;
  const VectorSearchJoinQueryTable* queryTable_{nullptr};

  // The packed indexes of 'input_'.
  DecodedVector packedIndexes_;
  // The matches 'nextMatches()' listed last, and the next to search.
  int32_t numMatches_{0};
  int32_t nextMatch_{0};
  // The decoded index of 'indexRow_', the probe row searched last.
  std::unique_ptr<common::vectorindex::LocalVectorIndex> index_;
  vector_size_t indexRow_{kNoRow};

  // For a right join, whether the drivers' barrier is passed and whether this
  // is the last driver.
  bool barrierPassed_{false};
  bool lastProbe_{false};
  // In the last driver, the unmatched query rows output so far: the container
  // of the rows that can match and the position in it, then the container of
  // the rows that cannot match and the position in it.
  size_t notProbedContainer_{0};
  RowContainerIterator notProbedRows_;
  size_t unmatchableContainer_{0};
  RowContainerIterator unmatchableRows_;
  bool unmatchedDone_{false};
  // Future for waiting on the other probe drivers of a right join.
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

} // namespace facebook::velox::exec
