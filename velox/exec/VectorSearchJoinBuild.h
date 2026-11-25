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
#include "velox/exec/VectorSearchJoinEmbeddings.h"
#include "velox/exec/VectorSearchJoinTable.h"

namespace facebook::velox::exec {

/// The common part of the build operators of a vector search join: adds the
/// build rows that can match to a hash table, or to a row container for a join
/// without keys. The last build driver gathers the rows of all drivers and
/// hands them to the probe side as a VectorSearchJoinTable.
///
/// VectorSearchJoinIndexBuild builds index rows and VectorSearchJoinQueryBuild
/// query rows. create() makes the one for the join's index side.
class BaseVectorSearchJoinBuild : public Operator {
 public:
  /// Makes the build operator for the side 'joinNode' has its index rows on.
  static std::unique_ptr<Operator> create(
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

 protected:
  /// @param value The column of the index of index rows, or of the embedding
  /// of query rows. A row with a null key or a null value cannot match.
  /// @param keepUnmatchableRows Whether to keep the rows that cannot match,
  /// the query rows of a right join.
  BaseVectorSearchJoinBuild(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode,
      const core::FieldAccessTypedExprPtr& value,
      bool keepUnmatchableRows);

  /// Called with the value column of each input before its rows are stored.
  virtual void addBatch(const VectorPtr& /*values*/) {}

  /// The number of bytes the decoded value of a row takes.
  virtual int32_t valueSize() const = 0;

  /// Writes the decoded value of 'row', which can match, to 'bytes'.
  virtual void storeValue(
      const DecodedVector& values,
      vector_size_t row,
      std::span<char> bytes) = 0;

  /// The error of values with different dimensions.
  virtual std::string_view dimensionError() const = 0;

  /// Sets 'dimension_' to 'dimension' if it is 0, or checks that they are
  /// equal. Ignores a 0 'dimension'.
  void updateDimension(int32_t dimension);

  /// Takes what 'peer' gathered besides its rows.
  virtual void takeFromPeer(BaseVectorSearchJoinBuild& /*peer*/) {}

  /// Makes the table of the rows of all drivers that can match: 'hashTable'
  /// for a join with keys, 'rows' for a join without keys.
  virtual std::shared_ptr<const VectorSearchJoinTable> makeTable(
      std::unique_ptr<BaseHashTable> hashTable,
      std::vector<std::unique_ptr<RowContainer>> rows,
      std::vector<std::unique_ptr<RowContainer>> unmatchableRows) = 0;

  /// The dimension of the indexes or the length of the query embeddings. 0
  /// before the first.
  int32_t dimension_{0};

 private:
  // Makes 'table_', or 'rows_' for a join without keys, with a raw dependent
  // of 'valueSize' bytes for the decoded values.
  void createTable(int32_t valueSize);

  // Adds 'rows' of the current input, which cannot match, to
  // 'unmatchableRows_'.
  void addUnmatchableRows(const SelectivityVector& rows);

  // The hashers of the keys, which 'table_' owns once it is made.
  const std::vector<std::unique_ptr<VectorHasher>>& keyHashers() const {
    return table_ != nullptr ? table_->hashers() : hashers_;
  }

  const bool keepUnmatchableRows_;
  const uint32_t minTableRowsForParallelJoinBuild_;
  const uint32_t vectorHasherMaxNumDistinct_;
  // The number of keys, 0 for a join without keys.
  const column_index_t numKeys_;
  // Decode the keys of the input, at their channels, until 'table_' takes
  // them.
  std::vector<std::unique_ptr<VectorHasher>> hashers_;
  // The channels of the build columns of the output and of the value.
  std::vector<column_index_t> dependentChannels_;
  column_index_t valueChannel_;
  // The types of the build columns of the output.
  std::vector<TypePtr> dependentTypes_;

  // The build columns and the values of the current input.
  std::vector<DecodedVector> decodedDependents_;
  DecodedVector values_;

  // The rows that can match, in a hash table for a join with keys, or in a
  // row container for a join without keys. Made with the first of them.
  std::unique_ptr<BaseHashTable> table_;
  std::unique_ptr<RowContainer> rows_;
  // Whether to analyze the keys, for the table to choose an array or
  // normalized key mode.
  bool analyzeKeys_{false};
  // The rows that cannot match, if kept.
  std::unique_ptr<RowContainer> unmatchableRows_;

  // Future for synchronizing with other Drivers of the same pipeline. All build
  // Drivers must be completed before making data available for the probe side.
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/// Builds a VectorSearchJoinIndexTable of index rows, decoding the index of
/// each.
class VectorSearchJoinIndexBuild : public BaseVectorSearchJoinBuild {
 public:
  VectorSearchJoinIndexBuild(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode);

  void close() override;

 private:
  void addBatch(const VectorPtr& values) override;

  int32_t valueSize() const override;

  void storeValue(
      const DecodedVector& values,
      vector_size_t row,
      std::span<char> bytes) override;

  std::string_view dimensionError() const override;

  void takeFromPeer(BaseVectorSearchJoinBuild& peer) override;

  std::shared_ptr<const VectorSearchJoinTable> makeTable(
      std::unique_ptr<BaseHashTable> hashTable,
      std::vector<std::unique_ptr<RowContainer>> rows,
      std::vector<std::unique_ptr<RowContainer>> unmatchableRows) override;

  // Reads the indexes with the join's metric and search settings, in the
  // format the join names, from the query's registry.
  std::unique_ptr<common::vectorindex::LocalVectorIndexReader> reader_;
  // The packed indexes of the current input.
  VectorPtr packedIndexes_;
  // The decoded indexes the rows point to.
  std::vector<VectorSearchJoinIndexTable::Index> indexes_;
};

/// Builds a VectorSearchJoinQueryTable of query rows.
class VectorSearchJoinQueryBuild : public BaseVectorSearchJoinBuild {
 public:
  VectorSearchJoinQueryBuild(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::VectorSearchJoinNodePtr& joinNode);

 private:
  void addBatch(const VectorPtr& values) override;

  int32_t valueSize() const override;

  void storeValue(
      const DecodedVector& values,
      vector_size_t row,
      std::span<char> bytes) override;

  std::string_view dimensionError() const override;

  std::shared_ptr<const VectorSearchJoinTable> makeTable(
      std::unique_ptr<BaseHashTable> hashTable,
      std::vector<std::unique_ptr<RowContainer>> rows,
      std::vector<std::unique_ptr<RowContainer>> unmatchableRows) override;

  const common::vectorindex::VectorIndexMetric metric_;
  // The embeddings of the current input.
  std::optional<VectorSearchJoinEmbeddings> embeddings_;
};

} // namespace facebook::velox::exec
