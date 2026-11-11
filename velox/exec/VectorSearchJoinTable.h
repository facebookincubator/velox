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
#include <span>

#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/exec/HashTable.h"

namespace facebook::velox::exec {

/// The build side of a vector search join: the build rows that can match. Each
/// row holds the keys, the build columns of the join's output, and in its raw
/// dependent the decoded value: a pointer to the index of an index row, or the
/// normalized embedding of a query row. A join with keys finds the rows by key
/// in a hash table. A join without keys matches every probe row with every
/// build row.
class VectorSearchJoinTable {
 public:
  virtual ~VectorSearchJoinTable() = default;

  /// The rows that can match, by key, for a join with keys. Null for a join
  /// without keys, or if no row can match.
  BaseHashTable* hashTable() const {
    return hashTable_.get();
  }

  /// The containers of the rows that can match, all with the same layout.
  /// Empty if no row can match.
  const std::vector<RowContainer*>& rows() const {
    return rows_;
  }

  /// The number of keys of a row of rows(). The build columns of the output
  /// follow them.
  column_index_t numKeys() const {
    return hashTable_ != nullptr ? hashTable_->hashers().size() : 0;
  }

  /// The dimension of the indexes or of the query embeddings. 0 if
  /// there are no rows that can match.
  int32_t dimension() const {
    return dimension_;
  }

 protected:
  /// Takes the rows that can match: 'hashTable' for a join with keys, or
  /// 'rows', one container per build driver, for a join without keys.
  VectorSearchJoinTable(
      std::unique_ptr<BaseHashTable> hashTable,
      std::vector<std::unique_ptr<RowContainer>> rows,
      int32_t dimension);

  /// The raw dependent of 'row', a row of rows().
  std::span<char> rawDependent(char* row) const {
    return rows_.front()->rawDependent(row);
  }

 private:
  const std::unique_ptr<BaseHashTable> hashTable_;
  const std::vector<std::unique_ptr<RowContainer>> ownedRows_;
  // The containers of 'hashTable_' or 'ownedRows_'.
  std::vector<RowContainer*> rows_;
  const int32_t dimension_;
};

/// A build side of index rows. The raw dependent of a row holds a pointer to
/// its decoded index. All indexes have the same dimension.
class VectorSearchJoinIndexTable : public VectorSearchJoinTable {
 public:
  /// A decoded index and the vector holding the bytes it refers to.
  struct Index {
    // Declared first so that it outlives 'index'.
    VectorPtr storage;
    std::unique_ptr<common::vectorindex::LocalVectorIndex> index;
  };

  /// 'indexes' are the indexes the rows point to.
  VectorSearchJoinIndexTable(
      std::unique_ptr<BaseHashTable> hashTable,
      std::vector<std::unique_ptr<RowContainer>> rows,
      int32_t dimension,
      std::vector<Index> indexes)
      : VectorSearchJoinTable(std::move(hashTable), std::move(rows), dimension),
        indexes_{std::move(indexes)} {}

  /// The index of 'row', a row of rows().
  const common::vectorindex::LocalVectorIndex& index(char* row) const {
    return **reinterpret_cast<common::vectorindex::LocalVectorIndex* const*>(
        rawDependent(row).data());
  }

 private:
  const std::vector<Index> indexes_;
};

/// A build side of query rows. The raw dependent of a row holds its embedding,
/// normalized for the metric. All embeddings have the same dimension. For a
/// right join, the query rows that cannot match, those with a null key or a
/// null embedding, are kept apart with the build columns of the output only.
class VectorSearchJoinQueryTable : public VectorSearchJoinTable {
 public:
  VectorSearchJoinQueryTable(
      std::unique_ptr<BaseHashTable> hashTable,
      std::vector<std::unique_ptr<RowContainer>> rows,
      int32_t dimension,
      std::vector<std::unique_ptr<RowContainer>> unmatchableRows)
      : VectorSearchJoinTable(std::move(hashTable), std::move(rows), dimension),
        unmatchableRows_{std::move(unmatchableRows)} {}

  /// The normalized embedding of 'row', a row of rows().
  std::span<const float> query(char* row) const {
    return {
        reinterpret_cast<const float*>(rawDependent(row).data()),
        static_cast<size_t>(dimension())};
  }

  /// The rows that cannot match, for a right join.
  const std::vector<std::unique_ptr<RowContainer>>& unmatchableRows() const {
    return unmatchableRows_;
  }

 private:
  const std::vector<std::unique_ptr<RowContainer>> unmatchableRows_;
};

} // namespace facebook::velox::exec
