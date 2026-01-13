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

#include "velox/vector/tests/utils/PartitionedVectorTestBase.h"

namespace facebook::velox::test {

VectorPtr PartitionedVectorTestBase::canonicalize(VectorPtr vector) {
  auto numRows = vector->size();

  auto indices = makeIndices(numRows, [&](auto row) { return row; });
  vector_size_t* indicesRange = indices->asMutable<vector_size_t>();

  // Sort the indices based on the vector values
  std::stable_sort(
      indicesRange,
      indicesRange + numRows,
      [&](vector_size_t left, vector_size_t right) {
        return vector->compare(vector.get(), left, right) < 0;
      });

  auto sortedVector = wrapInDictionary(indices, numRows, vector);
  return sortedVector;
}

std::vector<VectorPtr> PartitionedVectorTestBase::partitionVectorByWrapping(
    VectorPtr vector,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions) {
  auto numRows = vector->size();

  // Count the number of rows in each partition
  std::vector<uint32_t> partitionRowCounts(numPartitions, 0);
  for (int i = 0; i < numRows; i++) {
    partitionRowCounts[partitions[i]]++;
  }

  std::vector<VectorPtr> partitionedVectors(numPartitions, nullptr);

  for (int p = 0; p < numPartitions; p++) {
    auto numRowsInPartition = partitionRowCounts[p];

    if (numRowsInPartition == 0) {
      partitionedVectors[p] =
          BaseVector::create(vector->type(), 0, pool_.get());
      continue;
    }

    // Create an indices buffer for each partition, and fill it with the row
    // indices for that partition.
    std::vector<vector_size_t> rowIdsInPartition(numRowsInPartition);
    vector_size_t offset = 0;
    for (vector_size_t i = 0; i < numRows; ++i) {
      if (partitions[i] == p) {
        VELOX_DCHECK_LT(offset, numRowsInPartition);
        rowIdsInPartition[offset++] = i;
      }
    }
    VELOX_CHECK_EQ(offset, numRowsInPartition);
    auto indices = makeIndices(partitionRowCounts[p], [&](auto row) {
      return rowIdsInPartition[row];
    });

    // Simulate partitioning by building the DictionaryVector with the
    // partitioned indices
    // Copy firsts because wrapInDictionary would take the ownership of the
    // vector
    VectorPtr vectorCopy = BaseVector::copy(*vector, pool_.get());
    auto dictionaryVector = BaseVector::wrapInDictionary(
        nullptr, indices, numRowsInPartition, vectorCopy);
    partitionedVectors[p] = canonicalize(dictionaryVector);
  }
  return partitionedVectors;
}

std::vector<VectorPtr> PartitionedVectorTestBase::partitionRowVectors(
    const std::vector<RowVectorPtr>& rowVectors,
    int32_t numPartitions,
    core::PartitionFunction* partitionFunction) {
  //  RowVectorPtr mergedRowVector = mergeRowVectors(rowVectors);
  VectorPtr mergedRowVector =
      mergeVectors((const std::vector<VectorPtr>&)rowVectors);
  auto totalNumRows = mergedRowVector->size();

  std::vector<uint32_t> partitions(totalNumRows, 0);
  if (numPartitions > 1) {
    auto rowType = asRowType(mergedRowVector->type());
    //    auto partitionFunction = createPartitionFunction(rowType, {0});
    partitionFunction->partition(*mergedRowVector->as<RowVector>(), partitions);
  }

  std::vector<VectorPtr> partitionedVectors =
      partitionVectorByWrapping(mergedRowVector, partitions, numPartitions);

  for (auto& vector : partitionedVectors) {
    vector = canonicalize(vector);
  }
  return partitionedVectors;
}

VectorPtr PartitionedVectorTestBase::mergeVectors(
    const std::vector<VectorPtr>& vectors) {
  // We have to count the total number of rows first in order to allocate the
  // mergedRowVector.
  auto mergedVector = BaseVector::copy(*vectors[0]);
  for (auto i = 1; i < vectors.size(); ++i) {
    mergedVector->append(vectors[i].get());
  }

  return mergedVector;
}

} // namespace facebook::velox::test
