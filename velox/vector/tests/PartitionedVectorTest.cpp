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
#include <algorithm>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include "vector/tests/utils/VectorTestBase.h"
#include "velox/vector/PartitionedVector.h"
#include "velox/vector/tests/utils/PartitionedVectorTestBase.h"

namespace facebook::velox::test {

class PartitioningVectorTest : public testing::TestWithParam<int>,
                               public test::PartitionedVectorTestBase {
 protected:
  std::mt19937 gen_ = std::mt19937(std::random_device{}());

  PartitionBuildContext ctx_;
  BufferPtr partitionOffsets_;

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  void testPartitionedVector(
      VectorPtr vector,
      const std::vector<uint32_t>& partitions,
      uint32_t numPartitions) {
    // Back up the vector before calling PartitionedVector::create()
    VectorPtr vectorCopy = BaseVector::copy(*vector);
    // Build the expected vector using the reference implementation
    std::vector<VectorPtr> expectedVectors =
        partitionVectorByWrapping(vectorCopy, partitions, numPartitions);

    // Initialize buffers needed for PartitionedVector::create()
    ensureCapacity<vector_size_t>(
        ctx_.cursorPartitionOffsets, numPartitions, pool_.get());

    // Calculate the number of values for each partition
    std::vector<vector_size_t> partitionRowCounts(numPartitions, 0);
    for (auto partition : partitions) {
      partitionRowCounts[partition]++;
    }

    // Create the partitioned vector using the actual implementation
    auto partitionedVector = PartitionedVector::create(
        vector,
        partitions,
        numPartitions,
        //        partitionOffsets_,
        ctx_,
        pool_.get());
    VELOX_CHECK_NOT_NULL(partitionedVector);

    // Extract each partition and compare with expected results
    std::vector<VectorPtr> partitionedVectors;
    for (uint32_t i = 0; i < numPartitions; ++i) {
      auto partition = partitionedVector->partitionAt(i);
      partitionedVectors.push_back(partition);
    }

    for (uint32_t i = 0; i < numPartitions; ++i) {
      test::assertEqualVectors(
          expectedVectors[i], canonicalize(partitionedVectors[i]));
    }
  }

  void testVectorPartitioning(VectorPtr vector) {
    auto numRows = vector->size();
    std::vector<uint32_t> partitions(numRows);

    // Test with single partition
    std::fill(partitions.begin(), partitions.end(), 0);
    auto vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, 1);

    // Test with two partitions
    if (vector->size() >= 3) {
      for (uint32_t i = 0; i < partitions.size(); ++i) {
        partitions[i] = i % 2;
      }
      vectorCopy = BaseVector::copy(*vector, pool_.get());
      testPartitionedVector(vectorCopy, partitions, 2);
    }

    // Test with three partitions
    for (uint32_t i = 0; i < partitions.size(); ++i) {
      partitions[i] = i % 3;
    }
    vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, 3);

    if (vector->size() > 4) {
      // Test with four partitions where the first partition is empty
      for (uint32_t i = 0; i < partitions.size(); ++i) {
        partitions[i] = i % 3 + 1;
      }
      vectorCopy = BaseVector::copy(*vector, pool_.get());
      testPartitionedVector(vectorCopy, partitions, 4);

      // Test with four partitions where the last partition is empty
      for (uint32_t i = 0; i < partitions.size(); ++i) {
        partitions[i] = i % 3;
      }
      vectorCopy = BaseVector::copy(*vector, pool_.get());
      testPartitionedVector(vectorCopy, partitions, 4);
    }

    // Test with one value per partition
    if (vector->size() > 0) {
      std::iota(partitions.begin(), partitions.end(), 0);
      vectorCopy = BaseVector::copy(*vector, pool_.get());
      testPartitionedVector(vectorCopy, partitions, numRows);
    }

    // Test with random partitions (number of partitions <= number of values)
    std::uniform_int_distribution<> dis(0, numRows - 1);
    uint32_t maxPartition = 0;
    for (uint32_t i = 0; i < numRows; ++i) {
      partitions[i] = dis(gen_);
      maxPartition = std::max(maxPartition, partitions[i]);
    }
    vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, maxPartition + 1);
  }
};

TEST_P(PartitioningVectorTest, testFlatVector) {
  // Number of values in the vector to be partitioned. This is passed as a test
  // parameter and is used to test different vector sizes, including edge cases
  // like 0 and 1.
  const int numValues = GetParam();

  // Random values, no nulls
  testVectorPartitioning(
      makeFlatVector<int>(numValues, [](auto row) { return row; }));

  // Random values, with half number of nulls
  testVectorPartitioning(
      makeFlatVector<int>(
          numValues, [](auto row) { return row; }, nullEvery(2, 1)));

  // All nulls
  testVectorPartitioning(makeAllNullFlatVector<int>(numValues));
}

// Test with different vector sizes, including edge cases like 0 and 1.
INSTANTIATE_TEST_SUITE_P(
    FlatVectorSizes,
    PartitioningVectorTest,
    ::testing::Values(0, 1, 10, 10000));

} // namespace facebook::velox::test
