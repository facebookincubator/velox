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

TEST_P(PartitioningVectorTest, testFlatBoolVector) {
  const int numValues = GetParam();

  // Random values, no nulls
  testVectorPartitioning(
      makeFlatVector<bool>(numValues, [](auto row) { return row % 2 == 0; }));

  // Random values, with half number of nulls
  testVectorPartitioning(
      makeFlatVector<bool>(
          numValues, [](auto row) { return row % 2 == 0; }, nullEvery(2, 1)));

  // All nulls
  testVectorPartitioning(makeAllNullFlatVector<bool>(numValues));
}

TEST_P(PartitioningVectorTest, testRowVector) {
  const int numValues = GetParam();

  // Two flat columns, no nulls at any level.
  testVectorPartitioning(makeRowVector({
      makeFlatVector<int32_t>(numValues, [](auto row) { return row; }),
      makeFlatVector<int64_t>(numValues, [](auto row) { return row * 10; }),
  }));

  // Two flat columns with nullable children.
  testVectorPartitioning(makeRowVector({
      makeFlatVector<int32_t>(
          numValues, [](auto row) { return row; }, nullEvery(2)),
      makeFlatVector<int64_t>(
          numValues, [](auto row) { return row * 10; }, nullEvery(3)),
  }));

  // Row-level nulls with no child nulls.
  testVectorPartitioning(makeRowVector(
      {makeFlatVector<int32_t>(numValues, [](auto row) { return row; })},
      nullEvery(2)));

  // Row-level nulls combined with nullable children.
  testVectorPartitioning(makeRowVector(
      {makeFlatVector<int32_t>(
          numValues, [](auto row) { return row; }, nullEvery(3))},
      nullEvery(2)));

  // All rows null.
  testVectorPartitioning(makeRowVector(
      {makeFlatVector<int32_t>(numValues, [](auto row) { return row; })},
      [](auto /*row*/) { return true; }));

  // Nested RowVector.
  testVectorPartitioning(makeRowVector({
      makeFlatVector<int32_t>(numValues, [](auto row) { return row; }),
      makeRowVector({
          makeFlatVector<int64_t>(numValues, [](auto row) { return row; }),
      }),
  }));
}

TEST_P(PartitioningVectorTest, testConstantVector) {
  const int numValues = GetParam();

  testVectorPartitioning(makeConstant<int32_t>(7, numValues));
  testVectorPartitioning(makeConstant<int32_t>(std::nullopt, numValues));
  testVectorPartitioning(makeConstantRow(
      ROW({"c0", "c1"}, {INTEGER(), VARCHAR()}),
      variant::row({variant(11), variant("constant")}),
      numValues));
}

// Partitioning a null-free vector must not allocate a null buffer.
TEST_P(PartitioningVectorTest, noNullBufferAllocatedForNullFreeFlat) {
  const int numValues = GetParam();
  if (numValues == 0) {
    return;
  }

  auto flat = makeFlatVector<int32_t>(numValues, [](auto row) { return row; });
  ASSERT_FALSE(flat->mayHaveNulls());

  std::vector<uint32_t> partitions(numValues);
  for (int i = 0; i < numValues; ++i) {
    partitions[i] = i % 2;
  }

  auto pv = PartitionedVector::create(flat, partitions, 2, ctx_, pool_.get());
  EXPECT_FALSE(pv->baseVector()->mayHaveNulls())
      << "partition() must not allocate a null buffer for a null-free FlatVector";
}

// Partitioning a null-free RowVector must not allocate null buffers on the
// row vector or any of its children.
TEST_P(PartitioningVectorTest, noNullBufferAllocatedForNullFreeRow) {
  const int numValues = GetParam();
  if (numValues == 0) {
    return;
  }

  auto row = makeRowVector({
      makeFlatVector<int32_t>(numValues, [](auto row) { return row; }),
      makeFlatVector<int64_t>(numValues, [](auto row) { return row * 10; }),
  });
  ASSERT_FALSE(row->mayHaveNulls());
  ASSERT_FALSE(row->childAt(0)->mayHaveNulls());
  ASSERT_FALSE(row->childAt(1)->mayHaveNulls());

  std::vector<uint32_t> partitions(numValues);
  for (int i = 0; i < numValues; ++i) {
    partitions[i] = i % 2;
  }

  auto pv = PartitionedVector::create(row, partitions, 2, ctx_, pool_.get());
  auto* base = pv->baseVector()->as<RowVector>();
  EXPECT_FALSE(base->mayHaveNulls())
      << "partition() must not allocate a null buffer for a null-free RowVector";
  EXPECT_FALSE(base->childAt(0)->mayHaveNulls())
      << "partition() must not allocate a null buffer for null-free child 0";
  EXPECT_FALSE(base->childAt(1)->mayHaveNulls())
      << "partition() must not allocate a null buffer for null-free child 1";
}

// numNullsAt() tests
// ---------------------------------------------------------------------------

// A null-free flat vector must report zero nulls for every partition.
TEST_P(PartitioningVectorTest, numNullsAtFlatNoNulls) {
  const int numValues = GetParam();
  auto flat = makeFlatVector<int32_t>(numValues, [](auto row) { return row; });

  std::vector<uint32_t> partitions(numValues);
  for (int i = 0; i < numValues; ++i) {
    partitions[i] = i % 3;
  }
  auto pv = PartitionedVector::create(flat, partitions, 3, ctx_, pool_.get());
  for (uint32_t p = 0; p < 3; ++p) {
    EXPECT_EQ(pv->numNullsAt(p), 0) << "partition " << p;
  }
}

// A flat vector with every other row null must report the exact per-partition
// null count. The sum across all partitions must equal the total null count.
TEST_P(PartitioningVectorTest, numNullsAtFlatSomeNulls) {
  const int numValues = GetParam();
  auto flat = makeFlatVector<int32_t>(
      numValues, [](auto row) { return row; }, nullEvery(2));

  std::vector<uint32_t> partitions(numValues);
  for (int i = 0; i < numValues; ++i) {
    partitions[i] = i % 3;
  }
  auto pv = PartitionedVector::create(flat, partitions, 3, ctx_, pool_.get());

  // Per-partition counts must agree with manual bit-scan of the base vector.
  const auto* rawNulls = pv->baseVector()->rawNulls();
  const auto* rawOffsets = pv->rawPartitionOffsets();
  for (uint32_t p = 0; p < 3; ++p) {
    const vector_size_t begin = p == 0 ? 0 : rawOffsets[p - 1];
    const vector_size_t end = rawOffsets[p];
    const vector_size_t expected = rawNulls
        ? BaseVector::countNulls(pv->baseVector()->nulls(), begin, end)
        : 0;
    EXPECT_EQ(pv->numNullsAt(p), expected) << "partition " << p;
  }

  // Sum across partitions must equal the total null count in the source vector.
  const vector_size_t total =
      pv->numNullsAt(0) + pv->numNullsAt(1) + pv->numNullsAt(2);
  EXPECT_EQ(total, BaseVector::countNulls(flat->nulls(), 0, numValues));
}

// An all-null flat vector must report numNullsAt(p) == rows in that partition.
TEST_P(PartitioningVectorTest, numNullsAtFlatAllNulls) {
  const int numValues = GetParam();
  auto flat = makeAllNullFlatVector<int32_t>(numValues);

  std::vector<uint32_t> partitions(numValues);
  for (int i = 0; i < numValues; ++i) {
    partitions[i] = i % 3;
  }
  auto pv = PartitionedVector::create(flat, partitions, 3, ctx_, pool_.get());

  const auto* rawOffsets = pv->rawPartitionOffsets();
  for (uint32_t p = 0; p < 3; ++p) {
    const vector_size_t begin = p == 0 ? 0 : rawOffsets[p - 1];
    const vector_size_t numRowsInPartition = rawOffsets[p] - begin;
    EXPECT_EQ(pv->numNullsAt(p), numRowsInPartition) << "partition " << p;
  }
}

// A row vector with no row-level nulls must report zero per-partition nulls at
// the row level, even when child columns have nulls.
TEST_P(PartitioningVectorTest, numNullsAtRowNoRowLevelNulls) {
  const int numValues = GetParam();
  auto row = makeRowVector({
      makeFlatVector<int32_t>(
          numValues, [](auto row) { return row; }, nullEvery(2)),
  });
  ASSERT_FALSE(row->mayHaveNulls());

  std::vector<uint32_t> partitions(numValues);
  for (int i = 0; i < numValues; ++i) {
    partitions[i] = i % 3;
  }
  auto pv = PartitionedVector::create(row, partitions, 3, ctx_, pool_.get());
  for (uint32_t p = 0; p < 3; ++p) {
    EXPECT_EQ(pv->numNullsAt(p), 0)
        << "Row-level numNullsAt() must not count child nulls, partition " << p;
  }
}

// A row vector with row-level nulls must report per-partition counts that match
// a manual bit-scan. Child null counts must be counted independently.
TEST_P(PartitioningVectorTest, numNullsAtRowRowLevelNulls) {
  const int numValues = GetParam();
  auto row = makeRowVector(
      {makeFlatVector<int32_t>(
          numValues, [](auto row) { return row; }, nullEvery(3))},
      nullEvery(2));

  std::vector<uint32_t> partitions(numValues);
  for (int i = 0; i < numValues; ++i) {
    partitions[i] = i % 3;
  }
  auto pv = PartitionedVector::create(row, partitions, 3, ctx_, pool_.get());

  const auto* rawOffsets = pv->rawPartitionOffsets();
  for (uint32_t p = 0; p < 3; ++p) {
    const vector_size_t begin = p == 0 ? 0 : rawOffsets[p - 1];
    const vector_size_t end = rawOffsets[p];
    const vector_size_t expected =
        BaseVector::countNulls(pv->baseVector()->nulls(), begin, end);
    EXPECT_EQ(pv->numNullsAt(p), expected)
        << "Row-level null count mismatch, partition " << p;
  }

  // Child null counts must be tracked independently of row-level nulls.
  auto* prv = dynamic_cast<PartitionedRowVector*>(pv.get());
  ASSERT_NE(prv, nullptr);
  auto child = prv->childAt(0);
  const auto* childOffsets = child->rawPartitionOffsets();
  for (uint32_t p = 0; p < 3; ++p) {
    const vector_size_t begin = p == 0 ? 0 : childOffsets[p - 1];
    const vector_size_t end = childOffsets[p];
    const vector_size_t expected =
        BaseVector::countNulls(child->baseVector()->nulls(), begin, end);
    EXPECT_EQ(child->numNullsAt(p), expected)
        << "Child null count mismatch, partition " << p;
  }
}

// Test with different vector sizes, including edge cases like 0 and 1.
INSTANTIATE_TEST_SUITE_P(
    FlatVectorSizes,
    PartitioningVectorTest,
    ::testing::Values(0, 1, 10, 10000));

} // namespace facebook::velox::test
