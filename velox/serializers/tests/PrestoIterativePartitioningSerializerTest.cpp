/*
 * Copyright (c) International Business Machines Corporation
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

#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/serializers/PrestoIterativePartitioningSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::serializer::presto;
using namespace facebook::velox::test;

// ---------------------------------------------------------------------------
// Shared base fixture
// ---------------------------------------------------------------------------

class PrestoIterativePartitioningSerializerTestBase : public VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    if (!isRegisteredVectorSerde()) {
      PrestoVectorSerde::registerVectorSerde();
    }
  }

  /// Deserializes an IOBuf produced by PartitioningSerializer::flush().
  RowVectorPtr deserialize(folly::IOBuf& iobuf, const RowTypePtr& type) {
    auto ranges = byteRangesFromIOBuf(&iobuf);
    BufferInputStream stream(std::move(ranges));
    RowVectorPtr result;
    serde_.deserialize(&stream, pool_.get(), type, &result, nullptr);
    return result;
  }

  /// Extracts flat values from a column into a sorted vector.
  template <typename T>
  std::vector<T> sortedValues(const RowVectorPtr& row, int column) {
    auto* flat = row->childAt(column)->as<FlatVector<T>>();
    std::vector<T> vals(flat->rawValues(), flat->rawValues() + row->size());
    std::sort(vals.begin(), vals.end());
    return vals;
  }

  /// Extracts values from a nullable column, preserving order and nulls.
  template <typename T>
  std::vector<std::optional<T>> nullableValues(
      const RowVectorPtr& row,
      int column) {
    auto* vec = row->childAt(column).get();
    std::vector<std::optional<T>> result;
    result.reserve(row->size());
    for (int i = 0; i < row->size(); ++i) {
      if (vec->isNullAt(i)) {
        result.push_back(std::nullopt);
      } else {
        result.push_back(vec->as<FlatVector<T>>()->valueAt(i));
      }
    }
    return result;
  }

  /// Builds a PrestoIterativePartitioningSerializer with default serde options.
  std::unique_ptr<PrestoIterativePartitioningSerializer> makeSerializer(
      const RowTypePtr& type,
      uint32_t numPartitions) {
    SerdeOpts opts;
    return std::make_unique<PrestoIterativePartitioningSerializer>(
        type, numPartitions, opts, pool_.get());
  }

  PrestoVectorSerde serde_;
};

// ---------------------------------------------------------------------------
// Value-parameterized fixture — routing, null-handling over scalar TypePtrs.
// Uses BaseVector::create() + setNull() so no C++ type dispatch is needed.
// ---------------------------------------------------------------------------

class PrestoIterativePartitioningSerializerParamTest
    : public ::testing::TestWithParam<TypePtr>,
      public PrestoIterativePartitioningSerializerTestBase {
 public:
  static void SetUpTestSuite() {
    PrestoIterativePartitioningSerializerTestBase::SetUpTestSuite();
  }
};

// Short lowercase names for test output, matching the benchmark convention.
std::string scalarTypeName(const TypePtr& type) {
  if (type->kind() == TypeKind::BOOLEAN)
    return "bool";
  if (type->kind() == TypeKind::INTEGER)
    return "int";
  if (type->kind() == TypeKind::BIGINT)
    return "bigint";
  if (type->kind() == TypeKind::HUGEINT)
    return "hugeint";
  return type->toString();
}

INSTANTIATE_TEST_SUITE_P(
    ScalarTypes,
    PrestoIterativePartitioningSerializerParamTest,
    ::testing::Values(BOOLEAN(), INTEGER(), BIGINT(), HUGEINT()),
    [](const ::testing::TestParamInfo<TypePtr>& info) {
      return scalarTypeName(info.param);
    });

// ── Routing ──────────────────────────────────────────────────────────────────

// Single append, two equal-sized partitions; also verifies rowsBuffered and
// bytesBuffered lifecycle counters.
TEST_P(PrestoIterativePartitioningSerializerParamTest, basicTwoPartitions) {
  auto colType = GetParam();
  auto type = ROW({"a"}, {colType});
  auto col = BaseVector::create(colType, 6, pool_.get());
  auto input = makeRowVector({"a"}, {col});

  // Even rows → partition 0, odd rows → partition 1.
  auto serializer = makeSerializer(type, 2);
  serializer->append(input, {0, 1, 0, 1, 0, 1});

  EXPECT_EQ(serializer->rowsBuffered(), 6);

  auto ioBufs = serializer->flush();
  ASSERT_EQ(ioBufs.size(), 2);

  EXPECT_EQ(serializer->rowsBuffered(), 0);
  EXPECT_EQ(serializer->bytesBuffered(), 0);

  auto p0 = deserialize(*ioBufs.at(0).first, type);
  auto p1 = deserialize(*ioBufs.at(1).first, type);

  EXPECT_EQ(p0->size(), 3);
  EXPECT_EQ(p1->size(), 3);
}

// All rows routed to one non-zero partition; other partitions are absent.
TEST_P(PrestoIterativePartitioningSerializerParamTest, allRowsToOnePartition) {
  auto colType = GetParam();
  auto type = ROW({"x"}, {colType});
  auto col = BaseVector::create(colType, 5, pool_.get());
  auto input = makeRowVector({"x"}, {col});

  auto serializer = makeSerializer(type, 4);
  serializer->append(input, {2, 2, 2, 2, 2});
  auto ioBufs = serializer->flush();

  ASSERT_EQ(ioBufs.size(), 1);
  ASSERT_TRUE(ioBufs.count(2));
  EXPECT_EQ(deserialize(*ioBufs.at(2).first, type)->size(), 5);
}

// Single partition (numPartitions=1): all rows go to partition 0.
TEST_P(PrestoIterativePartitioningSerializerParamTest, singlePartition) {
  auto colType = GetParam();
  auto type = ROW({"a"}, {colType});
  auto col = BaseVector::create(colType, 5, pool_.get());
  auto input = makeRowVector({"a"}, {col});

  auto serializer = makeSerializer(type, 1);
  serializer->append(input, std::vector<uint32_t>(5, 0));
  auto ioBufs = serializer->flush();

  ASSERT_EQ(ioBufs.size(), 1);
  EXPECT_EQ(deserialize(*ioBufs.at(0).first, type)->size(), 5);
}

// Multiple columns of the same type: each is serialized independently by
// flushRowChildren.
TEST_P(PrestoIterativePartitioningSerializerParamTest, multipleColumns) {
  auto colType = GetParam();
  auto type = ROW({"a", "b"}, {colType, colType});
  auto colA = BaseVector::create(colType, 4, pool_.get());
  auto colB = BaseVector::create(colType, 4, pool_.get());
  auto input = makeRowVector({"a", "b"}, {colA, colB});

  auto serializer = makeSerializer(type, 2);
  serializer->append(input, {0, 0, 1, 1});
  auto ioBufs = serializer->flush();

  ASSERT_EQ(ioBufs.size(), 2);

  auto r0 = deserialize(*ioBufs.at(0).first, type);
  EXPECT_EQ(r0->size(), 2);
  EXPECT_EQ(r0->childAt(0)->size(), 2);
  EXPECT_EQ(r0->childAt(1)->size(), 2);

  auto r1 = deserialize(*ioBufs.at(1).first, type);
  EXPECT_EQ(r1->size(), 2);
  EXPECT_EQ(r1->childAt(0)->size(), 2);
  EXPECT_EQ(r1->childAt(1)->size(), 2);
}

// ── Null handling
// ─────────────────────────────────────────────────────────────

// Nulls appear only in one partition; the other partition is null-free.
// Rows 0,1,2 → p0; rows 3,4 → p1. Row 1 is null.
// p0: [not-null, null, not-null]; p1: [not-null, not-null].
TEST_P(PrestoIterativePartitioningSerializerParamTest, nullsInOnePartition) {
  auto colType = GetParam();
  auto type = ROW({"a"}, {colType});
  auto col = BaseVector::create(colType, 5, pool_.get());
  col->setNull(1, true);
  auto input = makeRowVector({"a"}, {col});

  auto serializer = makeSerializer(type, 2);
  serializer->append(input, {0, 0, 0, 1, 1});
  auto ioBufs = serializer->flush();

  ASSERT_EQ(ioBufs.size(), 2);

  auto r0 = deserialize(*ioBufs.at(0).first, type);
  ASSERT_EQ(r0->size(), 3);
  EXPECT_FALSE(r0->childAt(0)->isNullAt(0));
  EXPECT_TRUE(r0->childAt(0)->isNullAt(1));
  EXPECT_FALSE(r0->childAt(0)->isNullAt(2));

  auto r1 = deserialize(*ioBufs.at(1).first, type);
  ASSERT_EQ(r1->size(), 2);
  EXPECT_FALSE(r1->childAt(0)->isNullAt(0));
  EXPECT_FALSE(r1->childAt(0)->isNullAt(1));
}

// Nulls contributed by different appends to the same partition.
// Append 1: rows 0,1 → p0 (row 1 null); row 2 → p1.
// Append 2: row 0 → p0 (null); row 1 → p1.
// p0: [not-null, null, null]; p1: [not-null, not-null].
TEST_P(
    PrestoIterativePartitioningSerializerParamTest,
    nullsAcrossMultipleAppends) {
  auto colType = GetParam();
  auto type = ROW({"a"}, {colType});
  auto serializer = makeSerializer(type, 2);

  auto col1 = BaseVector::create(colType, 3, pool_.get());
  col1->setNull(1, true);
  serializer->append(makeRowVector({"a"}, {col1}), {0, 0, 1});

  auto col2 = BaseVector::create(colType, 2, pool_.get());
  col2->setNull(0, true);
  serializer->append(makeRowVector({"a"}, {col2}), {0, 1});

  auto ioBufs = serializer->flush();
  ASSERT_EQ(ioBufs.size(), 2);

  auto r0 = deserialize(*ioBufs.at(0).first, type);
  ASSERT_EQ(r0->size(), 3);
  EXPECT_FALSE(r0->childAt(0)->isNullAt(0));
  EXPECT_TRUE(r0->childAt(0)->isNullAt(1));
  EXPECT_TRUE(r0->childAt(0)->isNullAt(2));

  auto r1 = deserialize(*ioBufs.at(1).first, type);
  ASSERT_EQ(r1->size(), 2);
  EXPECT_FALSE(r1->childAt(0)->isNullAt(0));
  EXPECT_FALSE(r1->childAt(0)->isNullAt(1));
}

// Partition boundary falls in the middle of a null-bitmap byte, exercising the
// bit-extraction carry-over logic. 5 rows → p0, 4 rows → p1. The boundary at
// bit 5 is inside the first byte of the null bitmap. Rows 1,3,5,7 are null.
// p0: [not-null, null, not-null, null, not-null].
// p1: [null, not-null, null, not-null].
TEST_P(PrestoIterativePartitioningSerializerParamTest, nullsUnalignedBoundary) {
  auto colType = GetParam();
  auto type = ROW({"a"}, {colType});
  auto col = BaseVector::create(colType, 9, pool_.get());
  col->setNull(1, true);
  col->setNull(3, true);
  col->setNull(5, true);
  col->setNull(7, true);
  auto input = makeRowVector({"a"}, {col});

  auto serializer = makeSerializer(type, 2);
  serializer->append(input, {0, 0, 0, 0, 0, 1, 1, 1, 1});
  auto ioBufs = serializer->flush();

  ASSERT_EQ(ioBufs.size(), 2);

  auto r0 = deserialize(*ioBufs.at(0).first, type);
  ASSERT_EQ(r0->size(), 5);
  EXPECT_FALSE(r0->childAt(0)->isNullAt(0));
  EXPECT_TRUE(r0->childAt(0)->isNullAt(1));
  EXPECT_FALSE(r0->childAt(0)->isNullAt(2));
  EXPECT_TRUE(r0->childAt(0)->isNullAt(3));
  EXPECT_FALSE(r0->childAt(0)->isNullAt(4));

  auto r1 = deserialize(*ioBufs.at(1).first, type);
  ASSERT_EQ(r1->size(), 4);
  EXPECT_TRUE(r1->childAt(0)->isNullAt(0));
  EXPECT_FALSE(r1->childAt(0)->isNullAt(1));
  EXPECT_TRUE(r1->childAt(0)->isNullAt(2));
  EXPECT_FALSE(r1->childAt(0)->isNullAt(3));
}

// Both partitions contain nulls.
// Input: 4 rows, rows 1 and 2 null; rows 0,1 → p0; rows 2,3 → p1.
// p0: [not-null, null]; p1: [null, not-null].
TEST_P(PrestoIterativePartitioningSerializerParamTest, nullsInBothPartitions) {
  auto colType = GetParam();
  auto type = ROW({"a"}, {colType});
  auto col = BaseVector::create(colType, 4, pool_.get());
  col->setNull(1, true);
  col->setNull(2, true);
  auto input = makeRowVector({"a"}, {col});

  auto serializer = makeSerializer(type, 2);
  serializer->append(input, {0, 0, 1, 1});
  auto ioBufs = serializer->flush();

  ASSERT_EQ(ioBufs.size(), 2);

  auto r0 = deserialize(*ioBufs.at(0).first, type);
  ASSERT_EQ(r0->size(), 2);
  EXPECT_FALSE(r0->childAt(0)->isNullAt(0));
  EXPECT_TRUE(r0->childAt(0)->isNullAt(1));

  auto r1 = deserialize(*ioBufs.at(1).first, type);
  ASSERT_EQ(r1->size(), 2);
  EXPECT_TRUE(r1->childAt(0)->isNullAt(0));
  EXPECT_FALSE(r1->childAt(0)->isNullAt(1));
}

// All rows in one partition are null; the other partition is non-null.
// Input: 3 rows, rows 0,1 null; rows 0,1 → p0; row 2 → p1.
TEST_P(PrestoIterativePartitioningSerializerParamTest, allNullsInPartition) {
  auto colType = GetParam();
  auto type = ROW({"a"}, {colType});
  auto col = BaseVector::create(colType, 3, pool_.get());
  col->setNull(0, true);
  col->setNull(1, true);
  auto input = makeRowVector({"a"}, {col});

  auto serializer = makeSerializer(type, 2);
  serializer->append(input, {0, 0, 1});
  auto ioBufs = serializer->flush();

  ASSERT_EQ(ioBufs.size(), 2);

  auto r0 = deserialize(*ioBufs.at(0).first, type);
  ASSERT_EQ(r0->size(), 2);
  EXPECT_TRUE(r0->childAt(0)->isNullAt(0));
  EXPECT_TRUE(r0->childAt(0)->isNullAt(1));

  auto r1 = deserialize(*ioBufs.at(1).first, type);
  ASSERT_EQ(r1->size(), 1);
  EXPECT_FALSE(r1->childAt(0)->isNullAt(0));
}

// A null batch followed by a null-free batch for the same partition.
// Regression: bitmaps must be initialized to all-not-null so that rows from
// the null-free batch (rawNulls == nullptr) are not decoded as null.
TEST_P(
    PrestoIterativePartitioningSerializerParamTest,
    nullBatchFollowedByNullFreeBatch) {
  auto colType = GetParam();
  auto type = ROW({"a"}, {colType});
  auto serializer = makeSerializer(type, 2);

  // Append 1: row 0 → p0 (null); row 1 → p1 (not-null).  rawNulls non-null.
  auto col1 = BaseVector::create(colType, 2, pool_.get());
  col1->setNull(0, true);
  serializer->append(makeRowVector({"a"}, {col1}), {0, 1});

  // Append 2: all not-null (rawNulls == nullptr).  row 0 → p0; row 1 → p1.
  auto col2 = BaseVector::create(colType, 2, pool_.get());
  serializer->append(makeRowVector({"a"}, {col2}), {0, 1});

  auto ioBufs = serializer->flush();
  ASSERT_EQ(ioBufs.size(), 2);

  // p0: [null (append 1), not-null (append 2)]
  auto r0 = deserialize(*ioBufs.at(0).first, type);
  ASSERT_EQ(r0->size(), 2);
  EXPECT_TRUE(r0->childAt(0)->isNullAt(0));
  EXPECT_FALSE(r0->childAt(0)->isNullAt(1));

  // p1: [not-null (append 1), not-null (append 2)]
  auto r1 = deserialize(*ioBufs.at(1).first, type);
  ASSERT_EQ(r1->size(), 2);
  EXPECT_FALSE(r1->childAt(0)->isNullAt(0));
  EXPECT_FALSE(r1->childAt(0)->isNullAt(1));
}

// ---------------------------------------------------------------------------
// Non-typed fixture (TEST_F) — lifecycle, structural, regression
// ---------------------------------------------------------------------------

class PrestoIterativePartitioningSerializerTest
    : public ::testing::Test,
      public PrestoIterativePartitioningSerializerTestBase {
 public:
  static void SetUpTestSuite() {
    PrestoIterativePartitioningSerializerTestBase::SetUpTestSuite();
  }
};

// Appending an empty RowVector produces no ioBufs on flush.
TEST_F(PrestoIterativePartitioningSerializerTest, appendEmptyVector) {
  auto type = ROW({"a"}, {BIGINT()});
  auto serializer = makeSerializer(type, 2);
  serializer->append(makeRowVector({"a"}, {makeFlatVector<int64_t>({})}), {});
  EXPECT_TRUE(serializer->flush().empty());
}

// ── Lifecycle
// ─────────────────────────────────────────────────────────────────

// Multiple append() calls accumulate correctly before flush.
TEST_F(PrestoIterativePartitioningSerializerTest, multipleAppends) {
  auto type = ROW({"v"}, {BIGINT()});
  auto serializer = makeSerializer(type, 3);

  serializer->append(
      makeRowVector({"v"}, {makeFlatVector<int64_t>({100, 200, 300})}),
      {0, 1, 2});
  serializer->append(
      makeRowVector({"v"}, {makeFlatVector<int64_t>({400, 500, 600})}),
      {2, 0, 1});

  EXPECT_EQ(serializer->rowsBuffered(), 6);

  auto ioBufs = serializer->flush();
  ASSERT_EQ(ioBufs.size(), 3);

  auto r0 = deserialize(*ioBufs.at(0).first, type);
  auto r1 = deserialize(*ioBufs.at(1).first, type);
  auto r2 = deserialize(*ioBufs.at(2).first, type);

  ASSERT_EQ(r0->size(), 2);
  ASSERT_EQ(r1->size(), 2);
  ASSERT_EQ(r2->size(), 2);

  EXPECT_EQ(sortedValues<int64_t>(r0, 0), (std::vector<int64_t>{100, 500}));
  EXPECT_EQ(sortedValues<int64_t>(r1, 0), (std::vector<int64_t>{200, 600}));
  EXPECT_EQ(sortedValues<int64_t>(r2, 0), (std::vector<int64_t>{300, 400}));
}

// Flush twice: second flush on empty state returns an empty map.
TEST_F(PrestoIterativePartitioningSerializerTest, flushTwice) {
  auto type = ROW({"a"}, {BIGINT()});
  auto serializer = makeSerializer(type, 2);
  serializer->append(
      makeRowVector({"a"}, {makeFlatVector<int64_t>({10, 20})}), {0, 1});

  auto ioBufs1 = serializer->flush();
  ASSERT_EQ(ioBufs1.size(), 2);

  EXPECT_TRUE(serializer->flush().empty());
}

// Append and flush multiple independent cycles.
TEST_F(PrestoIterativePartitioningSerializerTest, multipleCycles) {
  auto type = ROW({"a"}, {INTEGER()});
  auto serializer = makeSerializer(type, 2);

  for (int cycle = 0; cycle < 3; ++cycle) {
    serializer->append(
        makeRowVector(
            {"a"}, {makeFlatVector<int32_t>({cycle * 2, cycle * 2 + 1})}),
        {0, 1});
    auto ioBufs = serializer->flush();
    ASSERT_EQ(ioBufs.size(), 2) << "cycle " << cycle;

    auto r0 = deserialize(*ioBufs.at(0).first, type);
    auto r1 = deserialize(*ioBufs.at(1).first, type);
    ASSERT_EQ(r0->size(), 1) << "cycle " << cycle;
    ASSERT_EQ(r1->size(), 1) << "cycle " << cycle;
    EXPECT_EQ(r0->childAt(0)->as<FlatVector<int32_t>>()->valueAt(0), cycle * 2);
    EXPECT_EQ(
        r1->childAt(0)->as<FlatVector<int32_t>>()->valueAt(0), cycle * 2 + 1);
  }
}

// ── Scale and regression
// ───────────────────────────────────────────────────────

// 1024 partitions with random int64 values: verify every value reaches
// exactly the right partition and nothing is lost or duplicated.
TEST_F(PrestoIterativePartitioningSerializerTest, manyPartitionsRandom) {
  constexpr uint32_t kNumPartitions = 1024;
  constexpr int32_t kNumRows = 64'000;

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int64_t> valueDist;
  std::uniform_int_distribution<uint32_t> partDist(0, kNumPartitions - 1);

  std::vector<int64_t> inputValues(kNumRows);
  std::vector<uint32_t> partitions(kNumRows);
  // expected[p] holds the sorted values assigned to partition p.
  std::vector<std::vector<int64_t>> expected(kNumPartitions);

  for (int i = 0; i < kNumRows; ++i) {
    inputValues[i] = valueDist(rng);
    partitions[i] = partDist(rng);
    expected[partitions[i]].push_back(inputValues[i]);
  }
  for (auto& v : expected) {
    std::sort(v.begin(), v.end());
  }

  auto type = ROW({"v"}, {BIGINT()});
  auto input = makeRowVector({"v"}, {makeFlatVector<int64_t>(inputValues)});

  auto serializer = makeSerializer(type, kNumPartitions);
  serializer->append(input, partitions);
  auto ioBufs = serializer->flush();

  // Every non-empty partition must have a page; empty partitions must not.
  for (uint32_t p = 0; p < kNumPartitions; ++p) {
    if (expected[p].empty()) {
      EXPECT_EQ(ioBufs.count(p), 0) << "partition " << p;
    } else {
      ASSERT_EQ(ioBufs.count(p), 1) << "partition " << p;
      auto result = deserialize(*ioBufs.at(p).first, type);
      ASSERT_EQ(result->size(), static_cast<int32_t>(expected[p].size()))
          << "partition " << p;
      EXPECT_EQ(sortedValues<int64_t>(result, 0), expected[p])
          << "partition " << p;
    }
  }
}

// 1024 partitions with random int64 values and ~25% nulls: verify every
// value and null reaches exactly the right partition in input order, and
// nothing is lost or duplicated.
TEST_F(
    PrestoIterativePartitioningSerializerTest,
    manyPartitionsRandomWithNulls) {
  constexpr uint32_t kNumPartitions = 1024;
  constexpr int32_t kNumRows = 64'000;
  constexpr int32_t kNullPct = 25;

  std::mt19937_64 rng(43);
  std::uniform_int_distribution<int64_t> valueDist;
  std::uniform_int_distribution<uint32_t> partDist(0, kNumPartitions - 1);
  std::uniform_int_distribution<int32_t> nullDist(0, 99);

  std::vector<std::optional<int64_t>> inputValues(kNumRows);
  std::vector<uint32_t> partitions(kNumRows);
  // expected[p] holds the sequence of (value-or-null) assigned to partition p
  // in input order.
  std::vector<std::vector<std::optional<int64_t>>> expected(kNumPartitions);

  for (int i = 0; i < kNumRows; ++i) {
    partitions[i] = partDist(rng);
    if (nullDist(rng) < kNullPct) {
      inputValues[i] = std::nullopt;
    } else {
      inputValues[i] = valueDist(rng);
    }
    expected[partitions[i]].push_back(inputValues[i]);
  }

  auto type = ROW({"v"}, {BIGINT()});
  auto input =
      makeRowVector({"v"}, {makeNullableFlatVector<int64_t>(inputValues)});

  auto serializer = makeSerializer(type, kNumPartitions);
  serializer->append(input, partitions);
  auto ioBufs = serializer->flush();

  // Partition rearranges values within each partition, so compare sorted.
  // std::optional<T> sorts with nullopt < any value, preserving null count.
  for (uint32_t p = 0; p < kNumPartitions; ++p) {
    if (expected[p].empty()) {
      EXPECT_EQ(ioBufs.count(p), 0) << "partition " << p;
    } else {
      ASSERT_EQ(ioBufs.count(p), 1) << "partition " << p;
      auto result = deserialize(*ioBufs.at(p).first, type);
      ASSERT_EQ(result->size(), static_cast<int32_t>(expected[p].size()))
          << "partition " << p;

      auto expectedSorted = expected[p];
      std::sort(expectedSorted.begin(), expectedSorted.end());

      auto actual = nullableValues<int64_t>(result, 0);
      std::sort(actual.begin(), actual.end());

      EXPECT_EQ(actual, expectedSorted) << "partition " << p;
    }
  }
}

// Regression: flushNulls previously wrote null bitmaps by obtaining a raw
// pointer via writePosition() then advancing the stream via seekp(). This
// assumed the pre-allocated IOBufOutputStream had a single contiguous range,
// but StreamArena::newRange caps each range at the size of one allocator run,
// which can be smaller than the requested size. seekp() then failed because
// the target position exceeded the end of the first (and only) range.
//
// Reproducing condition: 16 columns × 10'000 rows × 50% nulls in one
// partition generates enough output (~100 KB) to trigger the run-size cap.
TEST_F(
    PrestoIterativePartitioningSerializerTest,
    flushNullsBitmapManyColumnsLargeRowCount) {
  constexpr int32_t kNumCols = 16;
  constexpr int32_t kNumRows = 10'000;

  std::vector<std::string> names;
  std::vector<VectorPtr> children;
  names.reserve(kNumCols);
  children.reserve(kNumCols);

  for (int col = 0; col < kNumCols; ++col) {
    names.push_back(fmt::format("c{}", col));
    // Rows where (row % 2 == 0) are null; the rest hold (row * kNumCols + col).
    children.push_back(
        makeFlatVector<int64_t>(
            kNumRows,
            [col](auto row) {
              return static_cast<int64_t>(row * kNumCols + col);
            },
            [](auto row) { return (row % 2) == 0; }));
  }

  auto input = makeRowVector(names, children);
  auto rowType = std::static_pointer_cast<const RowType>(input->type());

  auto serializer = makeSerializer(rowType, 1);
  serializer->append(input, std::vector<uint32_t>(kNumRows, 0));
  auto ioBufs = serializer->flush();

  ASSERT_EQ(ioBufs.size(), 1);

  auto result = deserialize(*ioBufs.at(0).first, rowType);
  ASSERT_EQ(result->size(), kNumRows);

  for (int col = 0; col < kNumCols; ++col) {
    auto* flat = result->childAt(col)->as<FlatVector<int64_t>>();
    for (int row = 0; row < kNumRows; ++row) {
      if ((row % 2) == 0) {
        EXPECT_TRUE(result->childAt(col)->isNullAt(row))
            << "col=" << col << " row=" << row;
      } else {
        ASSERT_FALSE(result->childAt(col)->isNullAt(row))
            << "col=" << col << " row=" << row;
        EXPECT_EQ(
            flat->valueAt(row), static_cast<int64_t>(row * kNumCols + col))
            << "col=" << col << " row=" << row;
      }
    }
  }
}
