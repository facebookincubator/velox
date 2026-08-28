/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/writer/BufferGrowthPolicy.h"
#include "velox/dwio/nimble/writer/StreamData.h"

namespace facebook::nimble {
using ::testing::InSequence;
using ::testing::Return;

class MockGrowthPolicy : public InputBufferGrowthPolicy {
 public:
  MOCK_METHOD(
      uint64_t,
      getExtendedCapacity,
      (uint64_t newSize, uint64_t capacity),
      (const, override));
};

template <typename T>
uint64_t getAlignedBufferCapacity(
    uint64_t size,
    std::shared_ptr<velox::memory::MemoryPool> pool) {
  // Nimble Vector deoptimizes boolean from 1 bit per value to 1 byte per value
  using InnerType =
      typename std::conditional<std::is_same_v<T, bool>, uint8_t, T>::type;
  auto buf = velox::AlignedBuffer::allocateExact<InnerType>(size, pool.get());
  return buf->capacity() / sizeof(InnerType);
}

template <typename T>
class nullStreamDataTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("test");
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};
using NullTestTypes =
    ::testing::Types<NullsStreamData, NullableContentStreamData<int32_t>>;
TYPED_TEST_SUITE(nullStreamDataTest, NullTestTypes);

TEST(StreamDataTest, hasNullValuesFromSpan) {
  constexpr std::array<bool, 3> allNonNull = {true, true, true};
  constexpr std::array<bool, 3> hasNull = {true, false, true};

  struct TestCase {
    std::span<const bool> nonNulls;
    bool expected;
  };
  const std::array<TestCase, 3> testCases = {{
      {std::span<const bool>{}, false},
      {allNonNull, false},
      {hasNull, true},
  }};

  for (const auto& testCase : testCases) {
    EXPECT_EQ(StreamData::hasNullValues(testCase.nonNulls), testCase.expected);
  }
}

TYPED_TEST(nullStreamDataTest, hasNullValues) {
  using T = TypeParam;

  SchemaBuilder schemaBuilder;
  const auto scalarBuilder =
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  auto growthPolicy = DefaultInputBufferGrowthPolicy::withDefaultRanges();

  struct TestCase {
    const char* name{};
    bool mayHaveNulls{};
    uint64_t additionalSize{};
    std::vector<bool> nonNullValues;
    bool expectedHasNulls{};
    bool expectedHasNullValues{};
  };
  const std::vector<TestCase> testCases = {
      {
          "no validity bitmap",
          false,
          3,
          {},
          false,
          false,
      },
      {
          "all non-null validity bitmap",
          true,
          3,
          {true, true, true},
          true,
          false,
      },
      {
          "contains null value",
          true,
          3,
          {true, false, true},
          true,
          true,
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    T streamData(*this->pool_, descriptor, *growthPolicy);
    streamData.ensureAdditionalNullsCapacity(
        testCase.mayHaveNulls, testCase.additionalSize);
    for (const bool notNull : testCase.nonNullValues) {
      streamData.mutableNonNulls().push_back(notNull);
    }

    EXPECT_EQ(streamData.hasNulls(), testCase.expectedHasNulls);
    EXPECT_EQ(streamData.hasNullValues(), testCase.expectedHasNullValues);
  }
}

TYPED_TEST(nullStreamDataTest, ensureAdditionalNullsCapacity) {
  using T = TypeParam;
  auto helperPool = velox::memory::memoryManager()->addLeafPool("helper");

  SchemaBuilder schemaBuilder;
  const auto scalarBuilder =
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();

  // Grow from 0 to 20, but the actual capacity will be larger. Then grow again.
  MockGrowthPolicy growthPolicy;
  {
    InSequence seq;
    EXPECT_CALL(growthPolicy, getExtendedCapacity(20, 0)).WillOnce(Return(20));
    EXPECT_CALL(
        growthPolicy,
        getExtendedCapacity(
            1030, getAlignedBufferCapacity<bool>(20, helperPool)))
        .WillOnce(Return(1030));
  }
  T streamData(*this->pool_, descriptor, growthPolicy);

  // First ensure some capacity with nulls
  streamData.ensureAdditionalNullsCapacity(true, 20);
  EXPECT_EQ(this->pool_->stats().numAllocs, 1);

  // Check initial state
  EXPECT_TRUE(streamData.hasNulls());
  auto& nonNulls = streamData.mutableNonNulls();
  const auto actualCapacity1 = nonNulls.capacity();
  EXPECT_GE(actualCapacity1, 20);
  EXPECT_EQ(actualCapacity1, getAlignedBufferCapacity<bool>(20, helperPool));
  EXPECT_EQ(nonNulls.size(), 0);

  // Write 20 values: true for even indices, false for odd
  for (size_t i = 0; i < 20; ++i) {
    nonNulls.push_back(i % 2 == 0);
  }
  EXPECT_EQ(nonNulls.size(), 20);
  EXPECT_EQ(actualCapacity1, nonNulls.capacity());

  // Ask for additoinal capacity for 10 elements, it should be within the range
  // of already allocated capacity.
  EXPECT_GE(actualCapacity1, 20 + 10);
  streamData.ensureAdditionalNullsCapacity(true, 10);
  EXPECT_EQ(nonNulls.capacity(), actualCapacity1);
  EXPECT_EQ(nonNulls.size(), 20);
  EXPECT_EQ(this->pool_->stats().numAllocs, 1);

  // Grow the capacity
  streamData.ensureAdditionalNullsCapacity(true, 1000);
  const auto actualCapacity2 = nonNulls.capacity();
  EXPECT_EQ(this->pool_->stats().numAllocs, 2);

  // Check the capacity increased and size stayed the same
  EXPECT_GT(actualCapacity2, actualCapacity1);
  EXPECT_EQ(actualCapacity2, getAlignedBufferCapacity<bool>(1030, helperPool));
  EXPECT_EQ(nonNulls.size(), 20);

  // Check that previously written values are still present
  for (size_t i = 0; i < 20; ++i) {
    EXPECT_EQ(nonNulls[i], (i % 2 == 0))
        << "Null value at index " << i << " was corrupted";
  }

  // Ensure new capacity is usable
  for (size_t i = 20; i < actualCapacity2; ++i) {
    nonNulls.push_back(i % 2 == 0);
  }

  // Check the capacity and size not changed
  EXPECT_EQ(nonNulls.size(), actualCapacity2);
  EXPECT_EQ(nonNulls.capacity(), actualCapacity2);

  // Verify all data (old and new) is still correct
  for (size_t i = 0; i < actualCapacity2; ++i) {
    EXPECT_EQ(nonNulls[i], (i % 2 == 0))
        << "Null value at index " << i << " was corrupted";
  }
  EXPECT_EQ(this->pool_->stats().numAllocs, 2);
}

template <typename T>
class contentStreamDataTest : public ::testing::Test {};

using ContentTestTypes = ::testing::
    Types<ContentStreamData<int32_t>, NullableContentStreamData<int32_t>>;
TYPED_TEST_SUITE(contentStreamDataTest, ContentTestTypes);

TYPED_TEST(contentStreamDataTest, ensureMutableDataCapacity) {
  using T = TypeParam;
  velox::memory::MemoryManager::testingSetInstance(
      velox::memory::MemoryManager::Options{});
  auto memoryPool = velox::memory::memoryManager()->addLeafPool("test");
  auto helperPool = velox::memory::memoryManager()->addLeafPool("helper");
  SchemaBuilder schemaBuilder;
  const auto scalarBuilder =
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();

  // Grow from 0 to 20, then to 1000.
  MockGrowthPolicy growthPolicy;
  {
    InSequence seq;
    EXPECT_CALL(growthPolicy, getExtendedCapacity(20, 0)).WillOnce(Return(20));
    EXPECT_CALL(
        growthPolicy,
        getExtendedCapacity(
            1000, getAlignedBufferCapacity<int32_t>(20, helperPool)))
        .WillOnce(Return(1000));
  }
  T streamData(*memoryPool, descriptor, growthPolicy);
  EXPECT_EQ(memoryPool->stats().numAllocs, 0);

  // Ensure the size of 20, make sure capacity has changed
  streamData.ensureMutableDataCapacity(20);
  EXPECT_EQ(memoryPool->stats().numAllocs, 1);

  auto& data = streamData.mutableData();
  const auto actualCapacity1 = data.capacity();
  EXPECT_GE(actualCapacity1, 20);
  EXPECT_EQ(actualCapacity1, getAlignedBufferCapacity<int32_t>(20, helperPool));
  EXPECT_EQ(data.size(), 0);

  // Write 20 values into the data
  for (int32_t i = 0; i < 20; ++i) {
    data.push_back(i);
  }
  EXPECT_EQ(data.size(), 20);
  EXPECT_EQ(data.capacity(), actualCapacity1);

  // Call ensure with the same size 20, make sure capacity did not change
  streamData.ensureMutableDataCapacity(20);
  EXPECT_EQ(data.size(), 20);
  EXPECT_EQ(data.capacity(), actualCapacity1);
  EXPECT_EQ(memoryPool->stats().numAllocs, 1);

  // Grow the capacityfrom 20 to 1000
  streamData.ensureMutableDataCapacity(1000);
  const auto actualCapacity2 = data.capacity();
  EXPECT_GT(actualCapacity2, actualCapacity1);
  EXPECT_EQ(
      actualCapacity2, getAlignedBufferCapacity<int32_t>(1000, helperPool));
  EXPECT_EQ(data.size(), 20);
  EXPECT_EQ(memoryPool->stats().numAllocs, 2);

  // Check that previously written values are still present
  for (int32_t i = 0; i < 20; ++i) {
    EXPECT_EQ(data[i], i) << "Value at index " << i << " was corrupted";
  }

  // Ensure all new capacity is usable
  for (int32_t i = 20; i < actualCapacity2; ++i) {
    data.push_back(i);
  }
  // Check there are no changes in size and capacity
  EXPECT_EQ(data.size(), actualCapacity2);
  EXPECT_EQ(data.capacity(), actualCapacity2);

  // Verify all data (old and new) is still correct
  for (int32_t i = 0; i < actualCapacity2; ++i) {
    EXPECT_EQ(data[i], i) << "Value at index " << i << " was corrupted";
  }
  EXPECT_EQ(memoryPool->stats().numAllocs, 2);
}

class rowCountTest : public ::testing::Test {
 protected:
  void SetUp() override {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
    pool_ = velox::memory::memoryManager()->addLeafPool("test");
    schemaBuilder_ = std::make_unique<SchemaBuilder>();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<SchemaBuilder> schemaBuilder_;
};

TEST_F(rowCountTest, contentStreamDataRowCount) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  ContentStreamData<int32_t> streamData(*pool_, descriptor, growthPolicy);

  // Initially empty.
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_TRUE(streamData.empty());

  // Add some data.
  auto& data = streamData.mutableData();
  data.push_back(1);
  data.push_back(2);
  data.push_back(3);

  // rowCount() should match data size.
  EXPECT_EQ(streamData.rowCount(), 3);
  EXPECT_FALSE(streamData.empty());

  // Add more data.
  data.push_back(4);
  data.push_back(5);
  EXPECT_EQ(streamData.rowCount(), 5);

  // Reset clears row count.
  streamData.reset();
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_TRUE(streamData.empty());
}

TEST_F(rowCountTest, contentStreamDataStringRowCount) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  ContentStreamData<std::string_view> streamData(
      *pool_, descriptor, growthPolicy);

  EXPECT_EQ(streamData.rowCount(), 0);

  auto& data = streamData.mutableData();
  data.push_back("hello");
  data.push_back("world");
  EXPECT_EQ(streamData.rowCount(), 2);

  data.push_back("test");
  EXPECT_EQ(streamData.rowCount(), 3);

  streamData.reset();
  EXPECT_EQ(streamData.rowCount(), 0);
}

TEST_F(rowCountTest, nullsStreamDataRowCount) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  NullsStreamData streamData(*pool_, descriptor, growthPolicy);

  // Initially empty.
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_TRUE(streamData.empty());

  // Add data without nulls (bufferedCount_ is set via
  // ensureAdditionalNullsCapacity).
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/false, 5);
  EXPECT_EQ(streamData.rowCount(), 5);
  EXPECT_FALSE(streamData.empty());

  // Add more data.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/false, 3);
  EXPECT_EQ(streamData.rowCount(), 8);

  // Reset clears row count.
  streamData.reset();
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_TRUE(streamData.empty());
}

TEST_F(rowCountTest, nullsStreamDataRowCountWithNulls) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  NullsStreamData streamData(*pool_, descriptor, growthPolicy);

  // Add data with nulls.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 4);
  auto& nonNulls = streamData.mutableNonNulls();
  nonNulls.push_back(true);
  nonNulls.push_back(false);
  nonNulls.push_back(true);
  nonNulls.push_back(false);

  EXPECT_EQ(streamData.rowCount(), 4);
  EXPECT_TRUE(streamData.hasNulls());

  // Increase capacity and add more data.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 3);
  nonNulls.push_back(true);
  nonNulls.push_back(false);
  nonNulls.push_back(true);

  EXPECT_EQ(streamData.rowCount(), 7);
  EXPECT_TRUE(streamData.hasNulls());

  streamData.reset();
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_FALSE(streamData.hasNulls());
}

TEST_F(rowCountTest, nullableContentStreamDataRowCount) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  NullableContentStreamData<int32_t> streamData(
      *pool_, descriptor, growthPolicy);

  // Initially empty.
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_TRUE(streamData.empty());

  // Add data with nulls (rowCount is tracked by bufferedCount_ from
  // NullsStreamData).
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 5);
  auto& nonNulls = streamData.mutableNonNulls();
  auto& data = streamData.mutableData();
  // 5 rows: 3 non-null values, 2 nulls.
  nonNulls.push_back(true);
  data.push_back(1);
  nonNulls.push_back(false); // null
  nonNulls.push_back(true);
  data.push_back(2);
  nonNulls.push_back(false); // null
  nonNulls.push_back(true);
  data.push_back(3);

  // rowCount is based on bufferedCount_, not data size.
  EXPECT_EQ(streamData.rowCount(), 5);
  EXPECT_EQ(data.size(), 3); // Only non-null values stored.
  EXPECT_FALSE(streamData.empty());

  // Reset clears row count.
  streamData.reset();
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_TRUE(streamData.empty());
}

TEST_F(rowCountTest, nullableContentStreamDataRowCountWithoutNulls) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  NullableContentStreamData<int32_t> streamData(
      *pool_, descriptor, growthPolicy);

  // Add data without nulls.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/false, 3);
  auto& data = streamData.mutableData();
  data.push_back(10);
  data.push_back(20);
  data.push_back(30);

  EXPECT_EQ(streamData.rowCount(), 3);
  EXPECT_FALSE(streamData.hasNulls());

  streamData.reset();
  EXPECT_EQ(streamData.rowCount(), 0);
}

TEST_F(rowCountTest, nullableContentStreamDataRowCountWithAllNulls) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  NullableContentStreamData<int32_t> streamData(
      *pool_, descriptor, growthPolicy);

  // Add data with all nulls.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 5);
  auto& nonNulls = streamData.mutableNonNulls();
  // All 5 rows are nulls.
  nonNulls.push_back(false);
  nonNulls.push_back(false);
  nonNulls.push_back(false);
  nonNulls.push_back(false);
  nonNulls.push_back(false);

  // rowCount should be 5 even though no actual data values are stored.
  EXPECT_EQ(streamData.rowCount(), 5);
  EXPECT_TRUE(streamData.hasNulls());
  EXPECT_EQ(streamData.mutableData().size(), 0); // No non-null values stored.

  // Increase capacity and add more null rows.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 3);
  nonNulls.push_back(false);
  nonNulls.push_back(false);
  nonNulls.push_back(false);

  EXPECT_EQ(streamData.rowCount(), 8);
  EXPECT_TRUE(streamData.hasNulls());
  EXPECT_EQ(streamData.mutableData().size(), 0);

  streamData.reset();
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_FALSE(streamData.hasNulls());
}

TEST_F(rowCountTest, streamDataViewRowCount) {
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();

  // Create StreamDataView with explicit row count.
  std::vector<int32_t> testData = {1, 2, 3, 4, 5};
  std::string_view dataView(
      reinterpret_cast<const char*>(testData.data()),
      testData.size() * sizeof(int32_t));

  StreamDataView view(descriptor, dataView, /*rowCount=*/5);
  EXPECT_EQ(view.rowCount(), 5);
  EXPECT_EQ(view.data().size(), 5 * sizeof(int32_t));
  EXPECT_FALSE(view.hasNulls());

  StreamDataView zeroView(descriptor, std::string_view{}, /*rowCount=*/0);
  EXPECT_EQ(zeroView.rowCount(), 0);
  EXPECT_TRUE(zeroView.data().empty());
  EXPECT_FALSE(zeroView.hasNulls());
}

TEST_F(rowCountTest, streamDataViewRowCountWithNulls) {
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();

  // Create StreamDataView with nulls.
  std::vector<int32_t> testData = {1, 2, 3};
  std::string_view dataView(
      reinterpret_cast<const char*>(testData.data()),
      testData.size() * sizeof(int32_t));
  // Use std::array instead of std::vector<bool> because std::vector<bool> is
  // bit-packed and doesn't provide contiguous storage required by std::span.
  std::array<bool, 5> nonNulls = {true, false, true, false, true};

  // rowCount can differ from data element count when nulls are present.
  StreamDataView view(descriptor, dataView, /*rowCount=*/5, nonNulls);
  EXPECT_EQ(view.rowCount(), 5);
  EXPECT_EQ(view.data().size(), 3 * sizeof(int32_t)); // Only non-null data.
  EXPECT_TRUE(view.hasNulls());
  EXPECT_EQ(view.nonNulls().size(), 5);
}

TEST_F(rowCountTest, isNullStreamClassification) {
  ExactGrowthPolicy growthPolicy;
  const auto intBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto boolBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  const auto stringBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);

  ContentStreamData<int32_t> contentData(
      *pool_, intBuilder->scalarDescriptor(), growthPolicy);
  NullsStreamData nullData(
      *pool_, boolBuilder->scalarDescriptor(), growthPolicy);
  NullableContentStreamData<int32_t> nullableContentData(
      *pool_, intBuilder->scalarDescriptor(), growthPolicy);
  NullableContentStringStreamData nullableStringData(
      *pool_, stringBuilder->scalarDescriptor(), growthPolicy, growthPolicy);
  StreamDataView view(
      intBuilder->scalarDescriptor(), std::string_view{}, /*rowCount=*/0);

  EXPECT_FALSE(contentData.isNullStream());
  EXPECT_TRUE(nullData.isNullStream());
  EXPECT_FALSE(nullableContentData.isNullStream());
  EXPECT_FALSE(nullableStringData.isNullStream());
  EXPECT_FALSE(view.isNullStream());
}

class numNullsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
    pool_ = velox::memory::memoryManager()->addLeafPool("test");
    schemaBuilder_ = std::make_unique<SchemaBuilder>();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<SchemaBuilder> schemaBuilder_;
};

TEST_F(numNullsTest, contentStreamDataHasNoNulls) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  ContentStreamData<int32_t> streamData(*pool_, descriptor, growthPolicy);

  // Content-only streams never carry a validity bitmap.
  EXPECT_FALSE(streamData.hasNulls());
  EXPECT_EQ(streamData.numNulls(), 0);

  auto& data = streamData.mutableData();
  data.push_back(1);
  data.push_back(2);
  EXPECT_EQ(streamData.numNulls(), 0);
}

TEST_F(numNullsTest, nullsStreamData) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  NullsStreamData streamData(*pool_, descriptor, growthPolicy);

  // No validity bitmap -> no nulls, no scan.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/false, 3);
  EXPECT_FALSE(streamData.hasNulls());
  EXPECT_EQ(streamData.numNulls(), 0);

  // Validity bitmap [t, f, t, f] -> 2 nulls.
  streamData.reset();
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 4);
  auto& nonNulls = streamData.mutableNonNulls();
  nonNulls.push_back(true);
  nonNulls.push_back(false);
  nonNulls.push_back(true);
  nonNulls.push_back(false);
  EXPECT_TRUE(streamData.hasNulls());
  EXPECT_EQ(streamData.numNulls(), 2);
}

TEST_F(numNullsTest, nullableContentStreamData) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  NullableContentStreamData<int32_t> streamData(
      *pool_, descriptor, growthPolicy);

  // No nulls -> 0.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/false, 3);
  auto& data = streamData.mutableData();
  data.push_back(10);
  data.push_back(20);
  data.push_back(30);
  EXPECT_EQ(streamData.numNulls(), 0);

  // 5 rows: 3 non-null values, 2 nulls.
  streamData.reset();
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 5);
  auto& nonNulls = streamData.mutableNonNulls();
  auto& data2 = streamData.mutableData();
  nonNulls.push_back(true);
  data2.push_back(1);
  nonNulls.push_back(false);
  nonNulls.push_back(true);
  data2.push_back(2);
  nonNulls.push_back(false);
  nonNulls.push_back(true);
  data2.push_back(3);
  EXPECT_EQ(streamData.numNulls(), 2);

  // All 4 rows null.
  streamData.reset();
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 4);
  auto& nonNulls2 = streamData.mutableNonNulls();
  for (int i = 0; i < 4; ++i) {
    nonNulls2.push_back(false);
  }
  EXPECT_EQ(streamData.numNulls(), 4);
}

TEST_F(numNullsTest, streamDataView) {
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();

  std::vector<int32_t> testData = {1, 2, 3};
  std::string_view dataView(
      reinterpret_cast<const char*>(testData.data()),
      testData.size() * sizeof(int32_t));

  // View reports the null count precomputed by the chunker.
  StreamDataView noNulls(descriptor, dataView, /*rowCount=*/3);
  EXPECT_FALSE(noNulls.hasNulls());
  EXPECT_EQ(noNulls.numNulls(), 0);

  std::array<bool, 5> nonNulls = {true, false, true, false, true};
  StreamDataView view(
      descriptor, dataView, /*rowCount=*/5, nonNulls, /*nullCount=*/2);
  EXPECT_TRUE(view.hasNulls());
  EXPECT_EQ(view.numNulls(), 2);
}

TEST_F(numNullsTest, streamDataViewNullStream) {
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  const auto& descriptor = scalarBuilder->scalarDescriptor();

  // Null-only (struct/ROW) chunk view: validity is the payload, no bitmap, so
  // hasNulls() is false yet numNulls() reflects the chunker-precomputed count.
  const std::array<char, 5> validity = {1, 0, 1, 0, 0};
  std::string_view dataView(validity.data(), validity.size());
  StreamDataView view(
      descriptor,
      dataView,
      /*rowCount=*/5,
      /*nonNulls=*/std::nullopt,
      /*nullCount=*/3);
  EXPECT_FALSE(view.hasNulls());
  EXPECT_EQ(view.numNulls(), 3);
}

TEST_F(numNullsTest, nullableContentStringStreamData) {
  ExactGrowthPolicy growthPolicy;
  const auto scalarBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  const auto& descriptor = scalarBuilder->scalarDescriptor();
  NullableContentStringStreamData streamData(
      *pool_, descriptor, growthPolicy, growthPolicy);

  // 4 rows: "hello", null, "world", null -> 2 nulls.
  streamData.ensureAdditionalNullsCapacity(/*mayHaveNulls=*/true, 4);
  auto& nonNulls = streamData.mutableNonNulls();
  nonNulls.push_back(true);
  nonNulls.push_back(false);
  nonNulls.push_back(true);
  nonNulls.push_back(false);
  EXPECT_TRUE(streamData.hasNulls());
  EXPECT_EQ(streamData.numNulls(), 2);
}

class NullableContentStringStreamDataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
    pool_ = velox::memory::memoryManager()->addLeafPool("test");
    schemaBuilder_ = std::make_unique<SchemaBuilder>();
    scalarBuilder_ =
        schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  }

  NullableContentStringStreamData createStreamData(
      const InputBufferGrowthPolicy& dataPolicy,
      const InputBufferGrowthPolicy& bufferPolicy) {
    return NullableContentStringStreamData(
        *pool_, scalarBuilder_->scalarDescriptor(), dataPolicy, bufferPolicy);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<SchemaBuilder> schemaBuilder_;
  std::shared_ptr<ScalarTypeBuilder> scalarBuilder_;
};

TEST_F(NullableContentStringStreamDataTest, basicOperations) {
  ExactGrowthPolicy policy;
  auto streamData = createStreamData(policy, policy);

  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_TRUE(streamData.empty());

  // Add 4 rows: "hello", null, "world", null.
  const std::vector<bool> nonNullsData = {true, false, true, false};
  streamData.ensureAdditionalNullsCapacity(true, nonNullsData.size());
  for (const auto& nonNull : nonNullsData) {
    streamData.mutableNonNulls().push_back(nonNull);
  }

  const std::vector<std::string_view> testData = {"hello", "world"};
  const auto bufferSize = testData[0].size() + testData[1].size();
  streamData.ensureStringBufferCapacity(testData.size(), bufferSize);
  auto& nonNulls = streamData.mutableNonNulls();
  auto [buffer, lengths] = streamData.mutableData();
  for (const auto& str : testData) {
    buffer.insert(buffer.end(), str.begin(), str.end());
    lengths.push_back(str.size());
  }

  EXPECT_EQ(streamData.rowCount(), 4);
  EXPECT_TRUE(streamData.hasNulls());
  EXPECT_EQ(lengths.size(), 2);
  EXPECT_EQ(buffer.size(), 10);
  EXPECT_EQ(
      streamData.memoryUsed(),
      nonNulls.size() + buffer.size() +
          lengths.size() * sizeof(std::string_view));

  streamData.materialize();
  const auto* views =
      reinterpret_cast<const std::string_view*>(streamData.data().data());
  EXPECT_EQ(views[0], "hello");
  EXPECT_EQ(views[1], "world");

  streamData.reset();
  EXPECT_EQ(streamData.rowCount(), 0);
  EXPECT_TRUE(streamData.empty());
}

TEST_F(NullableContentStringStreamDataTest, ensureStringBufferCapacity) {
  MockGrowthPolicy dataPolicy;
  MockGrowthPolicy bufferPolicy;

  EXPECT_CALL(dataPolicy, getExtendedCapacity(10, 0)).WillOnce(Return(10));
  EXPECT_CALL(bufferPolicy, getExtendedCapacity(100, 0)).WillOnce(Return(100));

  auto streamData = createStreamData(dataPolicy, bufferPolicy);
  streamData.ensureStringBufferCapacity(10, 100);
  EXPECT_EQ(pool_->stats().numAllocs, 2);

  auto [buffer, lengths] = streamData.mutableData();
  EXPECT_GE(lengths.capacity(), 10);
  EXPECT_GE(buffer.capacity(), 100);

  // Zero strings is a no-op.
  streamData.ensureStringBufferCapacity(0, 100);
  EXPECT_EQ(pool_->stats().numAllocs, 2);
}
TEST(StreamDataUtilTest, isConstantBoolStream) {
  // Empty → constant.
  EXPECT_TRUE(isConstantBoolStream(""));

  // All-false → constant.
  EXPECT_TRUE(isConstantBoolStream(std::string_view("\0\0\0", 3)));

  // All-true → constant.
  EXPECT_TRUE(isConstantBoolStream(std::string_view("\1\1\1", 3)));
  EXPECT_TRUE(isConstantBoolStream(std::string_view("\1", 1)));

  // Mixed → not constant.
  EXPECT_FALSE(isConstantBoolStream(std::string_view("\0\1", 2)));
  EXPECT_FALSE(isConstantBoolStream(std::string_view("\1\0", 2)));
  EXPECT_FALSE(isConstantBoolStream(std::string_view("\0\1\0", 3)));
  EXPECT_FALSE(isConstantBoolStream(std::string_view("\1\0\1", 3)));
}

} // namespace facebook::nimble
