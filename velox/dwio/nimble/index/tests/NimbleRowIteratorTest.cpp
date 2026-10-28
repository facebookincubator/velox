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

#include "velox/dwio/nimble/index/tests/NimbleIndexTestBase.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/ScopeGuard.h>
#include <folly/coro/Timeout.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/buffer/BufferPool.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/HybridFlatMap.h"
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/common/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/NimbleIterator.h"
#include "velox/dwio/nimble/index/StripeStreams.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
#include "velox/dwio/nimble/reader/BatchReader.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileProperties.h"
#include "velox/dwio/nimble/tablet/FilePropertiesGenerated.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/tests/SchemaUtils.h"
#include "velox/dwio/nimble/tests/SharedDictionaryTestUtils.h"
#include "velox/dwio/nimble/writer/ChunkedStreamWriter.h"
#include "velox/dwio/nimble/writer/Writer.h"

#include "folly/FileUtil.h"
#include "folly/Random.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/IoUringReader.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/row/CompactRow.h"
#include "velox/row/UnsafeRowFast.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble::test {

class NimbleRowIteratorTest : public NimbleIndexProjectorTest {};

TEST_P(NimbleRowIteratorTest, rowIteratorLazilyMaterializesAfterSkippingRows) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  iterator.seekToFirst();

  constexpr uint32_t kSkippedRows{6};
  for (uint32_t skipped{0}; skipped < kSkippedRows; ++skipped) {
    iterator.next();
  }

  EXPECT_EQ(iterator.rowNumber(), kSkippedRows);
  EXPECT_EQ(iterator.stats().numReadStripes, 0);
  const auto serializedValue = iterator.value();
  EXPECT_EQ(iterator.stats().numReadStripes, 1);

  const auto actual = velox::row::CompactRow::deserialize(
      {serializedValue}, ROW({"value"}, {INTEGER()}), leafPool_.get());
  const auto* values = actual->childAt(0)->asFlatVector<int32_t>();
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(values->valueAt(0), 60);
}

TEST_P(NimbleRowIteratorTest, rowIteratorLoadsEachStripeOnce) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  iterator.seekToFirst();
  EXPECT_EQ(iterator.stats().numReadStripes, 0);

  for (int32_t row{0}; row < kRowsPerStripe * kNumStripes; ++row) {
    SCOPED_TRACE(row);
    ASSERT_TRUE(iterator.valid());
    // Preparing the value loads the stripe, and value() on its own does too.
    if (row % 2 == 0) {
      iterator.prepareValue();
      EXPECT_EQ(iterator.stats().numReadStripes, row / kRowsPerStripe + 1);
    }
    const auto serialized = iterator.value();
    const auto actual = velox::row::CompactRow::deserialize(
        {serialized}, ROW("value", INTEGER()), leafPool_.get());
    const auto expected = vectorMaker_->rowVector(
        {"value"}, {vectorMaker_->flatVector<int32_t>({row * 10})});
    expectVectorRows(actual, expected);
    EXPECT_EQ(iterator.value(), serialized);
    EXPECT_EQ(iterator.stats().numReadStripes, row / kRowsPerStripe + 1);
    iterator.next();
  }

  EXPECT_FALSE(iterator.valid());
}

TEST_P(NimbleRowIteratorTest, rowIteratorSerializesLargeStripes) {
  constexpr int32_t kRowsPerStripe{3'000};
  constexpr int32_t kNumStripes{2};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  iterator.seekToFirst();

  for (int32_t row{0}; row < kRowsPerStripe * kNumStripes; ++row) {
    ASSERT_TRUE(iterator.valid());
    const auto actual = velox::row::CompactRow::deserialize(
        {iterator.value()}, ROW("value", INTEGER()), leafPool_.get());
    const auto* values = actual->childAt(0)->asFlatVector<int32_t>();
    ASSERT_NE(values, nullptr);
    ASSERT_EQ(values->valueAt(0), row * 10) << "row " << row;
    iterator.next();
  }
  EXPECT_FALSE(iterator.valid());
  EXPECT_EQ(iterator.stats().numReadStripes, kNumStripes);
}

TEST_P(
    NimbleRowIteratorTest,
    rowIteratorKeepsKeysAlignedAfterSkippingUnreadRows) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  iterator.seekToFirst();

  const auto expectKey = [&](int64_t row) {
    SCOPED_TRACE(row);
    const auto point = makePointLookup(rowType, {"key"}, row);
    ASSERT_TRUE(point.lowerKey.has_value());
    EXPECT_EQ(iterator.rowNumber(), row);
    EXPECT_EQ(iterator.key(), *point.lowerKey);
    EXPECT_EQ(iterator.key(), *point.lowerKey);
  };
  const auto expectValue = [&](int32_t row) {
    SCOPED_TRACE(row);
    const auto actual = velox::row::CompactRow::deserialize(
        {iterator.value()}, ROW("value", INTEGER()), leafPool_.get());
    const auto expected = vectorMaker_->rowVector(
        {"value"}, {vectorMaker_->flatVector<int32_t>({row * 10})});
    expectVectorRows(actual, expected);
  };

  expectKey(0);
  constexpr uint32_t kSkippedRows{5};
  for (uint32_t skipped{0}; skipped < kSkippedRows; ++skipped) {
    SCOPED_TRACE(skipped);
    iterator.next();
  }
  expectKey(kSkippedRows);
  EXPECT_EQ(iterator.stats().numReadStripes, 0);
  expectValue(kSkippedRows);
  expectKey(kSkippedRows);

  iterator.next();
  expectKey(6);
  iterator.next();
  expectValue(7);
  iterator.next();
  expectKey(8);
}

TEST_P(NimbleRowIteratorTest, rowIteratorSkipsRowsAcrossPrefetchedStripes) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  TestNimbleIterator iterator{createRowIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 3,
      })};
  iterator.seekToFirst();

  constexpr uint32_t kSkippedRows{6};
  for (uint32_t skipped{0}; skipped < kSkippedRows; ++skipped) {
    iterator.next();
  }

  EXPECT_EQ(iterator.rowNumber(), kSkippedRows);
  const auto serializedValue = iterator.value();
  const auto actual = velox::row::CompactRow::deserialize(
      {serializedValue}, ROW({"value"}, {INTEGER()}), leafPool_.get());
  const auto* values = actual->childAt(0)->asFlatVector<int32_t>();
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(values->valueAt(0), 60);
}

TEST_P(NimbleRowIteratorTest, rowIteratorCountsSkippedPrefetchStripes) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    const NimbleIterator::Options options{
        .prefetchExecutor = prefetch ? folly::getKeepAliveToken(*executor)
                                     : folly::Executor::KeepAlive<>{},
        .maxPrefetchStripes = 6,
    };
    auto iterator = createRowIterator(projectedSubfields, options);
    iterator->seekToFirst();
    for (uint32_t row = 0; row < 8; ++row) {
      SCOPED_TRACE(row);
      iterator->nextRow();
    }
    const auto value = iterator->value();
    const auto actual = velox::row::CompactRow::deserialize(
        {value}, ROW("value", INTEGER()), leafPool_.get());
    const auto expected = vectorMaker_->rowVector(
        {"value"}, {vectorMaker_->flatVector<int32_t>({80})});
    velox::test::assertEqualVectors(expected, actual);
    EXPECT_EQ(iterator->stats().numSkippedStripes, prefetch ? 2 : 0);

    iterator->reset();
    EXPECT_EQ(iterator->stats().numSkippedStripes, prefetch ? 2 : 0);
    iterator->seekToFirst();
    iterator->value();
    EXPECT_EQ(iterator->stats().numSkippedStripes, prefetch ? 2 : 0);
    iterator->reset();
  }
}

TEST_P(NimbleRowIteratorTest, rowIteratorSupportsCustomOutputFormat) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto rowFormat = nextTestOutputRowFormat("InternalRow");
  registerOutputRowSerializer(
      rowFormat,
      [](const velox::RowVectorPtr&,
         velox::vector_size_t,
         velox::vector_size_t numRows,
         velox::memory::MemoryPool* pool,
         velox::BufferPool*,
         NimbleIterator::SerializedRows& output) {
        constexpr std::string_view kSerializedValue{"custom"};
        output.buffer = velox::AlignedBuffer::allocate<char>(
            kSerializedValue.size() * numRows, pool);
        auto* outputBytes = output.buffer->asMutable<char>();
        output.offsets.assign(1, 0);
        for (velox::vector_size_t row{0}; row < numRows; ++row) {
          std::copy(
              kSerializedValue.begin(),
              kSerializedValue.end(),
              outputBytes + output.offsets.back());
          output.offsets.push_back(
              output.offsets.back() + kSerializedValue.size());
        }
      });
  auto options = NimbleIterator::Options{};
  options.rowFormat = rowFormat;
  TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
  iterator.seekToFirst();

  EXPECT_EQ(iterator.value(), "custom");
  iterator.next();
  EXPECT_EQ(iterator.value(), "custom");
}

TEST_P(NimbleRowIteratorTest, rowIteratorSupportsEmptyCustomOutput) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto rowFormat = nextTestOutputRowFormat("EmptyRow");
  auto numSerializerCalls = std::make_shared<std::atomic<std::size_t>>(0);
  registerOutputRowSerializer(
      rowFormat,
      [numSerializerCalls](
          const velox::RowVectorPtr&,
          velox::vector_size_t,
          velox::vector_size_t numRows,
          velox::memory::MemoryPool*,
          velox::BufferPool*,
          NimbleIterator::SerializedRows& output) {
        numSerializerCalls->fetch_add(1, std::memory_order_relaxed);
        output.offsets.assign(numRows + 1, 0);
      });
  auto options = NimbleIterator::Options{};
  options.rowFormat = rowFormat;
  TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
  iterator.seekToFirst();

  // All four rows of the first stripe come from one serializer call.
  for (int32_t row{0}; row < 4; ++row) {
    EXPECT_TRUE(iterator.value().empty());
    EXPECT_TRUE(iterator.value().empty());
    iterator.next();
  }
  EXPECT_EQ(numSerializerCalls->load(std::memory_order_relaxed), 1);
}

TEST_P(
    NimbleRowIteratorTest,
    rowIteratorRecyclesSerializedRowsThroughBufferPool) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  NimbleIterator::Options options;
  options.bufferPool = &bufferPool;
  {
    TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
    iterator.seekToFirst();
    iterator.value();
    EXPECT_EQ(bufferPool.size(), 0);

    // The next stripe returns the buffer to the pool and takes it again.
    for (int32_t row{0}; row < 4; ++row) {
      iterator.next();
    }
    iterator.value();
    EXPECT_EQ(bufferPool.size(), 0);
  }
  // The destroyed iterator returned its buffer.
  EXPECT_EQ(bufferPool.size(), 1);

  {
    TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
    iterator.seekToFirst();
    EXPECT_EQ(bufferPool.size(), 1);
    iterator.value();
    EXPECT_EQ(bufferPool.size(), 0);
  }
  EXPECT_EQ(bufferPool.size(), 1);
}

TEST_P(NimbleRowIteratorTest, rowIteratorMaterializesOnlySelectedRows) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  {
    // An exact range decodes and serializes only its row.
    TestNimbleIterator iterator{
        createRowIterator(projectedSubfields, NimbleIterator::Options{})};
    iterator.seek(makePointLookup(rowType, {"key"}, 6));
    EXPECT_EQ(iterator.stats().numMaterializedRows, 0);
    iterator.value();
    EXPECT_EQ(iterator.stats().numMaterializedRows, 1);
    EXPECT_EQ(iterator.stats().numReadStripes, 1);
  }

  {
    // A range without an upper bound decodes the rest of every stripe it
    // reaches.
    auto bounds = makePointLookup(rowType, {"key"}, 6);
    bounds.upperKey.reset();
    TestNimbleIterator iterator{
        createRowIterator(projectedSubfields, NimbleIterator::Options{})};
    iterator.seek(bounds);
    iterator.value();
    EXPECT_EQ(iterator.stats().numMaterializedRows, 2);
    for (int32_t row{6}; row < kRowsPerStripe * kNumStripes; ++row) {
      iterator.value();
      iterator.next();
    }
    EXPECT_EQ(iterator.stats().numMaterializedRows, 2 + kRowsPerStripe);
  }
}

TEST_P(NimbleRowIteratorTest, rowIteratorTimesMaterializationOnlyWhenEnabled) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  for (const bool collectTimings : {true, false}) {
    SCOPED_TRACE(collectTimings);
    NimbleIterator::Options options;
    options.collectTimings = collectTimings;
    TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
    iterator.seekToFirst();
    EXPECT_EQ(iterator.stats().materializationTiming, velox::CpuWallTiming{});
    iterator.value();
    const auto timing = iterator.stats().materializationTiming;
    if (collectTimings) {
      EXPECT_EQ(timing.count, 1);
    } else {
      EXPECT_EQ(timing, velox::CpuWallTiming{});
    }
  }
}

TEST_P(NimbleRowIteratorTest, rowIteratorSeeksPoint) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  TestNimbleIterator iterator{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  // The first row, a row inside a stripe and the last row.
  for (const int64_t key : {0, 6, 11}) {
    SCOPED_TRACE(key);
    const auto bounds = makePointLookup(rowType, {"key"}, key);
    TestNimbleIterator rangeIterator{
        createRowIterator(projectedSubfields, NimbleIterator::Options{})};
    rangeIterator.seek(bounds);

    iterator.seek(*bounds.lowerKey);
    ASSERT_TRUE(iterator.valid());
    EXPECT_EQ(iterator.rowNumber(), key);
    EXPECT_EQ(iterator.key(), rangeIterator.key());
    EXPECT_EQ(iterator.value(), rangeIterator.value());
    iterator.next();
    EXPECT_FALSE(iterator.valid());
  }
  // Only the rows of the keys are decoded.
  EXPECT_EQ(iterator.stats().numMaterializedRows, 3);

  // A key without rows leaves the iterator invalid.
  iterator.seek(*makePointLookup(rowType, {"key"}, 1'000).lowerKey);
  EXPECT_FALSE(iterator.valid());
}

INSTANTIATE_TEST_SUITE_P(
    AllCacheParams,
    NimbleRowIteratorTest,
    ::testing::ValuesIn(NimbleRowIteratorTest::getIteratorTestParams()),
    NimbleRowIteratorTest::iteratorTestParamName);

} // namespace facebook::nimble::test
