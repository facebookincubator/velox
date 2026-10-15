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

#include "velox/dwio/nimble/velox/index/tests/NimbleIndexTestBase.h"

#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/ScopeGuard.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Timeout.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
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
#include "velox/dwio/nimble/velox/BatchReader.h"
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"
#include "velox/dwio/nimble/velox/HybridFlatMap.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/dwio/nimble/velox/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/velox/index/NimbleIterator.h"
#include "velox/dwio/nimble/velox/index/StripeStreams.h"
#include "velox/dwio/nimble/velox/tests/SchemaUtils.h"
#include "velox/dwio/nimble/velox/tests/SharedDictionaryTestUtils.h"
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
    folly::coro::blockingWait(iterator->co_seekToFirst());
    for (uint32_t row = 0; row < 8; ++row) {
      SCOPED_TRACE(row);
      folly::coro::blockingWait(iterator->co_nextRow());
    }
    const auto value = folly::coro::blockingWait(iterator->co_value());
    const auto actual = velox::row::CompactRow::deserialize(
        {value}, ROW("value", INTEGER()), leafPool_.get());
    const auto expected = vectorMaker_->rowVector(
        {"value"}, {vectorMaker_->flatVector<int32_t>({80})});
    velox::test::assertEqualVectors(expected, actual);
    EXPECT_EQ(iterator->stats().numSkippedStripes, prefetch ? 2 : 0);

    folly::coro::blockingWait(iterator->co_reset());
    EXPECT_EQ(iterator->stats().numSkippedStripes, prefetch ? 2 : 0);
    folly::coro::blockingWait(iterator->co_seekToFirst());
    folly::coro::blockingWait(iterator->co_value());
    EXPECT_EQ(iterator->stats().numSkippedStripes, prefetch ? 2 : 0);
    folly::coro::blockingWait(iterator->co_reset());
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
         velox::memory::MemoryPool* pool,
         velox::BufferPtr& output) {
        constexpr std::string_view kSerializedValue{"custom"};
        if (output == nullptr) {
          output = velox::AlignedBuffer::allocate<char>(
              kSerializedValue.size(), pool);
        }
        output->setSize(kSerializedValue.size());
        auto* outputBytes = output->asMutable<char>();
        for (std::size_t i{0}; i < kSerializedValue.size(); ++i) {
          outputBytes[i] = kSerializedValue[i];
        }
        return kSerializedValue.size();
      });
  auto options = NimbleIterator::Options{};
  options.rowFormat = rowFormat;
  TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
  iterator.seekToFirst();

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
          velox::memory::MemoryPool*,
          velox::BufferPtr&) {
        numSerializerCalls->fetch_add(1, std::memory_order_relaxed);
        return std::size_t{0};
      });
  auto options = NimbleIterator::Options{};
  options.rowFormat = rowFormat;
  TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
  iterator.seekToFirst();

  EXPECT_TRUE(iterator.value().empty());
  EXPECT_TRUE(iterator.value().empty());
  EXPECT_EQ(numSerializerCalls->load(std::memory_order_relaxed), 1);
}

INSTANTIATE_TEST_SUITE_P(
    AllCacheParams,
    NimbleRowIteratorTest,
    ::testing::ValuesIn(NimbleRowIteratorTest::getTestParams()),
    [](const ::testing::TestParamInfo<TestParam>& info) {
      return fmt::format(
          "cacheMeta{}_pin{}",
          info.param.cacheMetadata ? "On" : "Off",
          info.param.pinMetadata ? "On" : "Off");
    });

} // namespace facebook::nimble::test
