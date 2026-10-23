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

class NimbleBatchIteratorTest : public NimbleIndexProjectorTest {};

TEST_P(NimbleBatchIteratorTest, batchIteratorStopsAtStripeBoundaries) {
  constexpr uint32_t kRowsPerStripe{4};
  constexpr uint32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = std::numeric_limits<uint32_t>::max(),
      })};
  iterator.seekToFirst();

  std::vector<velox::vector_size_t> batchSizes;
  velox::RowVectorPtr batch;
  while (iterator.next(batch)) {
    batchSizes.push_back(batch->size());
  }

  EXPECT_EQ(
      batchSizes,
      std::vector<velox::vector_size_t>(kNumStripes, kRowsPerStripe));
  EXPECT_EQ(iterator.stats().numReadStripes, kNumStripes);
  EXPECT_EQ(iterator.stats().numMaterializedRows, kRowsPerStripe * kNumStripes);
}

TEST_P(NimbleBatchIteratorTest, batchIteratorRejectsRowAccess) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{
      createBatchIterator(projectedSubfields, NimbleIterator::Options{})};
  iterator.seekToFirst();

  NIMBLE_ASSERT_THROW(
      iterator.next(), "Batch iterator cannot produce serialized row output");
  NIMBLE_ASSERT_THROW(
      iterator.prepareValue(),
      "Batch iterator does not expose serialized row values");
  NIMBLE_ASSERT_THROW(
      iterator.value(), "Batch iterator does not expose serialized row values");
}

TEST_P(NimbleBatchIteratorTest, batchIteratorSlicesStripeStreams) {
  constexpr uint32_t kRowsPerStripe{4};
  constexpr uint32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = 3,
      })};
  iterator.seekToFirst();

  std::vector<velox::vector_size_t> batchSizes;
  std::vector<int32_t> actualValues;
  velox::RowVectorPtr batch;
  while (iterator.next(batch)) {
    batchSizes.push_back(batch->size());
    const auto* values = batch->childAt(0)->asFlatVector<int32_t>();
    ASSERT_NE(values, nullptr);
    for (velox::vector_size_t row{0}; row < batch->size(); ++row) {
      actualValues.push_back(values->valueAt(row));
    }
  }

  EXPECT_EQ(batchSizes, (std::vector<velox::vector_size_t>{3, 1, 3, 1, 3, 1}));
  EXPECT_EQ(
      actualValues,
      (std::vector<int32_t>{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110}));
  EXPECT_EQ(iterator.stats().numReadStripes, kNumStripes);
}

TEST_P(NimbleBatchIteratorTest, batchIteratorRetainsStringBatchesAcrossScans) {
  constexpr velox::vector_size_t kRowsPerStripe{4};
  constexpr velox::vector_size_t kNumRows{8};
  constexpr uint32_t kOutputBatchRows{3};
  const auto rowType = ROW({{"key", BIGINT()}, {"value", VARCHAR()}});
  const auto input = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(
           kNumRows, [](auto row) { return row; }),
       vectorMaker_->flatVectorNullable<StringView>({
           "first_stripe_first_long_string",
           std::nullopt,
           "first_stripe_third_long_string",
           "first_stripe_fourth_long_string",
           "second_stripe_first_long_string",
           "second_stripe_second_long_string",
           std::nullopt,
           "second_stripe_fourth_long_string",
       })});
  std::vector<RowVectorPtr> inputBatches;
  for (velox::vector_size_t row{0}; row < kNumRows; row += kRowsPerStripe) {
    SCOPED_TRACE(fmt::format("inputRow={}", row));
    inputBatches.push_back(vectorMaker_->rowVector(
        {"key", "value"},
        {input->childAt(0)->slice(row, kRowsPerStripe),
         input->childAt(1)->slice(row, kRowsPerStripe)}));
  }
  writeData(inputBatches, {"key"}, {}, /*stripeSize=*/1);
  const auto expected = vectorMaker_->rowVector({"value"}, {input->childAt(1)});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto iterator = createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{.outputBatchRows = kOutputBatchRows});
  iterator->seekToFirst();

  RowVectorPtr output;
  ASSERT_TRUE(folly::coro::blockingWait(iterator->co_nextBatch(output)));
  ASSERT_EQ(output->size(), kOutputBatchRows);
  const auto retainedBatch = output;
  const auto retainedChild = output->childAt(0);
  velox::vector_size_t nextRow{output->size()};
  while (folly::coro::blockingWait(iterator->co_nextBatch(output))) {
    SCOPED_TRACE(fmt::format("row={}", nextRow));
    velox::test::assertEqualVectors(
        expected->slice(nextRow, output->size()), output);
    nextRow += output->size();
  }
  EXPECT_EQ(nextRow, kNumRows);
  EXPECT_EQ(output, nullptr);

  iterator->seek(makeRangeLookup(rowType, {"key"}, 4, 6));
  ASSERT_TRUE(folly::coro::blockingWait(iterator->co_nextBatch(output)));
  velox::test::assertEqualVectors(expected->slice(4, 2), output);
  iterator->reset();
  EXPECT_FALSE(iterator->valid());
  EXPECT_FALSE(folly::coro::blockingWait(iterator->co_nextBatch(output)));
  EXPECT_EQ(output, nullptr);
  iterator.reset();

  velox::test::assertEqualVectors(
      expected->slice(0, kOutputBatchRows), retainedBatch);
  velox::test::assertEqualVectors(
      expected->childAt(0)->slice(0, kOutputBatchRows), retainedChild);
}

TEST_P(NimbleBatchIteratorTest, batchIteratorSeeksToFinalRowByKey) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/2);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = 3,
      })};

  iterator.seek(makeRangeLookup(rowType, {"key"}, 7, 8));
  EXPECT_EQ(iterator.rowNumber(), 7);
  velox::RowVectorPtr batch;
  ASSERT_TRUE(iterator.next(batch));
  const auto* values = batch->childAt(0)->asFlatVector<int32_t>();
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(batch->size(), 1);
  EXPECT_EQ(values->valueAt(0), 70);
  EXPECT_FALSE(iterator.next(batch));
  EXPECT_EQ(iterator.stats().numReadStripes, 1);
}

TEST_P(NimbleBatchIteratorTest, batchIteratorSeeksPoint) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/2);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = 3,
      })};

  iterator.seek(*makePointLookup(rowType, {"key"}, 7).lowerKey);
  EXPECT_EQ(iterator.rowNumber(), 7);
  velox::RowVectorPtr batch;
  ASSERT_TRUE(iterator.next(batch));
  const auto* values = batch->childAt(0)->asFlatVector<int32_t>();
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(batch->size(), 1);
  EXPECT_EQ(values->valueAt(0), 70);
  EXPECT_FALSE(iterator.next(batch));
  EXPECT_EQ(iterator.stats().numMaterializedRows, 1);

  // A key without rows leaves the iterator invalid.
  iterator.seek(*makePointLookup(rowType, {"key"}, 1'000).lowerKey);
  EXPECT_FALSE(iterator.valid());
}

TEST_P(
    NimbleBatchIteratorTest,
    batchIteratorTimesMaterializationOnlyWhenEnabled) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/2);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  for (const bool collectTimings : {true, false}) {
    SCOPED_TRACE(collectTimings);
    NimbleIterator::Options options;
    options.collectTimings = collectTimings;
    TestNimbleIterator iterator{
        createBatchIterator(projectedSubfields, options)};
    iterator.seekToFirst();
    EXPECT_EQ(iterator.stats().materializationTiming, velox::CpuWallTiming{});
    velox::RowVectorPtr batch;
    ASSERT_TRUE(iterator.next(batch));
    const auto timing = iterator.stats().materializationTiming;
    if (collectTimings) {
      EXPECT_EQ(timing.count, 1);
    } else {
      EXPECT_EQ(timing, velox::CpuWallTiming{});
    }
  }
}

TEST_P(NimbleBatchIteratorTest, batchIteratorReadsAheadAndReseeks) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = 3,
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
      })};

  iterator.seekToFirst();
  velox::RowVectorPtr batch;
  ASSERT_TRUE(iterator.next(batch));
  EXPECT_EQ(batch->size(), 3);

  iterator.seek(makeRangeLookup(rowType, {"key"}, 5, 9));
  std::vector<int32_t> actualValues;
  while (iterator.next(batch)) {
    const auto* values = batch->childAt(0)->asFlatVector<int32_t>();
    ASSERT_NE(values, nullptr);
    for (velox::vector_size_t row{0}; row < batch->size(); ++row) {
      actualValues.push_back(values->valueAt(row));
    }
  }
  EXPECT_EQ(actualValues, (std::vector<int32_t>{50, 60, 70, 80}));
}

TEST_P(
    NimbleBatchIteratorTest,
    batchIteratorDisablesPrefetchWithZeroStripeLimit) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = 3,
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 0,
      })};

  iterator.seekToFirst();
  EXPECT_EQ(iterator.stats().numReadStripes, 0);

  velox::RowVectorPtr batch;
  ASSERT_TRUE(iterator.next(batch));
  EXPECT_EQ(batch->size(), 3);
  EXPECT_EQ(iterator.stats().numReadStripes, 1);
}

TEST_P(NimbleBatchIteratorTest, emptySeekWithConfiguredPrefetch) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  auto iterator = createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = 4,
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 3,
      });
  iterator->seek(makeRangeLookup(rowType, {"key"}, 50, 60));
  EXPECT_FALSE(iterator->valid());
  EXPECT_EQ(iterator->stats().numReadStripes, 0);
  EXPECT_EQ(iterator->stats().numSkippedStripes, 0);
  RowVectorPtr output;
  EXPECT_FALSE(folly::coro::blockingWait(iterator->co_nextBatch(output)));
  EXPECT_EQ(output, nullptr);

  iterator->seekToFirst();
  ASSERT_TRUE(folly::coro::blockingWait(iterator->co_nextBatch(output)));
  EXPECT_EQ(output->size(), 4);
  EXPECT_EQ(iterator->stats().numSkippedStripes, 0);
  iterator->reset();
}

TEST_P(NimbleBatchIteratorTest, batchIteratorBoundsReadAheadByStripeCount) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/6);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  for (const uint32_t maxPrefetchStripes : {1, 2, 3, 4}) {
    SCOPED_TRACE(fmt::format("maxPrefetchStripes={}", maxPrefetchStripes));
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    std::promise<void> readAheadPaused;
    auto readAheadPausedFuture = readAheadPaused.get_future();
    TestNimbleIterator iterator{createBatchIterator(
        projectedSubfields,
        NimbleIterator::Options{
            .outputBatchRows = 4,
            .prefetchExecutor = folly::getKeepAliveToken(*executor),
            .maxPrefetchStripes = maxPrefetchStripes,
        })};

    iterator.seekToFirst();
    executor->add([&readAheadPaused] { readAheadPaused.set_value(); });
    readAheadPausedFuture.wait();
    EXPECT_EQ(iterator.stats().numReadStripes, maxPrefetchStripes);
  }
}

TEST_P(NimbleBatchIteratorTest, batchIteratorPrefetchesSelectedStripeRanges) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = 4,
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 6,
      })};
  iterator.seek(makeRangeLookup(rowType, {"key"}, 2, 10));

  std::vector<velox::vector_size_t> batchSizes;
  std::vector<int32_t> actualValues;
  velox::RowVectorPtr batch;
  while (iterator.next(batch)) {
    batchSizes.push_back(batch->size());
    const auto* values = batch->childAt(0)->asFlatVector<int32_t>();
    ASSERT_NE(values, nullptr);
    for (velox::vector_size_t row = 0; row < batch->size(); ++row) {
      actualValues.push_back(values->valueAt(row));
    }
  }
  EXPECT_THAT(batchSizes, ::testing::ElementsAre(2, 4, 2));
  EXPECT_THAT(
      actualValues, ::testing::ElementsAre(20, 30, 40, 50, 60, 70, 80, 90));
  EXPECT_EQ(iterator.stats().numOutputRows, 8);
  EXPECT_EQ(iterator.stats().numReadStripes, 3);
  EXPECT_EQ(dataIoStats_->read().count(), 1);
}

TEST_P(
    NimbleBatchIteratorTest,
    batchIteratorRetainsSharedBatchStripeReservation) {
  constexpr uint32_t kRowsPerStripe{4};
  constexpr uint32_t kNumStripes{6};
  constexpr uint32_t kStripesPerBatch{2};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = kRowsPerStripe,
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 2 * kStripesPerBatch,
      })};
  const auto waitForPrefetch = [&] {
    std::promise<void> readAheadPaused;
    auto readAheadPausedFuture = readAheadPaused.get_future();
    executor->add([&readAheadPaused] { readAheadPaused.set_value(); });
    readAheadPausedFuture.wait();
  };

  iterator.seekToFirst();
  waitForPrefetch();
  EXPECT_EQ(iterator.stats().numReadStripes, 4);
  EXPECT_EQ(dataIoStats_->read().count(), 2);

  velox::RowVectorPtr batch;
  const auto expectStripe = [&](uint32_t stripeIndex) {
    SCOPED_TRACE(stripeIndex);
    const auto expected = vectorMaker_->rowVector(
        {"value"},
        {vectorMaker_->flatVector<int32_t>(kRowsPerStripe, [&](auto row) {
          return (stripeIndex * kRowsPerStripe + row) * 10;
        })});
    velox::test::assertEqualVectors(expected, batch);
  };
  ASSERT_TRUE(iterator.next(batch));
  expectStripe(0);
  waitForPrefetch();
  EXPECT_EQ(iterator.stats().numReadStripes, 4);
  EXPECT_EQ(dataIoStats_->read().count(), 2);

  ASSERT_TRUE(iterator.next(batch));
  expectStripe(1);
  waitForPrefetch();
  EXPECT_EQ(iterator.stats().numReadStripes, kNumStripes);
  EXPECT_EQ(dataIoStats_->read().count(), 3);
  for (uint32_t stripeIndex{2}; stripeIndex < kNumStripes; ++stripeIndex) {
    SCOPED_TRACE(stripeIndex);
    ASSERT_TRUE(iterator.next(batch));
    expectStripe(stripeIndex);
  }
  EXPECT_FALSE(iterator.next(batch));
  EXPECT_EQ(iterator.stats().numOutputRows, kNumStripes * kRowsPerStripe);
}

TEST_P(
    NimbleBatchIteratorTest,
    batchIteratorPreservesNonUniqueKeyPositionAcrossStripes) {
  constexpr uint32_t kRowsPerStripe{4};
  constexpr uint32_t kNumStripes{3};
  std::vector<RowVectorPtr> batches;
  for (uint32_t stripe{0}; stripe < kNumStripes; ++stripe) {
    const auto firstRow = stripe * kRowsPerStripe;
    batches.emplace_back(vectorMaker_->rowVector(
        {"key", "value"},
        {vectorMaker_->flatVector<int64_t>(
             kRowsPerStripe,
             [firstRow](auto row) { return (firstRow + row) / 5; }),
         vectorMaker_->flatVector<int32_t>(
             kRowsPerStripe,
             [firstRow](auto row) { return firstRow + row; })}));
  }
  WriterOptions writerOptions;
  writerOptions.encodingSelectionPolicyCreator = encodingViewSelectionPolicy();
  writerOptions.clusterIndexConfig = ClusterIndexConfigBuilder{}
                                         .withKeyColumns({"key"})
                                         .withSortOrders({{.ascending = true}})
                                         .withEnforceKeyOrder(true)
                                         .withNoDuplicateKey(false)
                                         .build();
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        [](const StripeProgress&) { return true; });
  };
  writeBatches(batches, std::move(writerOptions));

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{createBatchIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .outputBatchRows = std::numeric_limits<uint32_t>::max(),
      })};
  iterator.seekToFirst();

  std::vector<velox::vector_size_t> batchSizes;
  std::vector<int32_t> actualValues;
  velox::RowVectorPtr batch;
  while (iterator.next(batch)) {
    batchSizes.push_back(batch->size());
    const auto* values = batch->childAt(0)->asFlatVector<int32_t>();
    ASSERT_NE(values, nullptr);
    for (velox::vector_size_t row{0}; row < batch->size(); ++row) {
      actualValues.push_back(values->valueAt(row));
    }
  }

  EXPECT_EQ(
      batchSizes,
      std::vector<velox::vector_size_t>(kNumStripes, kRowsPerStripe));
  EXPECT_EQ(
      actualValues,
      (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  EXPECT_EQ(iterator.stats().numReadStripes, kNumStripes);
  EXPECT_EQ(iterator.stats().numReadRows, actualValues.size());
}

INSTANTIATE_TEST_SUITE_P(
    AllCacheParams,
    NimbleBatchIteratorTest,
    ::testing::ValuesIn(NimbleBatchIteratorTest::getIteratorTestParams()),
    NimbleBatchIteratorTest::iteratorTestParamName);

} // namespace facebook::nimble::test
