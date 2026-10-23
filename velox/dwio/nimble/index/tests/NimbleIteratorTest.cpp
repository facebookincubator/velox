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
#include <folly/init/Init.h>

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

class NimbleIteratorTest : public NimbleIndexProjectorTest {};

TEST_P(NimbleIteratorTest, indexIterators) {
  registerDefaultOutputRowSerializers();
  registerDefaultOutputRowSerializers();
  auto rowType = ROW({
      {"key", BIGINT()},
      {"number", INTEGER()},
      {"score", REAL()},
      {"text", VARCHAR()},
  });
  constexpr uint32_t kNumRows{10};
  std::vector<int64_t> keys(kNumRows);
  std::vector<std::optional<int32_t>> numbers(kNumRows);
  std::vector<float> scores(kNumRows);
  std::vector<StringView> text(kNumRows);
  std::vector<std::string> textStorage(kNumRows);
  for (uint32_t row = 0; row < kNumRows; ++row) {
    keys[row] = row;
    if (row % 3 != 0) {
      numbers[row] = row * 10;
    }
    scores[row] = static_cast<float>(row) + 0.5f;
    textStorage[row] = fmt::format("value_{}", row);
    text[row] = StringView{textStorage[row]};
  }

  auto input = vectorMaker_->rowVector(
      {"key", "number", "score", "text"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVectorNullable<int32_t>(numbers),
       vectorMaker_->flatVector<float>(scores),
       vectorMaker_->flatVector<StringView>(text)});
  writeData({input}, {"key"});

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("number");
  projectedSubfields.emplace_back("score");
  projectedSubfields.emplace_back("text");
  const auto projectedRows = vectorMaker_->rowVector(
      {"number", "score", "text"},
      {vectorMaker_->flatVectorNullable<int32_t>(numbers),
       vectorMaker_->flatVector<float>(scores),
       vectorMaker_->flatVector<StringView>(text)});
  const auto bounds = makeRangeLookup(rowType, {"key"}, 2, 8);

  {
    TestNimbleIterator iterator{createBatchIterator(
        projectedSubfields, NimbleIterator::Options{.outputBatchRows = 3})};
    iterator.seek(bounds);
    velox::RowVectorPtr actual;
    velox::RowVectorPtr batch;
    while (iterator.next(batch)) {
      if (actual == nullptr) {
        actual = batch;
      } else {
        actual->append(batch.get());
      }
    }
    ASSERT_NE(actual, nullptr);
    expectVectorRows(actual, projectedRows->slice(2, 6));
    const NimbleIterator::Stats iteratorStats = iterator.stats();
    EXPECT_EQ(iteratorStats.numOutputRows, 6);
    EXPECT_EQ(iteratorStats.numSlicedStripes, 0);
  }

  for (const auto& format :
       {NimbleIterator::OutputRowFormat{NimbleIterator::kCompactRowFormat},
        NimbleIterator::OutputRowFormat{NimbleIterator::kUnsafeRowFormat}}) {
    SCOPED_TRACE(format);
    TestNimbleIterator iterator{createRowIterator(
        projectedSubfields,
        NimbleIterator::Options{
            .outputBatchRows = 3,
            .rowFormat = format,
        })};
    iterator.seek(bounds);
    EXPECT_EQ(iterator.stats().numReadStripes, 0);

    for (velox::vector_size_t row{2}; row < 8; ++row) {
      EXPECT_EQ(iterator.rowNumber(), row);
      const auto point = makePointLookup(rowType, {"key"}, row);
      ASSERT_TRUE(point.lowerKey.has_value());
      EXPECT_EQ(iterator.key(), *point.lowerKey);
      if (row == 2) {
        EXPECT_EQ(iterator.stats().numReadStripes, 0);
      }
      const auto value = iterator.value();
      EXPECT_GT(iterator.stats().numReadStripes, 0);

      velox::RowVectorPtr actual;
      if (format == NimbleIterator::kCompactRowFormat) {
        actual = velox::row::CompactRow::deserialize(
            {value}, velox::asRowType(projectedRows->type()), leafPool_.get());
      } else {
        std::vector<char*> rows{const_cast<char*>(value.data())};
        actual = velox::row::UnsafeRowFast::deserialize(
            rows, velox::asRowType(projectedRows->type()), leafPool_.get());
      }
      expectVectorRows(actual, projectedRows->slice(row, 1));
      iterator.next();
    }
  }
}

TEST_P(NimbleIteratorTest, fuzzesRowAndBatchIteratorMaterialization) {
  constexpr vector_size_t kNumRows{64};
  constexpr int kNumIterations{8};
  const auto seed = folly::Random::rand32();
  LOG(INFO) << "fuzzesRowAndBatchIteratorMaterialization seed: " << seed;
  folly::detail::DefaultGenerator random{seed};

  const auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  for (int iteration{0}; iteration < kNumIterations; ++iteration) {
    SCOPED_TRACE(fmt::format("seed={}, iteration={}", seed, iteration));
    VectorFuzzer fuzzer(
        {
            .vectorSize = kNumRows,
            .nullRatio = 0.25,
            .useRandomNullPattern = true,
        },
        leafPool_.get(),
        folly::Random::rand32(random));
    const auto input = vectorMaker_->rowVector(
        {"key", "value"},
        {vectorMaker_->flatVector<int64_t>(
             kNumRows, [](auto row) { return row; }),
         fuzzer.fuzzFlat(INTEGER(), kNumRows)});

    const auto rowsPerInputBatch =
        static_cast<vector_size_t>(1 + folly::Random::rand32(random) % 16);
    std::vector<RowVectorPtr> inputBatches;
    for (vector_size_t batchStart{0}; batchStart < kNumRows;
         batchStart += rowsPerInputBatch) {
      const auto numRows = std::min(rowsPerInputBatch, kNumRows - batchStart);
      inputBatches.push_back(vectorMaker_->rowVector(
          {"key", "value"},
          {input->childAt(0)->slice(batchStart, numRows),
           input->childAt(1)->slice(batchStart, numRows)}));
    }
    writeData(inputBatches, {"key"}, {}, /*stripeSize=*/1);

    const auto outputBatchRows =
        static_cast<uint32_t>(1 + folly::Random::rand32(random) % 16);
    const auto rangeStart = static_cast<vector_size_t>(
        folly::Random::rand32(random) % (kNumRows - 1));
    const auto rangeEnd = static_cast<vector_size_t>(
        rangeStart + 1 +
        folly::Random::rand32(random) % (kNumRows - rangeStart));
    const auto bounds = makeRangeLookup(rowType, {"key"}, rangeStart, rangeEnd);
    const auto expected = vectorMaker_->rowVector(
        {"value"},
        {input->childAt(1)->slice(rangeStart, rangeEnd - rangeStart)});

    {
      auto options = NimbleIterator::Options{};
      options.outputBatchRows = outputBatchRows;
      TestNimbleIterator iterator{
          createBatchIterator(projectedSubfields, options)};
      iterator.seek(bounds);
      RowVectorPtr actual;
      RowVectorPtr batch;
      while (iterator.next(batch)) {
        if (actual == nullptr) {
          actual = batch;
        } else {
          actual->append(batch.get());
        }
      }
      ASSERT_NE(actual, nullptr);
      expectVectorRows(actual, expected);
    }

    const bool useCompactRow = iteration % 2 == 0;
    const NimbleIterator::OutputRowFormat rowFormat = useCompactRow
        ? NimbleIterator::OutputRowFormat{NimbleIterator::kCompactRowFormat}
        : NimbleIterator::OutputRowFormat{NimbleIterator::kUnsafeRowFormat};
    auto options = NimbleIterator::Options{};
    options.outputBatchRows = outputBatchRows;
    options.rowFormat = rowFormat;
    TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
    iterator.seek(bounds);
    for (auto row = rangeStart; row < rangeEnd; ++row) {
      const bool skipValue =
          row != rangeStart && folly::Random::oneIn(3, random);
      if (skipValue) {
        iterator.next();
        continue;
      }

      const auto value = iterator.value();
      RowVectorPtr actual;
      if (useCompactRow) {
        actual = velox::row::CompactRow::deserialize(
            {value}, velox::asRowType(expected->type()), leafPool_.get());
      } else {
        std::vector<char*> serializedRows{const_cast<char*>(value.data())};
        actual = velox::row::UnsafeRowFast::deserialize(
            serializedRows,
            velox::asRowType(expected->type()),
            leafPool_.get());
      }
      expectVectorRows(actual, expected->slice(row - rangeStart, 1));
      iterator.next();
    }
  }
}

TEST_P(NimbleIteratorTest, indexIteratorRejectsNullProjector) {
  for (const bool serializedRows : {false, true}) {
    SCOPED_TRACE(serializedRows);
    try {
      auto iterator = serializedRows
          ? NimbleIterator::createRowIterator(nullptr, leafPool_.get(), {})
          : NimbleIterator::createBatchIterator(nullptr, leafPool_.get(), {});
      FAIL() << "Expected null projector rejection";
    } catch (const velox::VeloxRuntimeError& error) {
      EXPECT_THAT(
          std::string{error.what()}, ::testing::HasSubstr("pointer.get()"));
    }
  }
}

DEBUG_ONLY_TEST_P(NimbleIteratorTest, indexIteratorReportsStatsDuringPrefetch) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const bool wasTestValueEnabled =
      velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto resetTestValue = folly::makeGuard([&] {
    if (!wasTestValueEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  auto iterator = NimbleIterator::createBatchIterator(
      createProjectorWithFileHandle(
          projectedSubfields,
          makeFileHandle(),
          /*setDataIoStats=*/true,
          /*setMetadataIoStats=*/true,
          /*setIndexIoStats=*/true,
          /*cacheData=*/false),
      leafPool_.get(),
      NimbleIterator::Options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 1,
      });
  std::promise<void> loadStarted;
  auto loadStartedFuture = loadStarted.get_future();
  std::promise<void> continueLoad;
  const auto continueLoadFuture = continueLoad.get_future().share();
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::DirectDataInput::allocateBuffer",
      std::function<void(uint64_t*)>([&](uint64_t*) {
        loadStarted.set_value();
        continueLoadFuture.wait();
      }));
  bool loadReleased{false};
  const auto releaseLoad = [&] {
    if (!loadReleased) {
      loadReleased = true;
      continueLoad.set_value();
    }
  };
  std::promise<void> statsStarted;
  auto statsStartedFuture = statsStarted.get_future();
  std::future<NimbleIterator::Stats> pendingStats;
  auto cleanup = folly::makeGuard([&] {
    releaseLoad();
    if (pendingStats.valid()) {
      pendingStats.wait();
    }
    iterator->reset();
    iterator.reset();
    executor->join();
  });
  iterator->seekToFirst();
  ASSERT_EQ(
      loadStartedFuture.wait_for(std::chrono::seconds{5}),
      std::future_status::ready);
  pendingStats = std::async(std::launch::async, [&] {
    statsStarted.set_value();
    return iterator->stats();
  });
  ASSERT_EQ(
      statsStartedFuture.wait_for(std::chrono::seconds{5}),
      std::future_status::ready);
  ASSERT_EQ(
      pendingStats.wait_for(std::chrono::seconds{5}),
      std::future_status::ready);
  const auto partialStats = pendingStats.get();
  EXPECT_EQ(partialStats.numReadStripes, 0);
  EXPECT_GT(partialStats.numPlannedBytes, 0);
  EXPECT_EQ(dataIoStats_->read().count(), 0);

  releaseLoad();
  iterator->reset();
  const auto completedStats = iterator->stats();
  EXPECT_GT(completedStats.numPlannedBytes, 0);
  EXPECT_EQ(completedStats.numReadStripes, 1);
  EXPECT_EQ(dataIoStats_->read().count(), 1);
}

TEST_P(NimbleIteratorTest, indexIteratorCancelsPrefetchOnDestruction) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  for (const bool serializedRows : {false, true}) {
    SCOPED_TRACE(fmt::format("serializedRows={}", serializedRows));
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    std::unique_ptr<NimbleIterator> iterator;
    {
      const NimbleIterator::Options options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 1,
      };
      iterator = serializedRows
          ? createRowIterator(projectedSubfields, options)
          : createBatchIterator(projectedSubfields, options);
    }
    iterator->seekToFirst();

    auto producerPaused = std::make_shared<std::promise<void>>();
    auto producerPausedFuture = producerPaused->get_future();
    executor->add([producerPaused] { producerPaused->set_value(); });
    ASSERT_EQ(
        producerPausedFuture.wait_for(std::chrono::seconds{5}),
        std::future_status::ready);
    EXPECT_EQ(iterator->stats().numReadStripes, 1);

    iterator.reset();
    executor->join();
    EXPECT_EQ(dataIoStats_->read().count(), 1);
  }
}

DEBUG_ONLY_TEST_P(NimbleIteratorTest, indexIteratorCancelsPrefetchBeforeLoad) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const bool wasTestValueEnabled =
      velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto resetTestValue = folly::makeGuard([&] {
    if (!wasTestValueEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });

  for (const std::string pausePoint :
       {"facebook::nimble::ScanPrefetcher::run::afterReservation",
        "facebook::nimble::ScanPrefetcher::run::beforeLoad"}) {
    SCOPED_TRACE(pausePoint);
    for (const bool serializedRows : {false, true}) {
      SCOPED_TRACE(fmt::format("serializedRows={}", serializedRows));
      auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
      std::unique_ptr<NimbleIterator> iterator;
      {
        const NimbleIterator::Options options{
            .prefetchExecutor = folly::getKeepAliveToken(*executor),
            .maxPrefetchStripes = 1,
        };
        iterator = serializedRows
            ? createRowIterator(projectedSubfields, options)
            : createBatchIterator(projectedSubfields, options);
      }
      SCOPED_TESTVALUE_SET(
          pausePoint,
          std::function<void(std::atomic_bool*)>([](std::atomic_bool* stopped) {
            stopped->store(true, std::memory_order_release);
          }));
      auto cleanup = folly::makeGuard([&] {
        iterator->reset();
        iterator.reset();
        executor->join();
      });

      iterator->seekToFirst();
      std::promise<void> producerDone;
      auto producerDoneFuture = producerDone.get_future();
      executor->add([&] { producerDone.set_value(); });
      ASSERT_EQ(
          producerDoneFuture.wait_for(std::chrono::seconds{5}),
          std::future_status::ready);
      EXPECT_EQ(iterator->stats().numReadStripes, 0);
      EXPECT_EQ(dataIoStats_->read().count(), 0);
    }
  }
}

DEBUG_ONLY_TEST_P(
    NimbleIteratorTest,
    indexIteratorStopsPrefetchBeforeAsyncCleanup) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const bool wasTestValueEnabled =
      velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto resetTestValue = folly::makeGuard([&] {
    if (!wasTestValueEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });

  for (const bool serializedRows : {false, true}) {
    SCOPED_TRACE(fmt::format("serializedRows={}", serializedRows));
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    std::unique_ptr<NimbleIterator> iterator;
    {
      const NimbleIterator::Options options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 1,
      };
      iterator = serializedRows
          ? createRowIterator(projectedSubfields, options)
          : createBatchIterator(projectedSubfields, options);
    }
    std::promise<std::atomic_bool*> producerPaused;
    auto producerPausedFuture = producerPaused.get_future();
    std::promise<void> continueProducer;
    const auto continueProducerFuture = continueProducer.get_future().share();
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::ScanPrefetcher::run::afterReservation",
        std::function<void(std::atomic_bool*)>([&](std::atomic_bool* stopped) {
          producerPaused.set_value(stopped);
          continueProducerFuture.wait();
        }));
    bool producerReleased{false};
    const auto releaseProducer = [&] {
      if (!producerReleased) {
        producerReleased = true;
        continueProducer.set_value();
      }
    };
    auto cleanup = folly::makeGuard([&] {
      releaseProducer();
      iterator.reset();
      executor->join();
    });

    iterator->seekToFirst();
    ASSERT_EQ(
        producerPausedFuture.wait_for(std::chrono::seconds{5}),
        std::future_status::ready);
    const auto* stopped = producerPausedFuture.get();
    EXPECT_FALSE(stopped->load(std::memory_order_acquire));
    iterator.reset();
    EXPECT_TRUE(stopped->load(std::memory_order_acquire));
  }
}

DEBUG_ONLY_TEST_P(NimbleIteratorTest, indexIteratorCancelsPrefetchDuringLoad) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const bool wasTestValueEnabled =
      velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto resetTestValue = folly::makeGuard([&] {
    if (!wasTestValueEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });

  for (const bool serializedRows : {false, true}) {
    SCOPED_TRACE(fmt::format("serializedRows={}", serializedRows));
    std::promise<void> loadStarted;
    auto loadStartedFuture = loadStarted.get_future();
    std::promise<void> continueLoad;
    const auto continueLoadFuture = continueLoad.get_future().share();
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    auto projector = createProjectorWithFileHandle(
        projectedSubfields,
        makeFileHandle(),
        /*setDataIoStats=*/true,
        /*setMetadataIoStats=*/true,
        /*setIndexIoStats=*/true,
        /*cacheData=*/false);
    std::unique_ptr<NimbleIterator> iterator;
    {
      const NimbleIterator::Options options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 1,
      };
      iterator = serializedRows
          ? NimbleIterator::createRowIterator(
                std::move(projector), leafPool_.get(), options)
          : NimbleIterator::createBatchIterator(
                std::move(projector), leafPool_.get(), options);
    }
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::DirectDataInput::allocateBuffer",
        std::function<void(uint64_t*)>([&](uint64_t* /*bytes*/) {
          loadStarted.set_value();
          continueLoadFuture.wait();
        }));
    bool loadReleased{false};
    const auto releaseLoad = [&] {
      if (!loadReleased) {
        loadReleased = true;
        continueLoad.set_value();
      }
    };
    // Release the blocked load before joining, including assertion failures.
    auto cleanup = folly::makeGuard([&] {
      releaseLoad();
      iterator.reset();
      executor->join();
    });

    iterator->seekToFirst();
    ASSERT_EQ(
        loadStartedFuture.wait_for(std::chrono::seconds{5}),
        std::future_status::ready);
    EXPECT_EQ(dataIoStats_->read().count(), 0);

    // Release the iterator while the producer is loading a stripe.
    iterator.reset();
    releaseLoad();
    executor->join();
    cleanup.dismiss();
    EXPECT_EQ(dataIoStats_->read().count(), 1);
  }
}

TEST_P(NimbleIteratorTest, indexIteratorRepeatsPrefetchLoadErrors) {
  const auto input = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>({0, 1, 2, 3}),
       vectorMaker_->flatVector<int32_t>({0, 10, 20, 30})});
  WriterOptions writerOptions;
  writerOptions.encodingSelectionPolicyCreator = encodingViewSelectionPolicy();
  writerOptions.clusterIndexConfig = makeClusterIndexConfig({"key"});
  writerOptions.enableStreamChecksums = true;
  writeBatches({input}, std::move(writerOptions));
  corruptProjectedStreams();
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  for (const bool serializedRows : {false, true}) {
    SCOPED_TRACE(fmt::format("serializedRows={}", serializedRows));
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    auto projector = createProjectorWithFileHandle(
        projectedSubfields,
        makeFileHandle(),
        /*setDataIoStats=*/true,
        /*setMetadataIoStats=*/true,
        /*setIndexIoStats=*/true,
        /*cacheData=*/false,
        /*verifyStreamChecksums=*/true);
    const NimbleIterator::Options options{
        .prefetchExecutor = folly::getKeepAliveToken(*executor),
        .maxPrefetchStripes = 1,
    };
    auto iterator = serializedRows
        ? NimbleIterator::createRowIterator(
              std::move(projector), leafPool_.get(), options)
        : NimbleIterator::createBatchIterator(
              std::move(projector), leafPool_.get(), options);
    iterator->seekToFirst();
    const auto materialize = [&] {
      if (serializedRows) {
        iterator->value();
        return;
      }
      velox::RowVectorPtr output;
      folly::coro::blockingWait(
          folly::coro::timeoutNoDiscard(
              iterator->co_nextBatch(output), std::chrono::seconds{5}));
    };

    NIMBLE_ASSERT_THROW(materialize(), "Stream checksum mismatch");
    const auto numDataReads = dataIoStats_->read().count();
    NIMBLE_ASSERT_THROW(materialize(), "Stream checksum mismatch");
    EXPECT_EQ(dataIoStats_->read().count(), numDataReads);
    EXPECT_EQ(iterator->rowNumber(), 0);
    iterator->reset();
  }
}

TEST_P(NimbleIteratorTest, commonIteratorInterfaceNavigatesBothFlavors) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/2);
  const auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  const auto bounds = makeRangeLookup(rowType, {"key"}, 2, 6);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator batchIterator{createBatchIterator(
      projectedSubfields, NimbleIterator::Options{.outputBatchRows = 3})};
  TestNimbleIterator rowIterator{createRowIterator(
      projectedSubfields, NimbleIterator::Options{.outputBatchRows = 3})};

  for (TestNimbleIterator* iterator : {&batchIterator, &rowIterator}) {
    iterator->seek(bounds);
    EXPECT_TRUE(iterator->valid());
    EXPECT_EQ(iterator->rowNumber(), 2);
    iterator->seekToFirst();
    EXPECT_TRUE(iterator->valid());
    EXPECT_EQ(iterator->rowNumber(), 0);
    EXPECT_EQ(iterator->stats().numReadStripes, 0);
    iterator->seek(makePointLookup(rowType, {"key"}, 100));
    EXPECT_FALSE(iterator->valid());
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllCacheParams,
    NimbleIteratorTest,
    ::testing::ValuesIn(NimbleIteratorTest::getIteratorTestParams()),
    NimbleIteratorTest::iteratorTestParamName);

} // namespace facebook::nimble::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, /*removeFlags=*/false};
  return RUN_ALL_TESTS();
}
