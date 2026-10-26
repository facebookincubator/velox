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
#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "folly/CancellationToken.h"
#include "folly/ScopeGuard.h"
#include "folly/init/Init.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/IndexScanCursor.h"
#include "velox/dwio/nimble/selective/ReaderBase.h"

namespace facebook::nimble::test {

// Rejects one submission while retaining a working executor for the producer.
class RejectNextExecutor final : public folly::CPUThreadPoolExecutor {
 public:
  RejectNextExecutor() : folly::CPUThreadPoolExecutor(1) {}

  void rejectNext() {
    rejectNext_.store(true, std::memory_order_release);
  }

  void add(folly::Func operation) override {
    if (rejectNext_.exchange(false, std::memory_order_acq_rel)) {
      NIMBLE_FAIL("injected executor scheduling failure");
    }
    folly::CPUThreadPoolExecutor::add(std::move(operation));
  }

 private:
  std::atomic_bool rejectNext_{false};
};

// Exercises scan state and stripe loading through the cursor's own API.
class IndexScanCursorTest : public NimbleIndexProjectorTest {
 protected:
  // Takes 'options' by value so that a test can move its prefetch executor
  // token into the cursor. ThreadPoolExecutor::join() waits for every token, so
  // a token left in the test's own options would block joining the executor.
  std::shared_ptr<detail::IndexScanCursor> makeCursor(
      NimbleIterator::Options options = {}) {
    options.cacheEncodingView = GetParam().cacheEncodingView;
    std::vector<Subfield> subfields;
    subfields.emplace_back("value");
    return detail::IndexScanCursor::create(createProjector(subfields), options);
  }
};

TEST_P(IndexScanCursorTest, rejectsNullProjector) {
  try {
    auto cursor = detail::IndexScanCursor::create(nullptr, {});
    FAIL() << "Expected null projector rejection";
  } catch (const velox::VeloxRuntimeError& error) {
    EXPECT_THAT(
        std::string{error.what()}, ::testing::HasSubstr("pointer.get()"));
  }
}

TEST_P(IndexScanCursorTest, loadsSelectedStripesLazily) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  const auto range = cursor->seek(makeRangeLookup(rowType, {"key"}, 2, 10));
  EXPECT_EQ(range, std::optional<RowRange>(RowRange{2, 10}));
  EXPECT_EQ(cursor->stats().numReadStripes, 0);
  EXPECT_EQ(dataIoStats_->read().count(), 0);

  std::vector<RowRange> selectedRows;
  std::vector<uint32_t> stripeIndices;
  for (const uint32_t row : {2, 4, 8}) {
    SCOPED_TRACE(row);
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(row));
    selectedRows.push_back(stripe.selection.selectedRowRange());
    stripeIndices.push_back(stripe.selection.stripeIndex);
    EXPECT_NE(stripe.dataHandle, nullptr);
  }
  EXPECT_EQ(selectedRows, (std::vector<RowRange>{{2, 4}, {4, 8}, {8, 10}}));
  EXPECT_EQ(stripeIndices, (std::vector<uint32_t>{0, 1, 2}));
  EXPECT_EQ(cursor->stats().numReadStripes, 3);
  EXPECT_EQ(cursor->stats().numOutputRows, 8);
  cursor->reset();
}

TEST_P(IndexScanCursorTest, collectsTimingsOnlyWhenEnabled) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  for (const bool collectTimings : {true, false}) {
    SCOPED_TRACE(collectTimings);
    NimbleIterator::Options options;
    options.collectTimings = collectTimings;
    auto cursor = makeCursor(options);
    cursor->seek(makeRangeLookup(rowType, {"key"}, 2, 10));
    folly::coro::blockingWait(cursor->co_nextStripe(2));
    const auto stats = cursor->stats();
    EXPECT_EQ(stats.numReadStripes, 1);
    if (collectTimings) {
      EXPECT_GT(stats.lookupTiming.count, 0);
      EXPECT_GT(stats.scanTiming.count, 0);
    } else {
      for (const auto& timing :
           {stats.lookupTiming,
            stats.prepareTiming,
            stats.scanTiming,
            stats.projectionTiming}) {
        EXPECT_EQ(timing, velox::CpuWallTiming{});
      }
    }
    cursor->reset();
  }
}

TEST_P(IndexScanCursorTest, createsKeysWithinTheActiveRange) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  cursor->seek(makeRangeLookup(rowType, {"key"}, 2, 10));

  auto keys = cursor->keyCursor(5);
  for (int64_t row{5}; row < 10; ++row) {
    SCOPED_TRACE(row);
    ASSERT_TRUE(keys->hasNext());
    const auto expected = makePointLookup(rowType, {"key"}, row);
    ASSERT_TRUE(expected.lowerKey.has_value());
    EXPECT_EQ(keys->next(), *expected.lowerKey);
  }
  EXPECT_FALSE(keys->hasNext());
  EXPECT_EQ(cursor->stats().numReadStripes, 0);
  cursor->reset();
}

TEST_P(IndexScanCursorTest, rejectsRowsOutsideTheActiveScan) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  NIMBLE_ASSERT_THROW(
      folly::coro::blockingWait(cursor->co_nextStripe(0)),
      "No index scan is active");
  NIMBLE_ASSERT_THROW(cursor->keyCursor(0), "No index scan is active");

  cursor->seek(makeRangeLookup(rowType, {"key"}, 2, 10));
  for (const uint32_t row : {1U, 10U, std::numeric_limits<uint32_t>::max()}) {
    SCOPED_TRACE(row);
    NIMBLE_ASSERT_THROW(
        folly::coro::blockingWait(cursor->co_nextStripe(row)),
        "Row is outside the active scan");
    NIMBLE_ASSERT_THROW(
        cursor->keyCursor(row), "Row is outside the active scan");
  }
  cursor->reset();
  NIMBLE_ASSERT_THROW(cursor->keyCursor(2), "No index scan is active");
}

TEST_P(IndexScanCursorTest, clearsMissesAndSupportsReseek) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  EXPECT_FALSE(cursor->seek(makeRangeLookup(rowType, {"key"}, 50, 60)));
  NIMBLE_ASSERT_THROW(cursor->keyCursor(0), "No index scan is active");

  EXPECT_EQ(cursor->seekToFirst(), std::optional<RowRange>(RowRange{0, 12}));
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(0));
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(0, 4));
  }
  cursor->seek(makeRangeLookup(rowType, {"key"}, 6, 11));
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(6));
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(6, 8));
  }
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(8));
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(8, 11));
  }
  cursor->reset();
  EXPECT_EQ(cursor->stats().numReadStripes, 3);
  EXPECT_EQ(cursor->stats().numOutputRows, 9);
}

TEST_P(IndexScanCursorTest, skipsQueuedPrefetchStripes) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 3;
  auto cursor = makeCursor(options);
  cursor->seekToFirst();
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(8));
    EXPECT_EQ(stripe.selection.stripeIndex, 2);
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(8, 12));
  }
  cursor->reset();
  EXPECT_EQ(cursor->stats().numReadStripes, 3);
  EXPECT_EQ(cursor->stats().numSkippedStripes, 2);
  EXPECT_EQ(cursor->stats().numOutputRows, 4);
}

TEST_P(IndexScanCursorTest, zeroStripeLimitKeepsLoadsOnDemand) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 0;
  auto cursor = makeCursor(options);
  cursor->seekToFirst();
  EXPECT_EQ(cursor->stats().numReadStripes, 0);
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(8));
    EXPECT_EQ(stripe.selection.stripeIndex, 2);
  }
  EXPECT_EQ(cursor->stats().numReadStripes, 1);
  EXPECT_EQ(cursor->stats().numSkippedStripes, 0);
  cursor->reset();
}

TEST_P(IndexScanCursorTest, returnsSchemaAndCallerScratchOptions) {
  const auto input = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>({1, 2, 3}),
       vectorMaker_->flatVector<int32_t>({10, 20, 30})});
  for (const bool compactRowCount : {false, true}) {
    SCOPED_TRACE(compactRowCount);
    writeData(
        {input},
        {"key"},
        {},
        /*stripeSize=*/4 << 10,
        /*enableStreamDeduplication=*/true,
        /*compactRowCountEncoding=*/compactRowCount);
    auto cursor = makeCursor();
    const auto& schema = cursor->projectedNimbleType()->asRow();
    ASSERT_EQ(schema.childrenCount(), 1);
    EXPECT_EQ(schema.nameAt(0), "value");
    velox::BufferPool scratch{velox::BufferPool::kDefaultCapacity};
    for (auto* bufferPool :
         {&scratch, static_cast<velox::BufferPool*>(nullptr)}) {
      SCOPED_TRACE(bufferPool);
      const auto options = cursor->encodingOptions(bufferPool);
      EXPECT_EQ(options.bufferPool, bufferPool);
      EXPECT_EQ(options.useVarintRowCount, compactRowCount);
      EXPECT_EQ(options.cacheEncodingView, GetParam().cacheEncodingView);
    }
  }
}

TEST_P(IndexScanCursorTest, releasesSelectiveReaderStripeOwnership) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/1);
  velox::dwio::common::ReaderOptions options(leafPool_.get());
  configureReaderOptions(
      options,
      /*setDataIoStats=*/true,
      /*setMetadataIoStats=*/true,
      /*setIndexIoStats=*/true,
      /*cacheData=*/false,
      /*verifyStreamChecksums=*/false);
  auto file = std::make_shared<InMemoryReadFile>(std::string_view(sinkData_));
  auto input =
      std::make_unique<velox::dwio::common::BufferedInput>(file, *leafPool_);
  auto readerBase = ReaderBase::create(std::move(input), options);
  std::weak_ptr<ReaderBase> owner = readerBase;
  {
    facebook::nimble::StripeStreams selectiveStreams{readerBase};
    readerBase.reset();
    selectiveStreams.setStripe(0);
    auto cursor = makeCursor();
    cursor->seekToFirst();
    const auto projected = folly::coro::blockingWait(cursor->co_nextStripe(0));
    EXPECT_EQ(projected.selection.selectedRowRange(), RowRange(0, 4));
    EXPECT_FALSE(owner.expired());
    cursor->reset();
  }
  EXPECT_TRUE(owner.expired());
}

TEST_P(IndexScanCursorTest, seeksFirstMiddleAndLastPoints) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  for (const uint32_t row : {0, 5, 11}) {
    SCOPED_TRACE(row);
    const auto range = cursor->seek(makePointLookup(rowType, {"key"}, row));
    EXPECT_EQ(range, std::optional<RowRange>(RowRange{row, row + 1}));
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(row));
    EXPECT_EQ(stripe.selection.stripeIndex, row / 4);
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(row, row + 1));
    cursor->reset();
  }
}

TEST_P(IndexScanCursorTest, seeksKeys) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  for (const uint32_t row : {0, 5, 11}) {
    SCOPED_TRACE(row);
    const auto range =
        cursor->seek(*makePointLookup(rowType, {"key"}, row).lowerKey);
    EXPECT_EQ(range, std::optional<RowRange>(RowRange{row, row + 1}));
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(row));
    EXPECT_EQ(stripe.selection.stripeIndex, row / 4);
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(row, row + 1));
    cursor->reset();
  }

  // A key without rows is a miss, and the cursor seeks again afterwards.
  EXPECT_EQ(
      cursor->seek(*makePointLookup(rowType, {"key"}, 1'000).lowerKey),
      std::nullopt);
  EXPECT_EQ(
      cursor->seek(*makePointLookup(rowType, {"key"}, 7).lowerKey),
      std::optional<RowRange>(RowRange{7, 8}));
  cursor->reset();
}

TEST_P(IndexScanCursorTest, handlesUnboundedAndEmptyKeyRanges) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  NIMBLE_ASSERT_THROW(
      cursor->seek(velox::serializer::EncodedKeyBounds{}),
      "At least one of lowerKey or upperKey must be set");
  EXPECT_EQ(cursor->seekToFirst(), std::optional<RowRange>(RowRange{0, 12}));
  auto lowerOnly = makePointLookup(rowType, {"key"}, 5);
  lowerOnly.upperKey.reset();
  EXPECT_EQ(cursor->seek(lowerOnly), std::optional<RowRange>(RowRange{5, 12}));
  auto upperOnly = makePointLookup(rowType, {"key"}, 5);
  upperOnly.lowerKey.reset();
  EXPECT_EQ(cursor->seek(upperOnly), std::optional<RowRange>(RowRange{0, 6}));
  EXPECT_FALSE(cursor->seek(makeRangeLookup(rowType, {"key"}, 5, 5)));
  cursor->reset();
  cursor->reset();
}

TEST_P(IndexScanCursorTest, rejectsConsumedAndBackwardStripesWithoutDequeuing) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    NimbleIterator::Options options;
    if (prefetch) {
      options.prefetchExecutor = folly::getKeepAliveToken(*executor);
    }
    auto cursor = makeCursor(options);
    cursor->seekToFirst();
    {
      const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(4));
      EXPECT_EQ(stripe.selection.stripeIndex, 1);
    }
    for (const uint32_t row : {0U, 4U, 7U}) {
      SCOPED_TRACE(row);
      NIMBLE_ASSERT_THROW(
          folly::coro::blockingWait(cursor->co_nextStripe(row)),
          "Row is outside the selected stripe");
    }
    {
      const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(8));
      EXPECT_EQ(stripe.selection.stripeIndex, 2);
    }
    NIMBLE_ASSERT_THROW(
        folly::coro::blockingWait(cursor->co_nextStripe(8)),
        "No selected stripes remain");
    cursor->reset();
  }
}

TEST_P(IndexScanCursorTest, boundsPrefetchForAllLimitModes) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/6);
  for (const uint32_t limit : {0, 1, 2, 3, 4, 10}) {
    SCOPED_TRACE(limit);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    NimbleIterator::Options options;
    options.prefetchExecutor = folly::getKeepAliveToken(*executor);
    options.maxPrefetchStripes = limit;
    auto cursor = makeCursor(options);
    cursor->seekToFirst();
    std::promise<void> producerPaused;
    auto paused = producerPaused.get_future();
    executor->add([&] { producerPaused.set_value(); });
    paused.wait();
    EXPECT_EQ(cursor->stats().numReadStripes, std::min(limit, 6U));
    {
      const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(23));
      EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(23, 24));
    }
    cursor->reset();
    EXPECT_EQ(cursor->stats().numReadStripes, limit == 0 ? 1 : 6);
    EXPECT_EQ(cursor->stats().numSkippedStripes, limit == 0 ? 0 : 5);
  }
}

TEST_P(IndexScanCursorTest, retainsLoadedStorageAcrossResetAndReseek) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  auto cursor = makeCursor();
  cursor->seekToFirst();
  auto retained = folly::coro::blockingWait(cursor->co_nextStripe(0));
  std::vector<std::string> expected;
  for (const auto stream : retained.streams) {
    expected.emplace_back(stream);
  }
  cursor->reset();
  cursor->seekToFirst();
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(8));
    EXPECT_EQ(stripe.selection.stripeIndex, 2);
  }
  std::vector<std::string> actual;
  for (const auto stream : retained.streams) {
    actual.emplace_back(stream);
  }
  EXPECT_EQ(actual, expected);
  cursor->reset();
}

TEST_P(IndexScanCursorTest, preservesRowsWithAbsentProjectedStreams) {
  writeData(
      {singleFeatureBatch(0, StringView("a")),
       singleFeatureBatch(4, StringView("a")),
       singleFeatureBatch(8, StringView("a"))},
      {"key"},
      {{"features", {}}},
      /*stripeSize=*/1);
  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    NimbleIterator::Options options;
    if (prefetch) {
      options.prefetchExecutor = folly::getKeepAliveToken(*executor);
    }
    std::vector<Subfield> subfields;
    subfields.emplace_back("features[\"missing\"]");
    auto cursor =
        detail::IndexScanCursor::create(createProjector(subfields), options);
    cursor->seekToFirst();
    for (const uint32_t row : {0, 4, 8}) {
      SCOPED_TRACE(row);
      const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(row));
      EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(row, row + 4));
      EXPECT_EQ(stripe.dataHandle, nullptr);
      EXPECT_THAT(stripe.streams, ::testing::Each(::testing::IsEmpty()));
    }
    cursor->reset();
    EXPECT_EQ(cursor->stats().numReadStripes, 0);
    EXPECT_EQ(cursor->stats().numOutputRows, 12);
    EXPECT_EQ(dataIoStats_->read().count(), 0);
  }
}

// Cancels a consumer that waits for a prefetched stripe, then retries. The only
// prefetch thread is held busy, so the first co_nextStripe(0) call suspends on
// the empty stripe queue before any I/O starts. Cancelling it fails the call
// with "coroutine operation cancelled" and counts no output rows. Once the
// thread is released, a second co_nextStripe(0) call returns the first stripe,
// which shows that the cancelled call neither consumed nor lost a stripe.
TEST_P(IndexScanCursorTest, retriesAfterConsumerCancellation) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  std::promise<void> executorBusy;
  auto busy = executorBusy.get_future();
  std::promise<void> resumeExecutor;
  auto resume = resumeExecutor.get_future().share();
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 1;
  auto cursor = makeCursor(std::move(options));
  folly::CancellationSource cancellation;
  std::promise<void> consumerStarted;
  auto started = consumerStarted.get_future();
  std::future<folly::Try<ProjectedStripe>> pendingStripe;
  bool released{false};
  const auto release = [&] {
    if (!released) {
      released = true;
      resumeExecutor.set_value();
    }
  };
  auto cleanup = folly::makeGuard([&] {
    release();
    if (pendingStripe.valid()) {
      pendingStripe.wait();
    }
    cursor->reset();
    cursor.reset();
    executor->join();
  });
  executor->add([&] {
    executorBusy.set_value();
    resume.wait();
  });
  ASSERT_EQ(busy.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  cursor->seekToFirst();
  pendingStripe = std::async(std::launch::async, [&] {
    consumerStarted.set_value();
    return folly::coro::blockingWait(
        folly::coro::co_awaitTry(
            folly::coro::co_withCancellation(
                cancellation.getToken(), cursor->co_nextStripe(0))));
  });
  ASSERT_EQ(
      started.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  EXPECT_EQ(
      pendingStripe.wait_for(std::chrono::milliseconds{20}),
      std::future_status::timeout);
  cancellation.requestCancellation();
  ASSERT_EQ(
      pendingStripe.wait_for(std::chrono::seconds{5}),
      std::future_status::ready);
  const auto result = pendingStripe.get();
  ASSERT_TRUE(result.hasException());
  EXPECT_THAT(
      result.exception().what().toStdString(),
      ::testing::HasSubstr("coroutine operation cancelled"));
  EXPECT_EQ(cursor->stats().numOutputRows, 0);
  EXPECT_EQ(dataIoStats_->read().count(), 0);
  release();
  const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(0));
  EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(0, 4));
  EXPECT_EQ(cursor->stats().numOutputRows, 4);
}

TEST_P(IndexScanCursorTest, stopsPrefetchWhenDestroyedWithAFullQueue) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/6);
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 1;
  auto cursor = makeCursor(std::move(options));
  cursor->seekToFirst();
  std::promise<void> producerPaused;
  auto paused = producerPaused.get_future();
  executor->add([&] { producerPaused.set_value(); });
  paused.wait();
  EXPECT_EQ(cursor->stats().numReadStripes, 1);
  cursor.reset();
  executor->join();
  EXPECT_EQ(dataIoStats_->read().count(), 1);
}

TEST_P(IndexScanCursorTest, joinsAfterCleanupSchedulingFailure) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/1);
  auto executor = std::make_unique<RejectNextExecutor>();
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  auto cursor = makeCursor(std::move(options));
  cursor->seekToFirst();
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(0));
    EXPECT_EQ(stripe.selection.stripeIndex, 0);
  }
  std::promise<void> producerFinished;
  auto finished = producerFinished.get_future();
  executor->add([&] { producerFinished.set_value(); });
  ASSERT_EQ(
      finished.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  std::weak_ptr<detail::IndexScanCursor> owner = cursor;
  executor->rejectNext();
  cursor.reset();
  EXPECT_TRUE(owner.expired());
  executor->join();
  EXPECT_EQ(dataIoStats_->read().count(), 1);
}

DEBUG_ONLY_TEST_P(
    IndexScanCursorTest,
    stopsBeforeReservationAndLoadCheckpoints) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  for (const std::string hook :
       {"facebook::nimble::ScanPrefetcher::run::afterReservation",
        "facebook::nimble::ScanPrefetcher::run::beforeLoad"}) {
    SCOPED_TRACE(hook);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    NimbleIterator::Options options;
    options.prefetchExecutor = folly::getKeepAliveToken(*executor);
    auto cursor = makeCursor(options);
    {
      SCOPED_TESTVALUE_SET(
          hook,
          std::function<void(std::atomic_bool*)>([](std::atomic_bool* stopped) {
            stopped->store(true, std::memory_order_release);
          }));
      cursor->seekToFirst();
      std::promise<void> producerFinished;
      auto finished = producerFinished.get_future();
      executor->add([&] { producerFinished.set_value(); });
      finished.wait();
      EXPECT_EQ(cursor->stats().numReadStripes, 0);
      cursor->reset();
    }
    cursor->seekToFirst();
    {
      const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(0));
      EXPECT_EQ(stripe.selection.stripeIndex, 0);
    }
    cursor->reset();
  }
}

DEBUG_ONLY_TEST_P(IndexScanCursorTest, preservesTerminalLoadFailureUntilReset) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 1;
  auto cursor = makeCursor(options);
  {
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::DirectDataInput::allocateBuffer",
        std::function<void(uint64_t*)>(
            [](uint64_t*) { NIMBLE_FAIL("injected cursor load failure"); }));
    cursor->seekToFirst();
    for (int attempt{0}; attempt < 2; ++attempt) {
      SCOPED_TRACE(attempt);
      NIMBLE_ASSERT_THROW(
          folly::coro::blockingWait(cursor->co_nextStripe(0)),
          "injected cursor load failure");
    }
    EXPECT_EQ(dataIoStats_->read().count(), 0);
    cursor->reset();
  }
  cursor->seekToFirst();
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(0));
    EXPECT_EQ(stripe.selection.stripeIndex, 0);
  }
  cursor->reset();
}

DEBUG_ONLY_TEST_P(IndexScanCursorTest, retriesSynchronousLoadAfterFailure) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  auto cursor = makeCursor();
  cursor->seekToFirst();
  {
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::DirectDataInput::allocateBuffer",
        std::function<void(uint64_t*)>([](uint64_t*) {
          NIMBLE_FAIL("injected synchronous load failure");
        }));
    NIMBLE_ASSERT_THROW(
        folly::coro::blockingWait(cursor->co_nextStripe(0)),
        "injected synchronous load failure");
    EXPECT_EQ(cursor->stats().numOutputRows, 0);
    EXPECT_EQ(dataIoStats_->read().count(), 0);
  }
  const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(0));
  EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(0, 4));
  EXPECT_EQ(cursor->stats().numOutputRows, 4);
  EXPECT_EQ(dataIoStats_->read().count(), 1);
  cursor->reset();
}

DEBUG_ONLY_TEST_P(IndexScanCursorTest, preservesSkippedStatsAfterCancellation) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 1;
  auto cursor = makeCursor(std::move(options));
  folly::CancellationSource cancellation;
  std::promise<void> secondLoadStarted;
  auto started = secondLoadStarted.get_future();
  std::promise<void> continueLoad;
  auto continuation = continueLoad.get_future().share();
  std::future<folly::Try<ProjectedStripe>> pendingStripe;
  std::atomic_uint32_t attempts{0};
  bool released{false};
  const auto release = [&] {
    if (!released) {
      released = true;
      continueLoad.set_value();
    }
  };
  auto cleanup = folly::makeGuard([&] {
    release();
    if (pendingStripe.valid()) {
      pendingStripe.wait();
    }
    cursor->reset();
    cursor.reset();
    executor->join();
  });
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::DirectDataInput::allocateBuffer",
      std::function<void(uint64_t*)>([&](uint64_t*) {
        if (++attempts == 2) {
          secondLoadStarted.set_value();
          continuation.wait();
        }
      }));
  cursor->seekToFirst();
  pendingStripe = std::async(std::launch::async, [&] {
    return folly::coro::blockingWait(
        folly::coro::co_awaitTry(
            folly::coro::co_withCancellation(
                cancellation.getToken(), cursor->co_nextStripe(4))));
  });
  ASSERT_EQ(
      started.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  EXPECT_EQ(dataIoStats_->read().count(), 1);
  cancellation.requestCancellation();
  ASSERT_EQ(
      pendingStripe.wait_for(std::chrono::seconds{5}),
      std::future_status::ready);
  const auto result = pendingStripe.get();
  ASSERT_TRUE(result.hasException());
  EXPECT_THAT(
      result.exception().what().toStdString(),
      ::testing::HasSubstr("coroutine operation cancelled"));
  release();
  const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(4));
  EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(4, 8));
  cursor->reset();
  EXPECT_EQ(cursor->stats().numSkippedStripes, 1);
  EXPECT_EQ(cursor->stats().numOutputRows, 4);
}

DEBUG_ONLY_TEST_P(
    IndexScanCursorTest,
    reportsFailureAfterPreviouslyLoadedStripe) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 1;
  auto cursor = makeCursor(options);
  std::atomic_uint32_t attempts{0};
  {
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::DirectDataInput::allocateBuffer",
        std::function<void(uint64_t*)>([&](uint64_t*) {
          if (++attempts == 2) {
            NIMBLE_FAIL("injected second stripe failure");
          }
        }));
    cursor->seekToFirst();
    {
      const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(0));
      EXPECT_EQ(stripe.selection.stripeIndex, 0);
    }
    NIMBLE_ASSERT_THROW(
        folly::coro::blockingWait(cursor->co_nextStripe(4)),
        "injected second stripe failure");
    NIMBLE_ASSERT_THROW(
        folly::coro::blockingWait(cursor->co_nextStripe(4)),
        "injected second stripe failure");
    EXPECT_EQ(cursor->stats().numReadStripes, 1);
    cursor->reset();
  }
  cursor->seekToFirst();
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(4));
    EXPECT_EQ(stripe.selection.stripeIndex, 1);
  }
  cursor->reset();
}

DEBUG_ONLY_TEST_P(IndexScanCursorTest, resetJoinsAnInFlightLoad) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 1;
  auto cursor = makeCursor(std::move(options));
  std::promise<void> loadStarted;
  auto started = loadStarted.get_future();
  std::promise<void> continueLoad;
  auto continuation = continueLoad.get_future().share();
  std::promise<void> resetStarted;
  auto resetting = resetStarted.get_future();
  std::future<void> resetTask;
  bool released{false};
  const auto release = [&] {
    if (!released) {
      released = true;
      continueLoad.set_value();
    }
  };
  auto cleanup = folly::makeGuard([&] {
    release();
    if (resetTask.valid()) {
      resetTask.wait();
    }
    cursor->reset();
    cursor.reset();
    executor->join();
  });
  {
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::DirectDataInput::allocateBuffer",
        std::function<void(uint64_t*)>([&](uint64_t*) {
          loadStarted.set_value();
          continuation.wait();
        }));
    cursor->seekToFirst();
    ASSERT_EQ(
        started.wait_for(std::chrono::seconds{5}), std::future_status::ready);
    resetTask = std::async(std::launch::async, [&] {
      resetStarted.set_value();
      cursor->reset();
    });
    ASSERT_EQ(
        resetting.wait_for(std::chrono::seconds{5}), std::future_status::ready);
    EXPECT_EQ(
        resetTask.wait_for(std::chrono::milliseconds{20}),
        std::future_status::timeout);
    release();
    resetTask.get();
  }
  NIMBLE_ASSERT_THROW(cursor->keyCursor(0), "No index scan is active");
  EXPECT_EQ(dataIoStats_->read().count(), 1);
  cursor->seekToFirst();
  {
    const auto stripe = folly::coro::blockingWait(cursor->co_nextStripe(4));
    EXPECT_EQ(stripe.selection.stripeIndex, 1);
  }
  cursor->reset();
}

TEST_P(IndexScanCursorTest, stopsBeforeQueuedProducerStarts) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  std::promise<void> executorBusy;
  auto busy = executorBusy.get_future();
  std::promise<void> resumeExecutor;
  auto resume = resumeExecutor.get_future().share();
  std::shared_ptr<detail::IndexScanCursor> cursor;
  bool released{false};
  const auto release = [&] {
    if (!released) {
      released = true;
      resumeExecutor.set_value();
    }
  };
  auto cleanup = folly::makeGuard([&] {
    cursor.reset();
    release();
    executor->join();
  });
  executor->add([&] {
    executorBusy.set_value();
    resume.wait();
  });
  ASSERT_EQ(busy.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  cursor = makeCursor(std::move(options));
  cursor->seekToFirst();
  EXPECT_EQ(cursor->stats().numReadStripes, 0);
  cursor.reset();
  release();
  executor->join();
  EXPECT_EQ(dataIoStats_->read().count(), 0);
}

DEBUG_ONLY_TEST_P(IndexScanCursorTest, reportsStatsDuringAnInFlightLoad) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 1;
  auto cursor = makeCursor(std::move(options));
  std::promise<void> loadStarted;
  auto started = loadStarted.get_future();
  std::promise<void> continueLoad;
  auto continuation = continueLoad.get_future().share();
  std::promise<void> statsStarted;
  auto readingStats = statsStarted.get_future();
  std::future<NimbleIterator::Stats> statsTask;
  bool released{false};
  const auto release = [&] {
    if (!released) {
      released = true;
      continueLoad.set_value();
    }
  };
  auto cleanup = folly::makeGuard([&] {
    release();
    if (statsTask.valid()) {
      statsTask.wait();
    }
    cursor->reset();
    cursor.reset();
    executor->join();
  });
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::DirectDataInput::allocateBuffer",
      std::function<void(uint64_t*)>([&](uint64_t*) {
        loadStarted.set_value();
        continuation.wait();
      }));
  cursor->seekToFirst();
  ASSERT_EQ(
      started.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  statsTask = std::async(std::launch::async, [&] {
    statsStarted.set_value();
    return cursor->stats();
  });
  ASSERT_EQ(
      readingStats.wait_for(std::chrono::seconds{5}),
      std::future_status::ready);
  ASSERT_EQ(
      statsTask.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  const auto partialStats = statsTask.get();
  EXPECT_EQ(partialStats.numReadStripes, 0);
  EXPECT_GT(partialStats.numPlannedBytes, 0);
  EXPECT_EQ(dataIoStats_->read().count(), 0);
  release();
  cursor->reset();
  const auto stats = cursor->stats();
  EXPECT_EQ(stats.numReadStripes, 1);
  EXPECT_GT(stats.numPlannedBytes, 0);
  EXPECT_EQ(dataIoStats_->read().count(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    AllCacheParams,
    IndexScanCursorTest,
    ::testing::ValuesIn(IndexScanCursorTest::getIteratorTestParams()),
    IndexScanCursorTest::iteratorTestParamName);

} // namespace facebook::nimble::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, /*removeFlags=*/false};
  return RUN_ALL_TESTS();
}
