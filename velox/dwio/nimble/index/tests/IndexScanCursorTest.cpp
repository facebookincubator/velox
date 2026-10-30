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
#include <numeric>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "folly/CancellationToken.h"
#include "folly/Random.h"
#include "folly/ScopeGuard.h"
#include "folly/Synchronized.h"
#include "folly/container/F14Map.h"
#include "folly/coro/Timeout.h"
#include "folly/init/Init.h"
#include "folly/io/IOBuf.h"
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

// Materializes each stripe into entries that hold only the number of selected
// rows, and keeps the stripe until the test takes it, so that a test can
// inspect what the cursor loaded. A cursor that prefetches materializes on
// threads of its executor.
class RecordingMaterializer final
    : public detail::IndexScanCursor::StripeMaterializer {
 public:
  folly::coro::Task<std::unique_ptr<Entries>> materialize(
      ProjectedStripe stripe,
      const index::ClusterIndexBase& /*clusterIndex*/) override {
    numMaterializeCalls_.fetch_add(1, std::memory_order_relaxed);
    if (onMaterialize_ != nullptr) {
      onMaterialize_(stripe);
    }
    auto entries = std::make_unique<Entries>(iobufFactory_);
    entries->numRows = static_cast<velox::vector_size_t>(
        stripe.selection.selectedRowRange().numRows());
    // Replaces the stripe of entries that the cursor dropped unread, which can
    // have had the same address.
    stripes_.wlock()->insert_or_assign(entries.get(), std::move(stripe));
    co_return entries;
  }

  // Returns the stripe that 'entries' were materialized from.
  ProjectedStripe takeStripe(const Entries& entries) {
    auto stripes = stripes_.wlock();
    const auto it = stripes->find(&entries);
    NIMBLE_CHECK(it != stripes->end(), "Entries have no recorded stripe");
    auto stripe = std::move(it->second);
    stripes->erase(it);
    return stripe;
  }

  // Calls 'onMaterialize' with each stripe before it materializes, on the
  // thread that materializes it, so that a test can block or fail the stripe.
  // Set it before the cursor loads a stripe.
  void setOnMaterialize(
      std::function<void(const ProjectedStripe&)> onMaterialize) {
    onMaterialize_ = std::move(onMaterialize);
  }

  uint32_t numMaterializeCalls() const {
    return numMaterializeCalls_.load(std::memory_order_relaxed);
  }

 private:
  folly::IOBufFactory iobufFactory_{
      [](size_t capacity) { return folly::IOBuf::create(capacity); }};
  std::function<void(const ProjectedStripe&)> onMaterialize_;
  std::atomic_uint32_t numMaterializeCalls_{0};
  // The stripes of the entries that the test has not taken, by entries.
  folly::Synchronized<folly::F14FastMap<const Entries*, ProjectedStripe>>
      stripes_;
};

// Exercises scan state and stripe loading through the cursor's own API.
class IndexScanCursorTest : public NimbleIndexProjectorTest {
 protected:
  // Takes 'options' by value so that a test can move its prefetch executor
  // token into the cursor. ThreadPoolExecutor::join() waits for every token, so
  // a token left in the test's own options would block joining the executor.
  std::shared_ptr<detail::IndexScanCursor> makeCursor(
      NimbleIterator::Options options = {}) {
    std::vector<Subfield> subfields;
    subfields.emplace_back("value");
    return makeCursor(subfields, std::move(options));
  }

  std::shared_ptr<detail::IndexScanCursor> makeCursor(
      const std::vector<Subfield>& subfields,
      NimbleIterator::Options options) {
    options.cacheEncodingView = GetParam().cacheEncodingView;
    auto materializer = std::make_shared<RecordingMaterializer>();
    materializer_ = materializer.get();
    return detail::IndexScanCursor::create(
        createProjector(subfields), std::move(materializer), options);
  }

  // Materializes the stripe of the file row 'row' with the last cursor that
  // makeCursor() created, and returns the stripe that the cursor loaded for it.
  ProjectedStripe nextStripe(detail::IndexScanCursor& cursor, uint32_t row) {
    const auto entries = folly::coro::blockingWait(cursor.co_nextStripe(row));
    NIMBLE_CHECK_NOT_NULL(entries);
    return materializer_->takeStripe(*entries);
  }

  // The materializer of the last cursor that makeCursor() created. The cursor
  // owns it, so that the stripes that a test leaves untaken go away with the
  // cursor and its prefetch, before the memory pool of the test.
  RecordingMaterializer* materializer_{nullptr};
};

TEST_P(IndexScanCursorTest, rejectsNullProjector) {
  try {
    auto cursor = detail::IndexScanCursor::create(
        nullptr, std::make_shared<RecordingMaterializer>(), {});
    FAIL() << "Expected null projector rejection";
  } catch (const velox::VeloxRuntimeError& error) {
    EXPECT_THAT(
        std::string{error.what()}, ::testing::HasSubstr("pointer.get()"));
  }
}

TEST_P(IndexScanCursorTest, rejectsNullMaterializer) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/1);
  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  try {
    auto cursor = detail::IndexScanCursor::create(
        createProjector(subfields), /*materializer=*/nullptr, {});
    FAIL() << "Expected null materializer rejection";
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
    const auto& stripe = nextStripe(*cursor, row);
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
    nextStripe(*cursor, 2);
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

TEST_P(IndexScanCursorTest, rejectsRowsOutsideTheActiveScan) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  NIMBLE_ASSERT_THROW(nextStripe(*cursor, 0), "No index scan is active");

  cursor->seek(makeRangeLookup(rowType, {"key"}, 2, 10));
  for (const uint32_t row : {1U, 10U, std::numeric_limits<uint32_t>::max()}) {
    SCOPED_TRACE(row);
    NIMBLE_ASSERT_THROW(
        nextStripe(*cursor, row), "Row is outside the active scan");
  }
  cursor->reset();
  NIMBLE_ASSERT_THROW(nextStripe(*cursor, 2), "No index scan is active");
}

TEST_P(IndexScanCursorTest, requiresTheFirstSelectedRowOfTheNextStripe) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    NimbleIterator::Options options;
    if (prefetch) {
      options.prefetchExecutor = folly::getKeepAliveToken(*executor);
    }
    auto cursor = makeCursor(options);
    cursor->seek(makeRangeLookup(rowType, {"key"}, 2, 10));
    NIMBLE_ASSERT_THROW(
        nextStripe(*cursor, 3),
        "Row is not the first selected row of the next stripe");
    {
      const auto& stripe = nextStripe(*cursor, 2);
      EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(2, 4));
    }
    // A row inside the next stripe, and the first row of a later one.
    for (const uint32_t row : {5U, 8U}) {
      SCOPED_TRACE(row);
      NIMBLE_ASSERT_THROW(
          nextStripe(*cursor, row),
          "Row is not the first selected row of the next stripe");
    }
    EXPECT_EQ(cursor->stats().numOutputRows, 2);
    cursor->reset();
  }
}

TEST_P(IndexScanCursorTest, clearsMissesAndSupportsReseek) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  auto cursor = makeCursor();
  EXPECT_FALSE(cursor->seek(makeRangeLookup(rowType, {"key"}, 50, 60)));
  NIMBLE_ASSERT_THROW(nextStripe(*cursor, 0), "No index scan is active");

  EXPECT_EQ(cursor->seekToFirst(), std::optional<RowRange>(RowRange{0, 12}));
  {
    const auto& stripe = nextStripe(*cursor, 0);
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(0, 4));
  }
  cursor->seek(makeRangeLookup(rowType, {"key"}, 6, 11));
  {
    const auto& stripe = nextStripe(*cursor, 6);
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(6, 8));
  }
  {
    const auto& stripe = nextStripe(*cursor, 8);
    EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(8, 11));
  }
  cursor->reset();
  EXPECT_EQ(cursor->stats().numReadStripes, 3);
  EXPECT_EQ(cursor->stats().numOutputRows, 9);
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
    const auto& stripe = nextStripe(*cursor, 0);
    EXPECT_EQ(stripe.selection.stripeIndex, 0);
  }
  EXPECT_EQ(cursor->stats().numReadStripes, 1);
  cursor->reset();
}

TEST_P(IndexScanCursorTest, returnsCallerScratchEncodingOptions) {
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
    std::vector<Subfield> subfields;
    subfields.emplace_back("value");
    const auto projector = createProjector(subfields);
    velox::BufferPool scratch{velox::BufferPool::kDefaultCapacity};
    for (auto* bufferPool :
         {&scratch, static_cast<velox::BufferPool*>(nullptr)}) {
      for (const bool cacheEncodingView : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "bufferPool={}, cacheEncodingView={}",
                static_cast<void*>(bufferPool),
                cacheEncodingView));
        const auto options = detail::IndexScanCursor::encodingOptions(
            *projector, cacheEncodingView, bufferPool);
        EXPECT_EQ(options.bufferPool, bufferPool);
        EXPECT_EQ(options.useVarintRowCount, compactRowCount);
        EXPECT_EQ(options.cacheEncodingView, cacheEncodingView);
      }
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
    const auto& projected = nextStripe(*cursor, 0);
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
    const auto& stripe = nextStripe(*cursor, row);
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
    const auto& stripe = nextStripe(*cursor, row);
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
    for (const uint32_t stripeIndex : {0U, 1U}) {
      const auto& stripe = nextStripe(*cursor, stripeIndex * 4);
      EXPECT_EQ(stripe.selection.stripeIndex, stripeIndex);
    }
    for (const uint32_t row : {0U, 4U, 7U}) {
      SCOPED_TRACE(row);
      NIMBLE_ASSERT_THROW(
          nextStripe(*cursor, row),
          "Row is not the first selected row of the next stripe");
    }
    {
      const auto& stripe = nextStripe(*cursor, 8);
      EXPECT_EQ(stripe.selection.stripeIndex, 2);
    }
    NIMBLE_ASSERT_THROW(nextStripe(*cursor, 8), "No selected stripes remain");
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
    for (uint32_t row = 0; row < 24; row += 4) {
      SCOPED_TRACE(row);
      const auto& stripe = nextStripe(*cursor, row);
      EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(row, row + 4));
    }
    cursor->reset();
    EXPECT_EQ(cursor->stats().numReadStripes, 6);
  }
}

TEST_P(IndexScanCursorTest, retainsLoadedStorageAcrossResetAndReseek) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  auto cursor = makeCursor();
  cursor->seekToFirst();
  const auto& retained = nextStripe(*cursor, 0);
  std::vector<std::string> expected;
  for (const auto stream : retained.streams) {
    expected.emplace_back(stream);
  }
  cursor->reset();
  cursor->seekToFirst();
  for (const uint32_t stripeIndex : {0U, 1U, 2U}) {
    const auto& stripe = nextStripe(*cursor, stripeIndex * 4);
    EXPECT_EQ(stripe.selection.stripeIndex, stripeIndex);
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
    auto cursor = makeCursor(subfields, options);
    cursor->seekToFirst();
    for (const uint32_t row : {0, 4, 8}) {
      SCOPED_TRACE(row);
      const auto& stripe = nextStripe(*cursor, row);
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
  std::future<folly::Try<
      std::unique_ptr<detail::IndexScanCursor::StripeMaterializer::Entries>>>
      pendingStripe;
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
  const auto& stripe = nextStripe(*cursor, 0);
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
    const auto& stripe = nextStripe(*cursor, 0);
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
      const auto& stripe = nextStripe(*cursor, 0);
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
          nextStripe(*cursor, 0), "injected cursor load failure");
    }
    EXPECT_EQ(dataIoStats_->read().count(), 0);
    cursor->reset();
  }
  cursor->seekToFirst();
  {
    const auto& stripe = nextStripe(*cursor, 0);
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
        nextStripe(*cursor, 0), "injected synchronous load failure");
    EXPECT_EQ(cursor->stats().numOutputRows, 0);
    EXPECT_EQ(dataIoStats_->read().count(), 0);
  }
  const auto& stripe = nextStripe(*cursor, 0);
  EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(0, 4));
  EXPECT_EQ(cursor->stats().numOutputRows, 4);
  EXPECT_EQ(dataIoStats_->read().count(), 1);
  cursor->reset();
}

// Cancels a consumer that waits for the second stripe while its load is in
// flight, then retries: the cancelled call neither consumed nor lost the
// stripe.
DEBUG_ONLY_TEST_P(IndexScanCursorTest, retriesAfterCancellationDuringALoad) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  // The producer can start the second load before the task that materializes
  // the first stripe runs, and holds its thread in that load, so the task needs
  // a second thread.
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(2);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 1;
  auto cursor = makeCursor(std::move(options));
  folly::CancellationSource cancellation;
  std::promise<void> secondLoadStarted;
  auto started = secondLoadStarted.get_future();
  std::promise<void> continueLoad;
  auto continuation = continueLoad.get_future().share();
  std::future<folly::Try<
      std::unique_ptr<detail::IndexScanCursor::StripeMaterializer::Entries>>>
      pendingStripe;
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
  {
    const auto& stripe = nextStripe(*cursor, 0);
    EXPECT_EQ(stripe.selection.stripeIndex, 0);
  }
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
  const auto& stripe = nextStripe(*cursor, 4);
  EXPECT_EQ(stripe.selection.selectedRowRange(), RowRange(4, 8));
  cursor->reset();
  EXPECT_EQ(cursor->stats().numOutputRows, 8);
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
      const auto& stripe = nextStripe(*cursor, 0);
      EXPECT_EQ(stripe.selection.stripeIndex, 0);
    }
    NIMBLE_ASSERT_THROW(
        nextStripe(*cursor, 4), "injected second stripe failure");
    NIMBLE_ASSERT_THROW(
        nextStripe(*cursor, 4), "injected second stripe failure");
    EXPECT_EQ(cursor->stats().numReadStripes, 1);
    cursor->reset();
  }
  cursor->seekToFirst();
  for (const uint32_t stripeIndex : {0U, 1U}) {
    const auto& stripe = nextStripe(*cursor, stripeIndex * 4);
    EXPECT_EQ(stripe.selection.stripeIndex, stripeIndex);
  }
  cursor->reset();
}

TEST_P(IndexScanCursorTest, failsAtAStripeThatFailsToMaterialize) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(2);
    NimbleIterator::Options options;
    if (prefetch) {
      options.prefetchExecutor = folly::getKeepAliveToken(*executor);
      options.maxPrefetchStripes = 4;
    }
    auto cursor = makeCursor(options);
    materializer_->setOnMaterialize([](const ProjectedStripe& stripe) {
      NIMBLE_CHECK_NE(
          stripe.selection.stripeIndex, 1U, "injected materialization failure");
    });
    for (const bool restart : {false, true}) {
      SCOPED_TRACE(restart);
      cursor->seekToFirst();
      {
        const auto& stripe = nextStripe(*cursor, 0);
        EXPECT_EQ(stripe.selection.stripeIndex, 0);
      }
      // The failure is the same each time the iterator asks for the stripe.
      NIMBLE_ASSERT_THROW(
          nextStripe(*cursor, 4), "injected materialization failure");
      NIMBLE_ASSERT_THROW(
          nextStripe(*cursor, 4), "injected materialization failure");
    }
    cursor->reset();
  }
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
  NIMBLE_ASSERT_THROW(nextStripe(*cursor, 0), "No index scan is active");
  EXPECT_EQ(dataIoStats_->read().count(), 1);
  cursor->seekToFirst();
  for (const uint32_t stripeIndex : {0U, 1U}) {
    const auto& stripe = nextStripe(*cursor, stripeIndex * 4);
    EXPECT_EQ(stripe.selection.stripeIndex, stripeIndex);
  }
  cursor->reset();
}

TEST_P(IndexScanCursorTest, resetSkipsStripesThatWaitToMaterialize) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  NimbleIterator::Options options;
  options.prefetchExecutor = folly::getKeepAliveToken(*executor);
  options.maxPrefetchStripes = 6;
  auto cursor = makeCursor(std::move(options));
  std::promise<void> materializeStarted;
  auto started = materializeStarted.get_future();
  std::promise<void> continueMaterialize;
  auto continuation = continueMaterialize.get_future().share();
  std::atomic_bool blocked{false};
  // Blocks the first stripe that materializes on the only prefetch thread, so
  // that the tasks of the other stripes wait behind it.
  materializer_->setOnMaterialize([&](const ProjectedStripe& /*stripe*/) {
    if (!blocked.exchange(true)) {
      materializeStarted.set_value();
      continuation.wait();
    }
  });
  std::future<void> resetTask;
  bool released{false};
  const auto release = [&] {
    if (!released) {
      released = true;
      continueMaterialize.set_value();
    }
  };
  auto cleanup = folly::makeGuard([&] {
    release();
    if (resetTask.valid()) {
      resetTask.wait();
    }
    cursor.reset();
    executor->join();
  });
  cursor->seekToFirst();
  ASSERT_EQ(
      started.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  resetTask = std::async(std::launch::async, [&] { cursor->reset(); });
  // The reset stops the prefetch and waits for the stripe in flight.
  EXPECT_EQ(
      resetTask.wait_for(std::chrono::milliseconds{20}),
      std::future_status::timeout);
  release();
  resetTask.get();
  // The stripes that waited skip their materialization.
  EXPECT_EQ(materializer_->numMaterializeCalls(), 1);
}

// Fuzzes the failures that the prefetch of a scan hands to the iterator, over
// random scans and prefetch settings: a failed load, a failure at the
// reservation or the load checkpoint of the producer, and a failed
// materialization, injected with test values and through the materializer, and
// a cancelled read and a restart in the middle of a scan. Whatever happens, the
// stripes come in order, a failure carries the injected message, a failure that
// the prefetch reported repeats until the next seek while one without prefetch
// leaves the next read to retry, a cancelled read loses no stripe, and a scan
// without faults afterwards reads every stripe.
DEBUG_ONLY_TEST_P(IndexScanCursorTest, fuzzesPrefetchFailures) {
  constexpr int kNumIterations{60};
  // Stripes of different sizes, as each batch is a stripe.
  const std::vector<uint32_t> stripeSizes{3, 1, 4, 2, 5, 3};
  std::vector<uint32_t> stripeStarts;
  stripeStarts.reserve(stripeSizes.size());
  std::vector<RowVectorPtr> batches;
  batches.reserve(stripeSizes.size());
  uint32_t numRows{0};
  for (const auto stripeSize : stripeSizes) {
    stripeStarts.emplace_back(numRows);
    std::vector<int64_t> keys(stripeSize);
    std::iota(keys.begin(), keys.end(), static_cast<int64_t>(numRows));
    std::vector<int32_t> values(stripeSize);
    std::transform(keys.begin(), keys.end(), values.begin(), [](auto key) {
      return static_cast<int32_t>(key * 10);
    });
    batches.emplace_back(vectorMaker_->rowVector(
        {"key", "value"},
        {vectorMaker_->flatVector<int64_t>(keys),
         vectorMaker_->flatVector<int32_t>(values)}));
    numRows += stripeSize;
  }
  writeData(batches, {"key"}, {}, /*stripeSize=*/1);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});

  struct SelectedStripe {
    uint32_t stripeIndex;
    RowRange rows;
  };
  // Returns the selections of the stripes that the rows [startRow, endRow)
  // span.
  const auto selectStripes = [&](uint32_t startRow, uint32_t endRow) {
    std::vector<SelectedStripe> stripes;
    for (uint32_t stripe = 0; stripe < stripeSizes.size(); ++stripe) {
      const auto stripeStart = stripeStarts[stripe];
      const auto stripeEnd = stripeStart + stripeSizes[stripe];
      if (stripeEnd > startRow && stripeStart < endRow) {
        stripes.push_back(
            {stripe,
             RowRange{
                 std::max(startRow, stripeStart),
                 std::min(endRow, stripeEnd)}});
      }
    }
    return stripes;
  };

  enum class Fault {
    kNone,
    kLoad,
    kReservation,
    kLoadCheckpoint,
    kMaterialization,
    kCancellation,
    kRestart,
  };
  constexpr uint32_t kNumFaults{7};
  std::atomic<Fault> fault{Fault::kNone};
  // An injected fault fires once, the time that its point brings this to zero.
  std::atomic_int32_t countdown{0};
  const auto fires = [&](Fault point) {
    return fault.load() == point && countdown.fetch_sub(1) == 1;
  };

  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::DirectDataInput::allocateBuffer",
      std::function<void(uint64_t*)>([&](uint64_t*) {
        if (fires(Fault::kLoad)) {
          NIMBLE_FAIL("injected load failure");
        }
      }));
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::ScanPrefetcher::run::afterReservation",
      std::function<void(std::atomic_bool*)>([&](std::atomic_bool*) {
        if (fires(Fault::kReservation)) {
          NIMBLE_FAIL("injected reservation failure");
        }
      }));
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::ScanPrefetcher::run::beforeLoad",
      std::function<void(std::atomic_bool*)>([&](std::atomic_bool*) {
        if (fires(Fault::kLoadCheckpoint)) {
          NIMBLE_FAIL("injected checkpoint failure");
        }
      }));

  const auto seed = folly::Random::rand32();
  LOG(INFO) << "fuzzesPrefetchFailures seed: " << seed;
  folly::detail::DefaultGenerator random{seed};
  // Counts the reads that failed with an injected fault, so that the test
  // notices injection points that no longer fire.
  uint32_t numInjectedFailures{0};
  for (int iteration{0}; iteration < kNumIterations; ++iteration) {
    const bool usePrefetch = folly::Random::rand32(random) % 4 != 0;
    const auto numThreads = 1 + folly::Random::rand32(random) % 3;
    const auto maxPrefetchStripes = folly::Random::rand32(random) % 8;
    const bool fullScan = folly::Random::rand32(random) % 3 == 0;
    const auto startRow =
        fullScan ? 0 : folly::Random::rand32(random) % numRows;
    const auto endRow = fullScan
        ? numRows
        : startRow + 1 + folly::Random::rand32(random) % (numRows - startRow);
    const auto stripes = selectStripes(startRow, endRow);
    // Cycles through the faults, so that each runs in every test.
    const auto iterationFault = static_cast<Fault>(iteration % kNumFaults);
    // The time that the point of the fault fires it, or the stripe, counted
    // from one, at which the consumer cancels a read or restarts the scan.
    const auto trigger = static_cast<int32_t>(
        1 + folly::Random::rand32(random) % stripes.size());
    const bool resetFirst = folly::Random::rand32(random) % 2 == 0;
    SCOPED_TRACE(
        fmt::format(
            "seed={}, iteration={}, prefetch={}, threads={}, "
            "maxPrefetchStripes={}, rows=[{}, {}), fault={}, trigger={}",
            seed,
            iteration,
            usePrefetch,
            numThreads,
            maxPrefetchStripes,
            startRow,
            endRow,
            static_cast<int>(iterationFault),
            trigger));

    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(numThreads);
    NimbleIterator::Options options;
    if (usePrefetch) {
      options.prefetchExecutor = folly::getKeepAliveToken(*executor);
      options.maxPrefetchStripes = maxPrefetchStripes;
    }
    const bool prefetches = usePrefetch && maxPrefetchStripes > 0;
    auto cursor = makeCursor(options);
    materializer_->setOnMaterialize([&](const ProjectedStripe& /*stripe*/) {
      if (fires(Fault::kMaterialization)) {
        NIMBLE_FAIL("injected materialization failure");
      }
    });
    const auto seek = [&] {
      if (fullScan) {
        cursor->seekToFirst();
      } else {
        cursor->seek(makeRangeLookup(rowType, {"key"}, startRow, endRow));
      }
    };
    const auto expectStripe = [&](const ProjectedStripe& stripe,
                                  const SelectedStripe& expected) {
      EXPECT_EQ(stripe.selection.stripeIndex, expected.stripeIndex);
      EXPECT_EQ(stripe.selection.selectedRowRange(), expected.rows);
    };
    // Reads the stripe of 'row', and fails instead of hanging when the read
    // waits for a stripe that never comes.
    const auto readStripe = [&](uint32_t row) {
      const auto entries = folly::coro::blockingWait(
          folly::coro::timeout(
              cursor->co_nextStripe(row), std::chrono::seconds{10}));
      NIMBLE_CHECK_NOT_NULL(entries);
      return materializer_->takeStripe(*entries);
    };

    countdown = trigger;
    fault = iterationFault;
    seek();
    size_t next{0};
    bool acted{false};
    bool retried{false};
    while (next < stripes.size()) {
      const auto& expected = stripes[next];
      const auto row = expected.rows.startRow;
      if (!acted && next + 1 == static_cast<size_t>(trigger)) {
        if (iterationFault == Fault::kRestart) {
          acted = true;
          if (resetFirst) {
            cursor->reset();
          }
          seek();
          next = 0;
          continue;
        }
        if (iterationFault == Fault::kCancellation) {
          acted = true;
          folly::CancellationSource cancellation;
          cancellation.requestCancellation();
          auto result = folly::coro::blockingWait(
              folly::coro::co_awaitTry(
                  folly::coro::co_withCancellation(
                      cancellation.getToken(), cursor->co_nextStripe(row))));
          if (result.hasException()) {
            // The next read gets the stripe that the cancelled one did not.
            EXPECT_THAT(
                result.exception().what().toStdString(),
                ::testing::HasSubstr("cancelled"));
            continue;
          }
          ASSERT_NE(result.value(), nullptr);
          expectStripe(materializer_->takeStripe(*result.value()), expected);
          ++next;
          continue;
        }
      }
      try {
        expectStripe(readStripe(row), expected);
        ++next;
        retried = false;
      } catch (const std::exception& error) {
        const std::string message{error.what()};
        ASSERT_THAT(message, ::testing::HasSubstr("injected"));
        ++numInjectedFailures;
        if (!prefetches) {
          // A failure without prefetch leaves the next read to retry the
          // stripe, which then succeeds, as each fault fires once.
          ASSERT_FALSE(retried) << message;
          retried = true;
          continue;
        }
        // The prefetch reported the failure, which repeats until the next
        // seek.
        try {
          readStripe(row);
          FAIL() << "Expected the failure to repeat";
        } catch (const std::exception& repeated) {
          EXPECT_EQ(std::string{repeated.what()}, message);
        }
        break;
      }
    }

    // A scan without faults afterwards reads every stripe.
    fault = Fault::kNone;
    cursor->seekToFirst();
    for (const auto& expected : selectStripes(0, numRows)) {
      expectStripe(readStripe(expected.rows.startRow), expected);
    }
    cursor->reset();
  }
  EXPECT_GT(numInjectedFailures, 0);
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
