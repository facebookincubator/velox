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
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/ScopeGuard.h>
#include <folly/coro/Timeout.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/HybridFlatMap.h"
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/common/SchemaSerialization.h"
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
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileProperties.h"
#include "velox/dwio/nimble/tablet/FilePropertiesGenerated.h"
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
#include "velox/common/file/IoUringReader.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/row/CompactRow.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble::test {

class NimbleRowIteratorTest : public NimbleIndexProjectorTest {
 protected:
  // Writes the rows of writeResumeKeyTestData(): the key of each row is its row
  // number, and its value is ten times the key. Ends a stripe every
  // 'rowsPerStripe' rows and a key chunk of the cluster index every
  // 'rowsPerKeyChunk' rows, in the one partition of the index.
  void writeKeyChunkTestData(
      int32_t rowsPerStripe,
      int32_t numStripes,
      int32_t rowsPerKeyChunk) {
    std::vector<RowVectorPtr> batches;
    for (int32_t stripe{0}; stripe < numStripes; ++stripe) {
      std::vector<int64_t> keys(rowsPerStripe);
      std::vector<int32_t> values(rowsPerStripe);
      for (int32_t row{0}; row < rowsPerStripe; ++row) {
        keys[row] = stripe * rowsPerStripe + row;
        values[row] = static_cast<int32_t>(keys[row] * 10);
      }
      batches.emplace_back(vectorMaker_->rowVector(
          {"key", "value"},
          {vectorMaker_->flatVector<int64_t>(keys),
           vectorMaker_->flatVector<int32_t>(values)}));
    }
    WriterOptions options;
    options.encodingSelectionPolicyCreator = encodingViewSelectionPolicy();
    options.clusterIndexConfig =
        ClusterIndexConfigBuilder{}
            .withKeyColumns({"key"})
            .withSortOrders({SortOrder{.ascending = true}})
            .withEnforceKeyOrder(true)
            .withNoDuplicateKey(true)
            .withMaxRowsPerKeyChunk(rowsPerKeyChunk)
            .build();
    options.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*shouldFlush=*/[](const StripeProgress& /*progress*/) {
            return true;
          });
    };
    writeBatches(batches, std::move(options));
  }
};

namespace {

// Copies 'bytes' into 'storage' and appends a view of the copy to 'views'.
void appendBytes(
    std::string_view bytes,
    folly::IOBufQueue& storage,
    std::vector<std::string_view>& views) {
  auto [buffer, capacity] = storage.preallocate(bytes.size(), 4'096);
  NIMBLE_CHECK_GE(capacity, bytes.size());
  std::memcpy(buffer, bytes.data(), bytes.size());
  storage.postallocate(bytes.size());
  views.emplace_back(static_cast<const char*>(buffer), bytes.size());
}

// Decodes each row into two entries, whose keys increase with the value of the
// row, and checks that the key cursor has a key for each row of the stripe.
void decodeTwoEntriesPerRow(
    const velox::RowVectorPtr& rows,
    nimble::index::KeyCursor& keyCursor,
    folly::IOBufQueue& keyStorage,
    folly::IOBufQueue& valueStorage,
    std::vector<std::string_view>& keys,
    std::vector<std::string_view>& values) {
  const auto* numbers = rows->childAt(0)->asFlatVector<int32_t>();
  for (velox::vector_size_t row{0}; row < rows->size(); ++row) {
    NIMBLE_CHECK(keyCursor.hasNext(), "The stripe has a key for each row");
    keyCursor.next();
    for (const char suffix : {'a', 'b'}) {
      appendBytes(
          fmt::format("k{:06}{}", numbers->valueAt(row), suffix),
          keyStorage,
          keys);
      appendBytes(
          fmt::format("v{}{}", numbers->valueAt(row), suffix),
          valueStorage,
          values);
    }
  }
  NIMBLE_CHECK(!keyCursor.hasNext(), "The stripe has a key for each row");
}

} // namespace

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
    // Reading the key loads the stripe, and value() on its own does too.
    if (row % 2 == 0) {
      iterator.key();
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

TEST_P(NimbleRowIteratorTest, rowIteratorDecodesLargeStripes) {
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

TEST_P(NimbleRowIteratorTest, rowIteratorKeepsKeysAlignedAfterSkippingRows) {
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
  EXPECT_EQ(iterator.stats().numReadStripes, 2);
  expectValue(kSkippedRows);
  expectKey(kSkippedRows);

  iterator.next();
  expectKey(6);
  iterator.next();
  expectValue(7);
  iterator.next();
  expectKey(8);
}

TEST_P(NimbleRowIteratorTest, rowIteratorKeepsItsStreamReader) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  // Each StreamReader constructs its output type once. Retain the types so
  // that rebuilding a reader cannot reuse an earlier type's address.
  std::set<velox::TypePtr> decodedTypes;
  const auto decodeRows = compactRowDecoder();
  TestNimbleIterator iterator{createEntryIterator(
      projectedSubfields,
      [&](const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& keyCursor,
          folly::IOBufQueue& keyStorage,
          folly::IOBufQueue& valueStorage,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        decodedTypes.insert(rows->type());
        decodeRows(rows, keyCursor, keyStorage, valueStorage, keys, values);
      },
      /*options=*/{})};

  // The iterator builds a reader for its first stripe and keeps it for the
  // other stripes and for later seeks.
  for (int seek = 0; seek < 2; ++seek) {
    SCOPED_TRACE(seek);
    iterator.seekToFirst();
    int32_t numRows{0};
    for (; iterator.valid(); iterator.next()) {
      ++numRows;
    }
    EXPECT_EQ(numRows, 12);
  }
  EXPECT_THAT(decodedTypes, ::testing::SizeIs(1));
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

  const auto serializedValue = iterator.value();
  const auto actual = velox::row::CompactRow::deserialize(
      {serializedValue}, ROW({"value"}, {INTEGER()}), leafPool_.get());
  const auto* values = actual->childAt(0)->asFlatVector<int32_t>();
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(values->valueAt(0), 60);
}

TEST_P(NimbleRowIteratorTest, entryIteratorSupportsEmptyEntries) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  std::atomic_size_t numDecoderCalls{0};
  TestNimbleIterator iterator{createEntryIterator(
      projectedSubfields,
      [&numDecoderCalls](
          const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& /*keyCursor*/,
          folly::IOBufQueue& /*keyStorage*/,
          folly::IOBufQueue& /*valueStorage*/,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        numDecoderCalls.fetch_add(1, std::memory_order_relaxed);
        keys.assign(rows->size(), {});
        values.assign(rows->size(), {});
      },
      NimbleIterator::Options{})};
  iterator.seekToFirst();

  // All four rows of the first stripe come from one decoder call.
  for (int32_t row{0}; row < 4; ++row) {
    EXPECT_TRUE(iterator.key().empty());
    EXPECT_TRUE(iterator.value().empty());
    iterator.next();
  }
  EXPECT_EQ(numDecoderCalls.load(std::memory_order_relaxed), 1);
}

TEST_P(NimbleRowIteratorTest, rowIteratorReadsAheadAndDecodesOnConsumer) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{12};
  constexpr int32_t kMaxPrefetchStripes{4};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  std::mutex mutex;
  std::set<std::thread::id> decoderThreads;
  int32_t numDecoderCalls{0};
  const auto decodeRows = compactRowDecoder();
  auto iterator = createEntryIterator(
      projectedSubfields,
      [&](const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& keyCursor,
          folly::IOBufQueue& keyStorage,
          folly::IOBufQueue& valueStorage,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        {
          const std::lock_guard<std::mutex> lock{mutex};
          decoderThreads.insert(std::this_thread::get_id());
          ++numDecoderCalls;
        }
        decodeRows(rows, keyCursor, keyStorage, valueStorage, keys, values);
      },
      NimbleIterator::Options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = kMaxPrefetchStripes,
      });
  iterator->seekToFirst();

  std::promise<void> producerPaused;
  auto paused = producerPaused.get_future();
  executor->add([&] { producerPaused.set_value(); });
  ASSERT_EQ(
      paused.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  EXPECT_EQ(iterator->stats().numReadStripes, kMaxPrefetchStripes);
  EXPECT_EQ(iterator->stats().numMaterializedRows, 0);
  {
    const std::lock_guard<std::mutex> lock{mutex};
    EXPECT_EQ(numDecoderCalls, 0);
  }

  while (iterator->valid()) {
    iterator->value();
    iterator->nextRow();
  }
  iterator->reset();
  const std::lock_guard<std::mutex> lock{mutex};
  EXPECT_EQ(numDecoderCalls, kNumStripes);
  EXPECT_EQ(decoderThreads.size(), 1);
  EXPECT_EQ(decoderThreads.count(std::this_thread::get_id()), 1);
}

TEST_P(NimbleRowIteratorTest, rowIteratorResetSkipsPrefetchedDecoding) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{12};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  std::atomic_int32_t numCalls{0};
  const auto decodeRows = compactRowDecoder();
  auto iterator = createEntryIterator(
      projectedSubfields,
      [&](const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& keyCursor,
          folly::IOBufQueue& keyStorage,
          folly::IOBufQueue& valueStorage,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        numCalls.fetch_add(1);
        decodeRows(rows, keyCursor, keyStorage, valueStorage, keys, values);
      },
      NimbleIterator::Options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 4,
      });
  iterator->seekToFirst();
  std::promise<void> producerPaused;
  auto paused = producerPaused.get_future();
  executor->add([&] { producerPaused.set_value(); });
  ASSERT_EQ(
      paused.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  EXPECT_EQ(iterator->stats().numReadStripes, 4);
  iterator->key();
  iterator->reset();

  EXPECT_EQ(numCalls.load(), 1);
  EXPECT_EQ(iterator->stats().numMaterializedRows, kRowsPerStripe);
}

TEST_P(NimbleRowIteratorTest, entryIteratorMovesOverTheEntriesOfEveryRow) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(2);
    auto iterator = createEntryIterator(
        projectedSubfields,
        decodeTwoEntriesPerRow,
        NimbleIterator::Options{
            .prefetchExecutor = prefetch ? folly::getKeepAliveToken(*executor)
                                         : folly::Executor::KeepAlive<>{},
            .maxPrefetchStripes = 4,
        });
    iterator->seekToFirst();

    std::vector<std::pair<std::string, std::string>> entries;
    while (iterator->valid()) {
      entries.emplace_back(iterator->key(), iterator->value());
      iterator->nextRow();
    }

    std::vector<std::pair<std::string, std::string>> expectedEntries;
    for (int32_t row{0}; row < kRowsPerStripe * kNumStripes; ++row) {
      for (const char suffix : {'a', 'b'}) {
        expectedEntries.emplace_back(
            fmt::format("k{:06}{}", row * 10, suffix),
            fmt::format("v{}{}", row * 10, suffix));
      }
    }
    EXPECT_EQ(entries, expectedEntries);
    EXPECT_FALSE(iterator->valid());
    iterator->reset();
  }
}

TEST_P(NimbleRowIteratorTest, entryIteratorStartsAtTheSelectedRows) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto keysOf = [](NimbleIterator& iterator) {
    std::vector<std::string> keys;
    while (iterator.valid()) {
      keys.emplace_back(iterator.key());
      iterator.nextRow();
    }
    return keys;
  };
  auto iterator = createEntryIterator(
      projectedSubfields, decodeTwoEntriesPerRow, NimbleIterator::Options{});

  // A range that starts in the middle of a stripe decodes only its rows, to the
  // end of the scan.
  auto bounds = makePointLookup(rowType, {"key"}, 6);
  bounds.upperKey.reset();
  iterator->seek(bounds);
  std::vector<std::string> expected;
  for (int32_t row{6}; row < kRowsPerStripe * kNumStripes; ++row) {
    expected.push_back(fmt::format("k{:06}a", row * 10));
    expected.push_back(fmt::format("k{:06}b", row * 10));
  }
  EXPECT_EQ(keysOf(*iterator), expected);

  // A point lookup reads the entries of its row.
  iterator->seek(makePointLookup(rowType, {"key"}, 6));
  EXPECT_THAT(
      keysOf(*iterator), ::testing::ElementsAre("k000060a", "k000060b"));
  EXPECT_EQ(iterator->stats().numMaterializedRows, 6 + 1);

  // A key without rows leaves the iterator invalid.
  iterator->seek(makePointLookup(rowType, {"key"}, 1'000));
  EXPECT_FALSE(iterator->valid());
}

// The row iterator keeps the decoded rows of a stripe after the decoder
// converted them, to decode the next stripe into them, and frees them on
// reset().
TEST_P(NimbleRowIteratorTest, rowIteratorFreesTheDecodedRowsOnReset) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  std::vector<std::weak_ptr<velox::RowVector>> decodedRows;
  const auto decodeRows = compactRowDecoder();
  const auto iterator = createEntryIterator(
      projectedSubfields,
      [&](const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& keyCursor,
          folly::IOBufQueue& keyStorage,
          folly::IOBufQueue& valueStorage,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        decodedRows.emplace_back(rows);
        decodeRows(rows, keyCursor, keyStorage, valueStorage, keys, values);
      },
      NimbleIterator::Options{});
  iterator->seekToFirst();

  const auto actual = velox::row::CompactRow::deserialize(
      {iterator->value()}, ROW("value", INTEGER()), leafPool_.get());
  ASSERT_EQ(decodedRows.size(), 1);
  EXPECT_FALSE(decodedRows[0].expired());
  iterator->reset();
  EXPECT_TRUE(decodedRows[0].expired());
  expectVectorRows(
      actual,
      vectorMaker_->rowVector(
          {"value"}, {vectorMaker_->flatVector<int32_t>({0})}));
}

// The row iterator gives the decoder empty storages for every stripe, so that
// the size of an earlier stripe does not reserve room in the entries of the
// later ones, which the prefetch holds at the same time.
TEST_P(NimbleRowIteratorTest, rowIteratorReservesNoStorageForTheDecoder) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto decodeRows = compactRowDecoder();
  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(2);
    std::atomic_int32_t numDecodedStripes{0};
    std::atomic_int32_t numReservedStorages{0};
    auto iterator = createEntryIterator(
        projectedSubfields,
        [&](const velox::RowVectorPtr& rows,
            nimble::index::KeyCursor& keyCursor,
            folly::IOBufQueue& keyStorage,
            folly::IOBufQueue& valueStorage,
            std::vector<std::string_view>& keys,
            std::vector<std::string_view>& values) {
          ++numDecodedStripes;
          if (keyStorage.front() != nullptr ||
              valueStorage.front() != nullptr) {
            ++numReservedStorages;
          }
          decodeRows(rows, keyCursor, keyStorage, valueStorage, keys, values);
        },
        NimbleIterator::Options{
            .prefetchExecutor = prefetch ? folly::getKeepAliveToken(*executor)
                                         : folly::Executor::KeepAlive<>{},
            .maxPrefetchStripes = 4,
        });
    iterator->seekToFirst();
    while (iterator->valid()) {
      iterator->value();
      iterator->nextRow();
    }
    EXPECT_EQ(numDecodedStripes, kNumStripes);
    EXPECT_EQ(numReservedStorages, 0);
  }
}

TEST_P(NimbleRowIteratorTest, entryIteratorKeepsRowsThatTheStorageOwns) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(2);
    auto iterator = createEntryIterator(
        projectedSubfields,
        [](const velox::RowVectorPtr& rows,
           nimble::index::KeyCursor& keyCursor,
           folly::IOBufQueue& keyStorage,
           folly::IOBufQueue& valueStorage,
           std::vector<std::string_view>& keys,
           std::vector<std::string_view>& values) {
          // The values view the numbers of the rows, which an IOBuf of the
          // value storage keeps alive.
          auto freeBuffer = [rows](uint8_t* buffer) { delete[] buffer; };
          valueStorage.append(
              folly::IOBuf::takeOwnership(
                  std::unique_ptr<uint8_t, decltype(freeBuffer)>{
                      new uint8_t[1], std::move(freeBuffer)},
                  1));
          const auto* numbers = rows->childAt(0)->asFlatVector<int32_t>();
          for (velox::vector_size_t row{0}; row < rows->size(); ++row) {
            appendBytes(keyCursor.next(), keyStorage, keys);
            values.emplace_back(
                reinterpret_cast<const char*>(numbers->rawValues() + row),
                sizeof(int32_t));
          }
        },
        NimbleIterator::Options{
            .prefetchExecutor = prefetch ? folly::getKeepAliveToken(*executor)
                                         : folly::Executor::KeepAlive<>{},
            .maxPrefetchStripes = 4,
        });
    iterator->seekToFirst();

    for (int32_t row{0}; row < kRowsPerStripe * kNumStripes; ++row) {
      SCOPED_TRACE(row);
      ASSERT_TRUE(iterator->valid());
      const auto value = iterator->value();
      ASSERT_EQ(value.size(), sizeof(int32_t));
      int32_t number{0};
      std::memcpy(&number, value.data(), sizeof(number));
      EXPECT_EQ(number, row * 10);
      iterator->nextRow();
    }
    EXPECT_FALSE(iterator->valid());
    iterator->reset();
  }
}

TEST_P(NimbleRowIteratorTest, entryIteratorReportsWhetherAnEntryFollows) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  // Moves over the entries, and returns whether an entry follows each of them.
  const auto hasNextOfEntries = [](NimbleIterator& iterator) {
    std::vector<bool> hasNext;
    while (iterator.valid()) {
      hasNext.push_back(iterator.hasNext());
      iterator.nextRow();
    }
    return hasNext;
  };
  // Only the last of 'numEntries' has no entry after it.
  const auto expectedHasNext = [](size_t numEntries) {
    std::vector<bool> hasNext(numEntries, true);
    hasNext.back() = false;
    return hasNext;
  };

  for (const bool prefetch : {false, true}) {
    for (const bool twoEntriesPerRow : {false, true}) {
      SCOPED_TRACE(
          fmt::format(
              "prefetch={}, twoEntriesPerRow={}", prefetch, twoEntriesPerRow));
      auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(2);
      auto iterator = createEntryIterator(
          projectedSubfields,
          twoEntriesPerRow
              ? NimbleIterator::EntryDecoder{decodeTwoEntriesPerRow}
              : compactRowDecoder(),
          NimbleIterator::Options{
              .prefetchExecutor = prefetch ? folly::getKeepAliveToken(*executor)
                                           : folly::Executor::KeepAlive<>{},
              .maxPrefetchStripes = 4,
          });
      const size_t numEntriesPerRow = twoEntriesPerRow ? 2 : 1;

      iterator->seekToFirst();
      EXPECT_EQ(
          hasNextOfEntries(*iterator),
          expectedHasNext(kRowsPerStripe * kNumStripes * numEntriesPerRow));

      // A range that starts and ends inside stripes.
      iterator->seek(makeRangeLookup(rowType, {"key"}, 2, 6));
      EXPECT_EQ(
          hasNextOfEntries(*iterator), expectedHasNext(4 * numEntriesPerRow));

      iterator->seek(makePointLookup(rowType, {"key"}, 6));
      EXPECT_EQ(hasNextOfEntries(*iterator), expectedHasNext(numEntriesPerRow));
      iterator->reset();
    }
  }
}

TEST_P(NimbleRowIteratorTest, hasNextDoesNotLoadTheNextStripe) {
  constexpr int32_t kRowsPerStripe{4};
  writeResumeKeyTestData(kRowsPerStripe, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  iterator.seekToFirst();
  EXPECT_EQ(iterator.stats().numReadStripes, 0);

  // Loads the stripe of the current entry.
  EXPECT_TRUE(iterator.hasNext());
  EXPECT_EQ(iterator.stats().numReadStripes, 1);

  // The last entry of the stripe is followed by the first entry of the next
  // stripe, which it does not load.
  for (int32_t entry{1}; entry < kRowsPerStripe; ++entry) {
    iterator.next();
  }
  EXPECT_TRUE(iterator.hasNext());
  EXPECT_EQ(iterator.stats().numReadStripes, 1);
}

TEST_P(NimbleRowIteratorTest, hasNextRequiresAnEntry) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/1);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  const std::string kMessage{"Iterator is not positioned on a row"};

  // The iterator has not sought yet.
  NIMBLE_ASSERT_THROW(iterator.hasNext(), kMessage);

  // A key without rows leaves the iterator invalid.
  iterator.seek(makePointLookup(rowType, {"key"}, 1'000));
  NIMBLE_ASSERT_THROW(iterator.hasNext(), kMessage);

  // The iterator moved past the last entry.
  iterator.seekToFirst();
  while (iterator.valid()) {
    iterator.next();
  }
  NIMBLE_ASSERT_THROW(iterator.hasNext(), kMessage);
}

TEST_P(NimbleRowIteratorTest, entryIteratorFailsAtAStripeThatFailsToDecode) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  for (const bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(2);
    auto iterator = createEntryIterator(
        projectedSubfields,
        [](const velox::RowVectorPtr& rows,
           nimble::index::KeyCursor& keyCursor,
           folly::IOBufQueue& keyStorage,
           folly::IOBufQueue& valueStorage,
           std::vector<std::string_view>& keys,
           std::vector<std::string_view>& values) {
          NIMBLE_CHECK_NE(
              rows->childAt(0)->asFlatVector<int32_t>()->valueAt(0),
              40,
              "Decoder failed");
          decodeTwoEntriesPerRow(
              rows, keyCursor, keyStorage, valueStorage, keys, values);
        },
        NimbleIterator::Options{
            .prefetchExecutor = prefetch ? folly::getKeepAliveToken(*executor)
                                         : folly::Executor::KeepAlive<>{},
            .maxPrefetchStripes = 4,
        });

    for (const bool restart : {false, true}) {
      SCOPED_TRACE(restart);
      iterator->seekToFirst();
      // The entries of the stripe before the failing one are served.
      for (int32_t entry{0}; entry < 2 * kRowsPerStripe; ++entry) {
        iterator->key();
        iterator->nextRow();
      }
      ASSERT_TRUE(iterator->valid());
      // The caller must seek again after a prefetched decode failure.
      NIMBLE_ASSERT_THROW(iterator->key(), "Decoder failed");
      if (!prefetch) {
        NIMBLE_ASSERT_THROW(iterator->value(), "Decoder failed");
      }
    }
  }
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

// A point-lookup iterator keeps the stripe of its last lookup until it decodes
// another stripe or resets, so that later lookups in the stripe, in any order,
// decode without loading it again. A scan iterator keeps no stripe.
TEST_P(NimbleRowIteratorTest, rowIteratorKeepsTheStripeOfAPointLookup) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto expectLookup = [&](NimbleIterator& iterator, int64_t key) {
    iterator.seek(makePointLookup(rowType, {"key"}, key));
    if (key >= kRowsPerStripe * kNumStripes) {
      EXPECT_FALSE(iterator.valid());
      return;
    }
    ASSERT_TRUE(iterator.valid());
    const auto actual = velox::row::CompactRow::deserialize(
        {iterator.value()}, ROW("value", INTEGER()), leafPool_.get());
    expectVectorRows(
        actual,
        vectorMaker_->rowVector(
            {"value"},
            {vectorMaker_->flatVector<int32_t>(
                {static_cast<int32_t>(key * 10)})}));
  };

  struct TestParam {
    std::string description;
    int64_t key;
    bool reset;
    uint32_t numReadStripes;

    std::string debugString() const {
      return fmt::format(
          "{}: key {}, reset {}, numReadStripes {}",
          description,
          key,
          reset,
          numReadStripes);
    }
  };
  const std::vector<TestParam> testSettings{
      {"loads stripe 0 and keeps it", 1, false, 1},
      {"reads on in the kept stripe", 2, false, 1},
      {"reads an earlier row of the kept stripe", 0, false, 1},
      {"a miss keeps the stripe", 1'000, false, 1},
      {"reads the last row of the kept stripe", 3, false, 1},
      {"keeps the stripe after its last row", 2, false, 1},
      {"loads stripe 1, which replaces stripe 0", 5, false, 2},
      {"loads stripe 0 again", 0, false, 3},
      {"reset() releases the stripe", 0, true, 4},
  };
  const auto pointLookupIterator = createRowIterator(
      projectedSubfields, NimbleIterator::Options{.forGet = true});
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    if (testData.reset) {
      pointLookupIterator->reset();
    }
    expectLookup(*pointLookupIterator, testData.key);
    EXPECT_EQ(
        pointLookupIterator->stats().numReadStripes, testData.numReadStripes);
  }

  const auto scanIterator =
      createRowIterator(projectedSubfields, NimbleIterator::Options{});
  for (const uint32_t numReadStripes : {1, 2}) {
    SCOPED_TRACE(numReadStripes);
    expectLookup(*scanIterator, 1);
    EXPECT_EQ(scanIterator->stats().numReadStripes, numReadStripes);
  }
}

// A stripe that fails to bind is not kept, so that the next lookup in it loads
// it again.
TEST_P(NimbleRowIteratorTest, rowIteratorDoesNotKeepAStripeThatFailsToBind) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/2);
  // Makes the first chunk of each stream of stripe 0 longer than its stream:
  // the fourth byte of a chunk header is the high byte of its length.
  corruptStripe0Streams(
      [](const TabletReader::StreamMetadata& /*location*/) -> uint64_t {
        return 3;
      });

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto iterator = createRowIterator(
      projectedSubfields, NimbleIterator::Options{.forGet = true});
  for (const uint32_t numReadStripes : {1, 2}) {
    SCOPED_TRACE(numReadStripes);
    iterator->seek(makePointLookup(rowType, {"key"}, 1));
    NIMBLE_ASSERT_FILE_THROW(
        iterator->value(), "Chunk data exceeds stream boundary");
    EXPECT_EQ(iterator->stats().numReadStripes, numReadStripes);
  }

  // A lookup in the other stripe decodes.
  iterator->seek(makePointLookup(rowType, {"key"}, 5));
  const auto actual = velox::row::CompactRow::deserialize(
      {iterator->value()}, ROW("value", INTEGER()), leafPool_.get());
  expectVectorRows(
      actual,
      vectorMaker_->rowVector(
          {"value"}, {vectorMaker_->flatVector<int32_t>({50})}));
}

// Ranges of rows decode from the kept stripe too, and the keys of their entries
// stay aligned with their rows, also when a range goes on into the next stripe.
TEST_P(NimbleRowIteratorTest, entryIteratorServesRangesFromAKeptStripe) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{3};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto iterator = createEntryIterator(
      projectedSubfields,
      decodeTwoEntriesPerRow,
      NimbleIterator::Options{.forGet = true});

  struct TestParam {
    std::string description;
    int64_t startKey;
    int64_t endKey;
    uint32_t numReadStripes;

    std::string debugString() const {
      return fmt::format(
          "{}: keys [{}, {}), numReadStripes {}",
          description,
          startKey,
          endKey,
          numReadStripes);
    }
  };
  const std::vector<TestParam> testSettings{
      {"loads stripe 0 and keeps it", 1, 3, 1},
      {"ends in the kept stripe 0 and keeps stripe 1", 3, 5, 2},
      {"reads on in the kept stripe", 6, 7, 2},
      {"reads the whole kept stripe", 4, 8, 2},
      {"reads an earlier row of the kept stripe", 5, 6, 2},
      {"loads stripe 0 again", 0, 2, 3},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    iterator->seek(
        makeRangeLookup(rowType, {"key"}, testData.startKey, testData.endKey));
    std::vector<std::pair<std::string, std::string>> entries;
    while (iterator->valid()) {
      entries.emplace_back(iterator->key(), iterator->value());
      iterator->nextRow();
    }

    std::vector<std::pair<std::string, std::string>> expectedEntries;
    for (auto row = testData.startKey; row < testData.endKey; ++row) {
      for (const char suffix : {'a', 'b'}) {
        expectedEntries.emplace_back(
            fmt::format("k{:06}{}", row * 10, suffix),
            fmt::format("v{}{}", row * 10, suffix));
      }
    }
    EXPECT_EQ(entries, expectedEntries);
    EXPECT_EQ(iterator->stats().numReadStripes, testData.numReadStripes);
  }
}

// A get iterator does not prefetch, also when its options set a prefetch
// executor, so that its lookups read the stripe that it holds.
TEST_P(NimbleRowIteratorTest, getIteratorDoesNotPrefetch) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  const auto iterator = createRowIterator(
      projectedSubfields,
      NimbleIterator::Options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 1,
          .forGet = true,
      });
  for (const auto& [key, numReadStripes] :
       std::vector<std::pair<int64_t, uint32_t>>{
           {1, 1}, {2, 1}, {0, 1}, {5, 2}}) {
    SCOPED_TRACE(key);
    iterator->seek(makePointLookup(rowType, {"key"}, key));
    ASSERT_TRUE(iterator->valid());
    const auto actual = velox::row::CompactRow::deserialize(
        {iterator->value()}, ROW("value", INTEGER()), leafPool_.get());
    expectVectorRows(
        actual,
        vectorMaker_->rowVector(
            {"value"},
            {vectorMaker_->flatVector<int32_t>(
                {static_cast<int32_t>(key * 10)})}));
    EXPECT_EQ(iterator->stats().numReadStripes, numReadStripes);
  }
  iterator->reset();
}

// The row iterator decodes into the output of its last decode when the entry
// decoder released it, for gets and for the stripes of a scan, and into a new
// vector when the decoder still holds it. reset() frees the output.
TEST_P(NimbleRowIteratorTest, rowIteratorReusesTheOutputOfTheLastDecode) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto decodeRows = compactRowDecoder();
  bool holdRows{false};
  std::vector<velox::RowVectorPtr> heldRows;
  std::vector<std::weak_ptr<velox::RowVector>> outputs;
  const auto iterator = createEntryIterator(
      projectedSubfields,
      [&](const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& keyCursor,
          folly::IOBufQueue& keyStorage,
          folly::IOBufQueue& valueStorage,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        outputs.emplace_back(rows);
        if (holdRows) {
          heldRows.emplace_back(rows);
        }
        decodeRows(rows, keyCursor, keyStorage, valueStorage, keys, values);
      },
      NimbleIterator::Options{.forGet = true});
  const auto expectGet = [&](int64_t key) {
    iterator->seek(makePointLookup(rowType, {"key"}, key));
    ASSERT_TRUE(iterator->valid());
    const auto actual = velox::row::CompactRow::deserialize(
        {iterator->value()}, ROW("value", INTEGER()), leafPool_.get());
    expectVectorRows(
        actual,
        vectorMaker_->rowVector(
            {"value"},
            {vectorMaker_->flatVector<int32_t>(
                {static_cast<int32_t>(key * 10)})}));
  };

  for (const int64_t key : {1, 2, 0}) {
    SCOPED_TRACE(key);
    expectGet(key);
  }
  // The materializer holds the output between the gets and reuses it.
  ASSERT_EQ(outputs.size(), 3);
  ASSERT_NE(outputs[0].lock(), nullptr);
  EXPECT_EQ(outputs[1].lock(), outputs[0].lock());
  EXPECT_EQ(outputs[2].lock(), outputs[0].lock());

  // A decoder that holds the rows makes the next get decode into a new vector.
  holdRows = true;
  for (const int64_t key : {3, 1}) {
    SCOPED_TRACE(key);
    expectGet(key);
  }
  ASSERT_EQ(outputs.size(), 5);
  ASSERT_NE(outputs[3].lock(), nullptr);
  EXPECT_NE(outputs[4].lock(), outputs[3].lock());

  // A get in another stripe reuses the output too, once the decoder releases
  // it.
  holdRows = false;
  heldRows.clear();
  EXPECT_TRUE(outputs[0].expired());
  expectGet(5);
  expectGet(6);
  ASSERT_EQ(outputs.size(), 7);
  ASSERT_NE(outputs[4].lock(), nullptr);
  EXPECT_EQ(outputs[5].lock(), outputs[4].lock());
  EXPECT_EQ(outputs[6].lock(), outputs[4].lock());

  iterator->reset();
  EXPECT_TRUE(outputs[6].expired());

  // A scan decodes its stripes into the same output.
  outputs.clear();
  const auto scanIterator = createEntryIterator(
      projectedSubfields,
      [&](const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& keyCursor,
          folly::IOBufQueue& keyStorage,
          folly::IOBufQueue& valueStorage,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        outputs.emplace_back(rows);
        decodeRows(rows, keyCursor, keyStorage, valueStorage, keys, values);
      },
      NimbleIterator::Options{});
  scanIterator->seekToFirst();
  for (int32_t row{0}; row < 12; ++row) {
    ASSERT_TRUE(scanIterator->valid());
    const auto actual = velox::row::CompactRow::deserialize(
        {scanIterator->value()}, ROW("value", INTEGER()), leafPool_.get());
    expectVectorRows(
        actual,
        vectorMaker_->rowVector(
            {"value"}, {vectorMaker_->flatVector<int32_t>({row * 10})}));
    scanIterator->nextRow();
  }
  EXPECT_FALSE(scanIterator->valid());
  ASSERT_EQ(outputs.size(), 3);
  ASSERT_NE(outputs[0].lock(), nullptr);
  EXPECT_EQ(outputs[1].lock(), outputs[0].lock());
  EXPECT_EQ(outputs[2].lock(), outputs[0].lock());
  scanIterator->reset();
  EXPECT_TRUE(outputs[2].expired());
}

// The kept stripe holds memory until the iterator resets, so that an idle
// iterator holds no stripe.
TEST_P(NimbleRowIteratorTest, rowIteratorFreesTheKeptStripeOnReset) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  const auto iterator = createRowIterator(
      projectedSubfields, NimbleIterator::Options{.forGet = true});
  const auto lookup = [&](int64_t key) {
    iterator->seek(makePointLookup(rowType, {"key"}, key));
    ASSERT_TRUE(iterator->valid());
    iterator->value();
  };

  // The first lookup builds the stream reader, which the iterator keeps.
  lookup(1);
  iterator->reset();
  const auto idleBytes = leafPool_->usedBytes();

  lookup(1);
  EXPECT_GT(leafPool_->usedBytes(), idleBytes);
  iterator->reset();
  EXPECT_LE(leafPool_->usedBytes(), idleBytes);
}

TEST_P(
    NimbleRowIteratorTest,
    rowIteratorTimesDecodingAndConversionOnlyWhenEnabled) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");

  for (const bool collectTimings : {true, false}) {
    SCOPED_TRACE(collectTimings);
    NimbleIterator::Options options;
    options.collectTimings = collectTimings;
    TestNimbleIterator iterator{createRowIterator(projectedSubfields, options)};
    iterator.seekToFirst();
    EXPECT_EQ(iterator.stats().decodingTiming, velox::CpuWallTiming{});
    EXPECT_EQ(iterator.stats().conversionTiming, velox::CpuWallTiming{});
    iterator.value();
    // The first stripe is decoded once and converted once.
    const auto stats = iterator.stats();
    if (collectTimings) {
      EXPECT_EQ(stats.decodingTiming.count, 1);
      EXPECT_EQ(stats.conversionTiming.count, 1);
    } else {
      EXPECT_EQ(stats.decodingTiming, velox::CpuWallTiming{});
      EXPECT_EQ(stats.conversionTiming, velox::CpuWallTiming{});
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

// Debug-only: counting decodes relies on a TestValue hook, which is compiled
// out under NDEBUG.
DEBUG_ONLY_TEST_P(
    NimbleRowIteratorTest,
    rowIteratorReadsTheKeyChunksOfItsSeek) {
  // The index keeps one decoded chunk of its partition for all the iterators
  // of the file.
  writeKeyChunkTestData(
      /*rowsPerStripe=*/4, /*numStripes=*/3, /*rowsPerKeyChunk=*/4);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  const auto encodeKey = [&](int64_t key) {
    return *makePointLookup(rowType, {"key"}, key).lowerKey;
  };
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator first{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  TestNimbleIterator second{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  TestNimbleIterator scan{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};

  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  // Fires only after a cache miss, so it counts decodes rather than reads.
  std::atomic_uint32_t numDecodes{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(index::DecodedKeyChunk*)>(
          [&](index::DecodedKeyChunk* /*decodedChunk*/) { ++numDecodes; }));

  first.seek(encodeKey(1));
  EXPECT_EQ(numDecodes.load(), 1);
  // The iterators share the index, so a seek in the same chunk decodes nothing.
  second.seek(encodeKey(2));
  EXPECT_EQ(numDecodes.load(), 1);
  // A seek in the last stripe replaces the chunk of the first stripe in the
  // index.
  second.seek(encodeKey(9));
  EXPECT_EQ(numDecodes.load(), 2);

  // The first iterator reads its key from the chunk that its seek decoded.
  EXPECT_EQ(first.key(), encodeKey(1));
  EXPECT_EQ(numDecodes.load(), 2);

  // A scan from the first row has no key chunks of a seek, so it decodes the
  // chunk of the first stripe, which replaces the chunk of the last stripe in
  // the index.
  scan.seekToFirst();
  EXPECT_EQ(scan.key(), encodeKey(0));
  EXPECT_EQ(numDecodes.load(), 3);

  // The second iterator reads its key from the chunk that its seek decoded.
  EXPECT_EQ(second.key(), encodeKey(9));
  EXPECT_EQ(numDecodes.load(), 3);
}

// Debug-only: counting decodes relies on a TestValue hook, which is compiled
// out under NDEBUG.
DEBUG_ONLY_TEST_P(
    NimbleRowIteratorTest,
    rowIteratorHoldsOnlyTheKeyChunkThatItReads) {
  writeKeyChunkTestData(
      /*rowsPerStripe=*/4, /*numStripes=*/3, /*rowsPerKeyChunk=*/4);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator iterator{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};

  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  // Fires only after a cache miss, so it counts decodes rather than reads.
  std::atomic_uint32_t numDecodes{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(index::DecodedKeyChunk*)>(
          [&](index::DecodedKeyChunk* /*decodedChunk*/) { ++numDecodes; }));

  // The bounds of the rows 1 to 8 are in the chunks of the first and of the
  // last stripe, which the seek decodes.
  iterator.seek(makeRangeLookup(rowType, {"key"}, 1, 9));
  EXPECT_EQ(numDecodes.load(), 2);

  // The first stripe reads its keys from the chunk of the first row, which the
  // key cursor of the seek keeps.
  for (int64_t row{1}; row < 4; ++row) {
    SCOPED_TRACE(row);
    EXPECT_EQ(iterator.key(), *makePointLookup(rowType, {"key"}, row).lowerKey);
    iterator.next();
  }
  EXPECT_EQ(numDecodes.load(), 2);

  // The second stripe decodes its chunk, which replaces the chunk of the last
  // stripe in the index. The key cursor releases the chunk of the first stripe.
  for (int64_t row{4}; row < 8; ++row) {
    SCOPED_TRACE(row);
    EXPECT_EQ(iterator.key(), *makePointLookup(rowType, {"key"}, row).lowerKey);
    iterator.next();
  }
  EXPECT_EQ(numDecodes.load(), 3);

  // The key cursor kept only the chunk of the first row of the seek, so the
  // last stripe decodes its chunk again.
  EXPECT_EQ(iterator.key(), *makePointLookup(rowType, {"key"}, 8).lowerKey);
  EXPECT_EQ(numDecodes.load(), 4);
  iterator.next();
  EXPECT_FALSE(iterator.valid());
}

// Debug-only: counting decodes relies on a TestValue hook, which is compiled
// out under NDEBUG.
DEBUG_ONLY_TEST_P(
    NimbleRowIteratorTest,
    rowIteratorReadsAKeyChunkOnceAcrossStripes) {
  // The first key chunk holds the rows of the first two stripes.
  writeKeyChunkTestData(
      /*rowsPerStripe=*/4, /*numStripes=*/3, /*rowsPerKeyChunk=*/8);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  const auto encodeKey = [&](int64_t key) {
    return *makePointLookup(rowType, {"key"}, key).lowerKey;
  };
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  TestNimbleIterator scan{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};
  TestNimbleIterator get{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};

  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  // Fires only after a cache miss, so it counts decodes rather than reads.
  std::atomic_uint32_t numDecodes{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(index::DecodedKeyChunk*)>(
          [&](index::DecodedKeyChunk* /*decodedChunk*/) { ++numDecodes; }));

  scan.seekToFirst();
  EXPECT_EQ(scan.key(), encodeKey(0));
  EXPECT_EQ(numDecodes.load(), 1);
  // A get of the last chunk replaces the first chunk in the index.
  get.seek(encodeKey(9));
  EXPECT_EQ(get.key(), encodeKey(9));
  EXPECT_EQ(numDecodes.load(), 2);

  // The second stripe reads on from the key cursor of the scan, which holds
  // the first chunk.
  for (int64_t row{0}; row < 8; ++row) {
    SCOPED_TRACE(row);
    EXPECT_EQ(scan.key(), encodeKey(row));
    scan.next();
  }
  EXPECT_EQ(numDecodes.load(), 2);

  // The index holds the last chunk.
  EXPECT_EQ(scan.key(), encodeKey(8));
  EXPECT_EQ(numDecodes.load(), 2);
}

// Debug-only: counting decodes relies on a TestValue hook, which is compiled
// out under NDEBUG.
DEBUG_ONLY_TEST_P(
    NimbleRowIteratorTest,
    rowIteratorSkipsTheKeysThatItsDecoderLeavesUnread) {
  // The first key chunk holds the rows of the first two stripes.
  writeKeyChunkTestData(
      /*rowsPerStripe=*/4, /*numStripes=*/3, /*rowsPerKeyChunk=*/8);
  const auto rowType = ROW({{"key", BIGINT()}, {"value", INTEGER()}});
  const auto encodeKey = [&](int64_t key) {
    return *makePointLookup(rowType, {"key"}, key).lowerKey;
  };
  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  // Reads the key of the first row of each stripe from the key cursor, and
  // takes the keys of the entries from the values, which are ten times the
  // keys.
  std::vector<std::string> firstKeys;
  TestNimbleIterator scan{createEntryIterator(
      projectedSubfields,
      [&](const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& keyCursor,
          folly::IOBufQueue& keyStorage,
          folly::IOBufQueue& valueStorage,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        firstKeys.emplace_back(keyCursor.next());
        const auto* numbers = rows->childAt(0)->asFlatVector<int32_t>();
        for (velox::vector_size_t row{0}; row < rows->size(); ++row) {
          appendBytes(encodeKey(numbers->valueAt(row) / 10), keyStorage, keys);
          appendBytes(
              std::to_string(numbers->valueAt(row)), valueStorage, values);
        }
      },
      NimbleIterator::Options{})};
  TestNimbleIterator get{
      createRowIterator(projectedSubfields, NimbleIterator::Options{})};

  const bool wasEnabled = velox::common::testutil::TestValue::enabled();
  velox::common::testutil::TestValue::enable();
  auto restoreTestValue = folly::makeGuard([&] {
    if (!wasEnabled) {
      velox::common::testutil::TestValue::disable();
    }
  });
  // Fires only after a cache miss, so it counts decodes rather than reads.
  std::atomic_uint32_t numDecodes{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(index::DecodedKeyChunk*)>(
          [&](index::DecodedKeyChunk* /*decodedChunk*/) { ++numDecodes; }));

  scan.seekToFirst();
  EXPECT_EQ(scan.key(), encodeKey(0));
  EXPECT_EQ(numDecodes.load(), 1);
  // A get of the last chunk replaces the first chunk in the index.
  get.seek(encodeKey(9));
  EXPECT_EQ(get.key(), encodeKey(9));
  EXPECT_EQ(numDecodes.load(), 2);

  // The key cursor of the scan skipped the keys of the first stripe that the
  // decoder left unread, and holds the first chunk for the second stripe.
  for (int64_t row{0}; row < 8; ++row) {
    SCOPED_TRACE(row);
    EXPECT_EQ(scan.key(), encodeKey(row));
    scan.next();
  }
  EXPECT_EQ(numDecodes.load(), 2);
  EXPECT_EQ(scan.key(), encodeKey(8));
  EXPECT_EQ(numDecodes.load(), 2);
  EXPECT_EQ(
      firstKeys,
      (std::vector<std::string>{encodeKey(0), encodeKey(4), encodeKey(8)}));
}

INSTANTIATE_TEST_SUITE_P(
    AllCacheParams,
    NimbleRowIteratorTest,
    ::testing::ValuesIn(NimbleRowIteratorTest::getIteratorTestParams()),
    NimbleRowIteratorTest::iteratorTestParamName);

} // namespace facebook::nimble::test
