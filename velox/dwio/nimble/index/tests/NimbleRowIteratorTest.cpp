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
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble::test {

class NimbleRowIteratorTest : public NimbleIndexProjectorTest {};

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

TEST_P(NimbleRowIteratorTest, rowIteratorDecodesPrefetchedStripesAhead) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{12};
  constexpr int32_t kMaxPrefetchStripes{4};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(4);
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

  // The prefetch decodes its stripe limit before the iterator reads an entry.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{10};
  const auto numCalls = [&] {
    const std::lock_guard<std::mutex> lock{mutex};
    return numDecoderCalls;
  };
  while (numCalls() < kMaxPrefetchStripes) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline);
    std::this_thread::yield();
  }

  while (iterator->valid()) {
    iterator->value();
    iterator->nextRow();
  }
  // Waits for the prefetch, which could still hold the decoder that refers to
  // this scope.
  iterator->reset();
  const std::lock_guard<std::mutex> lock{mutex};
  EXPECT_EQ(numDecoderCalls, kNumStripes);
  // The prefetch decodes every stripe, and the reader none.
  EXPECT_EQ(decoderThreads.count(std::this_thread::get_id()), 0);
}

TEST_P(NimbleRowIteratorTest, rowIteratorDecodesStripesConcurrently) {
  constexpr int32_t kRowsPerStripe{4};
  constexpr int32_t kNumStripes{12};
  writeResumeKeyTestData(kRowsPerStripe, kNumStripes);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(4);
  std::atomic_int32_t numCalls{0};
  std::atomic_int32_t numConcurrentCalls{0};
  const auto decodeRows = compactRowDecoder();
  auto iterator = createEntryIterator(
      projectedSubfields,
      [&](const velox::RowVectorPtr& rows,
          nimble::index::KeyCursor& keyCursor,
          folly::IOBufQueue& keyStorage,
          folly::IOBufQueue& valueStorage,
          std::vector<std::string_view>& keys,
          std::vector<std::string_view>& values) {
        if (numCalls.fetch_add(1) < 2) {
          // The two stripes of the first prefetch batch wait until both run.
          const auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds{10};
          while (numCalls.load() < 2 &&
                 std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
          }
          if (numCalls.load() >= 2) {
            numConcurrentCalls.fetch_add(1);
          }
        }
        decodeRows(rows, keyCursor, keyStorage, valueStorage, keys, values);
      },
      NimbleIterator::Options{
          .prefetchExecutor = folly::getKeepAliveToken(*executor),
          .maxPrefetchStripes = 4,
      });
  iterator->seekToFirst();
  iterator->key();
  // Waits for the prefetch, which calls the decoder that refers to this scope.
  iterator->reset();

  EXPECT_EQ(numConcurrentCalls.load(), 2);
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

TEST_P(NimbleRowIteratorTest, rowIteratorReleasesTheDecodedRows) {
  writeResumeKeyTestData(/*rowsPerBatch=*/4, /*numBatches=*/3);

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  std::vector<std::weak_ptr<velox::RowVector>> decodedRows;
  const auto decodeRows = compactRowDecoder();
  TestNimbleIterator iterator{createEntryIterator(
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
      NimbleIterator::Options{})};
  iterator.seekToFirst();

  // The entries of the first stripe outlive its decoded rows.
  const auto actual = velox::row::CompactRow::deserialize(
      {iterator.value()}, ROW("value", INTEGER()), leafPool_.get());
  ASSERT_EQ(decodedRows.size(), 1);
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
      // The failure is the same each time the iterator asks for the stripe.
      NIMBLE_ASSERT_THROW(iterator->key(), "Decoder failed");
      NIMBLE_ASSERT_THROW(iterator->value(), "Decoder failed");
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

INSTANTIATE_TEST_SUITE_P(
    AllCacheParams,
    NimbleRowIteratorTest,
    ::testing::ValuesIn(NimbleRowIteratorTest::getIteratorTestParams()),
    NimbleRowIteratorTest::iteratorTestParamName);

} // namespace facebook::nimble::test
