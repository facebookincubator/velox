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

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <folly/coro/BlockingWait.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::nimble {
namespace {

// Builds ordered, disjoint ranges with randomized gaps and lengths.
std::vector<RowRange> makeRandomRanges(uint32_t rowCount, std::mt19937& rng) {
  std::vector<RowRange> ranges;
  uint32_t row = static_cast<uint32_t>(rng() % std::min<uint32_t>(rowCount, 8));
  while (row < rowCount) {
    const auto maxLength = std::min<uint32_t>(rowCount - row, 16);
    const uint32_t length{1 + static_cast<uint32_t>(rng() % maxLength)};
    const auto end = row + length;
    ranges.emplace_back(row, end);
    row = std::min<uint32_t>(rowCount, end + static_cast<uint32_t>(rng() % 8));
  }
  return ranges;
}

class StreamReaderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("stream_reader_test");
  }

  template <typename T>
  std::string encodeChunk(
      std::span<const T> values,
      EncodingType encodingType,
      CompressionParams compressionParams = {
          .type = CompressionType::Uncompressed}) {
    Buffer buffer{*pool_};
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<EncodingType, float>>{{encodingType, 1.0}},
        CompressionOptions{},
        std::nullopt);
    const auto encoded =
        EncodingFactory::encode<T>(std::move(policy), values, buffer);
    ChunkedStreamWriter writer{buffer, compressionParams};
    std::string stream;
    for (const auto segment : writer.encode(encoded)) {
      stream.append(segment);
    }
    return stream;
  }

  template <typename T>
  std::string encodeNullableChunk(
      const std::vector<std::optional<T>>& values,
      EncodingType encodingType) {
    std::vector<T> nonNullValues;
    Vector<bool> isNonNull{pool_.get(), values.size()};
    for (size_t i{0}; i < values.size(); ++i) {
      isNonNull[i] = values[i].has_value();
      if (values[i].has_value()) {
        nonNullValues.push_back(*values[i]);
      }
    }

    Buffer buffer{*pool_};
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<EncodingType, float>>{{encodingType, 1.0}},
        CompressionOptions{},
        std::nullopt);
    const auto encoded = EncodingFactory::encodeNullable<T>(
        std::move(policy), nonNullValues, isNonNull, buffer);
    ChunkedStreamWriter writer{buffer};
    std::string stream;
    for (const auto segment : writer.encode(encoded)) {
      stream.append(segment);
    }
    return stream;
  }

  void runRead(
      StreamReader& reader,
      std::span<const std::string_view> streams,
      std::span<const RowRange> ranges,
      velox::vector_size_t outputOffset,
      velox::VectorPtr& output) {
    reader.read(streams, ranges, outputOffset, output);
  }

  std::vector<std::optional<std::vector<int64_t>>> readArrays(
      const velox::VectorPtr& output,
      velox::vector_size_t offset,
      velox::vector_size_t count) {
    const auto* arrays = output->as<velox::ArrayVector>();
    NIMBLE_CHECK_NOT_NULL(arrays);
    const auto* elements = arrays->elements()->asFlatVector<int64_t>();
    NIMBLE_CHECK_NOT_NULL(elements);
    std::vector<std::optional<std::vector<int64_t>>> result;
    result.reserve(count);
    for (velox::vector_size_t row{offset}; row < offset + count; ++row) {
      if (arrays->isNullAt(row)) {
        result.emplace_back(std::nullopt);
        continue;
      }
      std::vector<int64_t> values;
      for (velox::vector_size_t i{0}; i < arrays->sizeAt(row); ++i) {
        values.push_back(elements->valueAt(arrays->offsetAt(row) + i));
      }
      result.emplace_back(std::move(values));
    }
    return result;
  }

  std::vector<std::optional<std::vector<std::pair<std::string, int64_t>>>>
  readMaps(
      const velox::VectorPtr& output,
      velox::vector_size_t offset,
      velox::vector_size_t count) {
    const auto* maps = output->as<velox::MapVector>();
    NIMBLE_CHECK_NOT_NULL(maps);
    const auto* keys = maps->mapKeys()->asFlatVector<velox::StringView>();
    const auto* values = maps->mapValues()->asFlatVector<int64_t>();
    NIMBLE_CHECK_NOT_NULL(keys);
    NIMBLE_CHECK_NOT_NULL(values);
    std::vector<std::optional<std::vector<std::pair<std::string, int64_t>>>>
        result;
    result.reserve(count);
    for (velox::vector_size_t row{offset}; row < offset + count; ++row) {
      if (maps->isNullAt(row)) {
        result.emplace_back(std::nullopt);
        continue;
      }
      std::vector<std::pair<std::string, int64_t>> entries;
      for (velox::vector_size_t i{0}; i < maps->sizeAt(row); ++i) {
        const auto entry = maps->offsetAt(row) + i;
        entries.emplace_back(
            keys->valueAt(entry).str(), values->valueAt(entry));
      }
      result.emplace_back(std::move(entries));
    }
    return result;
  }

  std::vector<std::optional<int64_t>> readScalars(
      const velox::VectorPtr& output,
      velox::vector_size_t offset,
      velox::vector_size_t count) {
    const auto* values = output->asFlatVector<int64_t>();
    NIMBLE_CHECK_NOT_NULL(values);
    std::vector<std::optional<int64_t>> result;
    result.reserve(count);
    for (velox::vector_size_t row{offset}; row < offset + count; ++row) {
      if (values->isNullAt(row)) {
        result.emplace_back(std::nullopt);
        continue;
      }
      result.emplace_back(values->valueAt(row));
    }
    return result;
  }

  std::vector<std::optional<std::vector<std::vector<int64_t>>>>
  readNestedArrays(
      const velox::VectorPtr& output,
      velox::vector_size_t offset,
      velox::vector_size_t count) {
    const auto* outer = output->as<velox::ArrayVector>();
    NIMBLE_CHECK_NOT_NULL(outer);
    const auto* inner = outer->elements()->as<velox::ArrayVector>();
    NIMBLE_CHECK_NOT_NULL(inner);
    const auto* elements = inner->elements()->asFlatVector<int64_t>();
    NIMBLE_CHECK_NOT_NULL(elements);
    std::vector<std::optional<std::vector<std::vector<int64_t>>>> result;
    result.reserve(count);
    for (velox::vector_size_t row{offset}; row < offset + count; ++row) {
      if (outer->isNullAt(row)) {
        result.emplace_back(std::nullopt);
        continue;
      }
      std::vector<std::vector<int64_t>> outerValues;
      for (velox::vector_size_t i{0}; i < outer->sizeAt(row); ++i) {
        const auto innerRow = outer->offsetAt(row) + i;
        std::vector<int64_t> innerValues;
        for (velox::vector_size_t j{0}; j < inner->sizeAt(innerRow); ++j) {
          innerValues.push_back(
              elements->valueAt(inner->offsetAt(innerRow) + j));
        }
        outerValues.emplace_back(std::move(innerValues));
      }
      result.emplace_back(std::move(outerValues));
    }
    return result;
  }

  // Returns each timestamp as a seconds and nanoseconds pair, which prints
  // legibly when a matcher fails.
  std::vector<std::optional<std::pair<int64_t, uint64_t>>> readTimestamps(
      const velox::VectorPtr& output,
      velox::vector_size_t offset,
      velox::vector_size_t count) {
    const auto* timestamps = output->asFlatVector<velox::Timestamp>();
    NIMBLE_CHECK_NOT_NULL(timestamps);
    std::vector<std::optional<std::pair<int64_t, uint64_t>>> result;
    result.reserve(count);
    for (velox::vector_size_t row{offset}; row < offset + count; ++row) {
      if (timestamps->isNullAt(row)) {
        result.emplace_back(std::nullopt);
        continue;
      }
      const auto value = timestamps->valueAt(row);
      result.emplace_back(std::pair{value.getSeconds(), value.getNanos()});
    }
    return result;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(StreamReaderTest, decodesSparseRowsFromSingleEncodingChunk) {
  const std::vector<int64_t> values{
      10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25};
  const auto stream = encodeChunk<int64_t>(values, EncodingType::Trivial);

  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{
      type, pool_.get(), Encoding::Options{.bufferPool = &bufferPool}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 7, pool_.get());
  const std::array<RowRange, 4> ranges{{{1, 3}, {3, 5}, {5, 7}, {10, 11}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* flat = output->asFlatVector<int64_t>();
  ASSERT_NE(flat, nullptr);
  std::vector<int64_t> actual;
  actual.reserve(output->size());
  for (velox::vector_size_t row{0}; row < output->size(); ++row) {
    actual.push_back(flat->valueAt(row));
  }
  EXPECT_EQ(actual, (std::vector<int64_t>{11, 12, 13, 14, 15, 20, 24}));
}

TEST_F(StreamReaderTest, decodesRowsWithCoroutineApi) {
  const auto stream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12}, EncodingType::Trivial);
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"value"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}),
      });
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 2> streams{std::string_view{}, stream};
  auto output = velox::BaseVector::create(
      velox::ROW("value", velox::BIGINT()), 2, pool_.get());
  const std::array<RowRange, 1> ranges{{{1, 3}}};

  folly::coro::blockingWait(
      reader.co_read(streams, ranges, /*outputOffset=*/0, output));

  const auto* row = output->as<velox::RowVector>();
  ASSERT_NE(row, nullptr);
  const auto* flat = row->childAt(0)->asFlatVector<int64_t>();
  ASSERT_NE(flat, nullptr);
  EXPECT_EQ(
      (std::vector<int64_t>{flat->valueAt(0), flat->valueAt(1)}),
      (std::vector<int64_t>{11, 12}));
}

TEST_F(StreamReaderTest, rejectsStreamsWithMultipleEncodingChunks) {
  const auto stream =
      encodeChunk<int64_t>(
          std::vector<int64_t>{10, 11, 12}, EncodingType::Trivial) +
      encodeChunk<int64_t>(
          std::vector<int64_t>{20, 21, 22}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 1, pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 1}}};

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, ranges, /*outputOffset=*/0, output),
      "StreamReader requires exactly one encoded chunk per stream");
}

TEST_F(StreamReaderTest, decodesCompressedStreamChunk) {
  const std::string value(500, 'x');
  const std::vector<std::string_view> values(256, value);
  const auto stream = encodeChunk<std::string_view>(
      values,
      EncodingType::Trivial,
      CompressionParams{
          .type = CompressionType::Zstd,
          .acceptRatio = 1.0F,
      });
  const char* position = stream.data();
  EXPECT_EQ(readChunkHeader(position).compressionType, CompressionType::Zstd);

  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::String});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::VARCHAR(), 2, pool_.get());
  const std::array<RowRange, 2> ranges{{{7, 8}, {200, 201}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* flat = output->asFlatVector<velox::StringView>();
  ASSERT_NE(flat, nullptr);
  const std::vector<std::string> actual{
      flat->valueAt(0).str(), flat->valueAt(1).str()};
  EXPECT_EQ(actual, (std::vector<std::string>{value, value}));
}

TEST_F(StreamReaderTest, rejectsOverlappingSourceRanges) {
  const auto stream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12, 13, 20, 21, 22, 23},
      EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 7, pool_.get());
  const std::array<RowRange, 2> ranges{{{1, 6}, {3, 5}}};

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, ranges, /*outputOffset=*/0, output),
      "Read ranges must be ordered and disjoint");
}

TEST_F(StreamReaderTest, rejectsUnorderedSourceRanges) {
  const auto stream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12, 13}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 2, pool_.get());
  const std::array<RowRange, 2> ranges{{{2, 3}, {0, 1}}};

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, ranges, /*outputOffset=*/0, output),
      "Read ranges must be ordered and disjoint");
}

TEST_F(StreamReaderTest, rejectsOutputOverflow) {
  const auto stream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 2, pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 2}}};

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, ranges, /*outputOffset=*/1, output),
      "Read rows exceed the output vector");
}

TEST_F(StreamReaderTest, rejectsNegativeOutputOffset) {
  const auto stream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 1, pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 1}}};

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, ranges, /*outputOffset=*/-1, output),
      "Output offset must be non-negative");
}

TEST_F(StreamReaderTest, readsSelectedBooleans) {
  const std::array<bool, 6> values{false, true, false, true, true, false};
  const auto stream = encodeChunk<bool>(values, EncodingType::Trivial);
  const auto type =
      std::make_shared<const ScalarType>(StreamDescriptor{0, ScalarKind::Bool});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BOOLEAN(), 4, pool_.get());
  const std::array<RowRange, 2> ranges{{{1, 3}, {4, 6}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* flat = output->asFlatVector<bool>();
  ASSERT_NE(flat, nullptr);
  const std::vector<bool> actual{
      flat->valueAt(0),
      flat->valueAt(1),
      flat->valueAt(2),
      flat->valueAt(3),
  };
  EXPECT_EQ(actual, (std::vector<bool>{true, false, true, false}));
}

TEST_F(StreamReaderTest, rejectsEmptyRanges) {
  const auto stream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 1, pool_.get());

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, {}, /*outputOffset=*/0, output),
      "Read ranges must not be empty");
}

TEST_F(StreamReaderTest, rejectsEmptySourceRange) {
  const auto stream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 1, pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 0}}};

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, ranges, /*outputOffset=*/0, output),
      "Read range must not be empty");
}

TEST_F(StreamReaderTest, rejectsEncodingWithoutView) {
  const std::vector<int64_t> values{10, 11, 12};
  const auto stream = encodeChunk<int64_t>(values, EncodingType::Varint);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 1, pool_.get());
  const std::array<RowRange, 1> ranges{{{1, 2}}};

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, ranges, /*outputOffset=*/0, output),
      "Varint does not support EncodingView");
}

TEST_F(StreamReaderTest, readsNewStreamsAcrossCalls) {
  const auto firstStream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12}, EncodingType::Trivial);
  const auto secondStream = encodeChunk<int64_t>(
      std::vector<int64_t>{20, 21, 22}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  auto output = velox::BaseVector::create(velox::BIGINT(), 2, pool_.get());
  const std::array<RowRange, 2> ranges{{{0, 1}, {2, 3}}};

  const auto outputValues = [&]() {
    const auto* flat = output->asFlatVector<int64_t>();
    return std::vector<int64_t>{flat->valueAt(0), flat->valueAt(1)};
  };

  const std::array<std::string_view, 1> firstStreams{firstStream};
  runRead(reader, firstStreams, ranges, /*outputOffset=*/0, output);
  EXPECT_EQ(outputValues(), (std::vector<int64_t>{10, 12}));

  const std::array<std::string_view, 1> secondStreams{secondStream};
  runRead(reader, secondStreams, ranges, /*outputOffset=*/0, output);
  EXPECT_EQ(outputValues(), (std::vector<int64_t>{20, 22}));
}

TEST_F(StreamReaderTest, retainsScatteredStringsAcrossStreamBindings) {
  const auto firstStream = encodeChunk<std::string_view>(
      std::vector<std::string_view>{
          "first long string", "unused long string", "third long string"},
      EncodingType::Trivial);
  const auto secondStream = encodeChunk<std::string_view>(
      std::vector<std::string_view>{"second long string", "fourth long string"},
      EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::String});
  StreamReader reader{type, pool_.get(), {}};
  auto output = velox::BaseVector::create(velox::VARCHAR(), 4, pool_.get());

  const std::array<std::string_view, 1> firstStreams{firstStream};
  const std::array<RowRange, 1> firstRanges{{{0, 2}}};
  runRead(reader, firstStreams, firstRanges, /*outputOffset=*/0, output);

  const std::array<std::string_view, 1> secondStreams{secondStream};
  const std::array<RowRange, 1> secondRanges{{{0, 2}}};
  runRead(reader, secondStreams, secondRanges, /*outputOffset=*/2, output);

  const auto* flat = output->asFlatVector<velox::StringView>();
  ASSERT_NE(flat, nullptr);
  std::vector<std::string> actual;
  actual.reserve(output->size());
  for (velox::vector_size_t row{0}; row < output->size(); ++row) {
    actual.push_back(flat->valueAt(row).str());
  }
  EXPECT_EQ(
      actual,
      (std::vector<std::string>{
          "first long string",
          "unused long string",
          "second long string",
          "fourth long string"}));
}

TEST_F(StreamReaderTest, readsSelectedFlatRowThroughFieldReader) {
  const auto firstStream = encodeChunk<int64_t>(
      std::vector<int64_t>{10, 11, 12, 13}, EncodingType::Trivial);
  const auto secondStream = encodeChunk<int32_t>(
      std::vector<int32_t>{20, 21, 22, 23}, EncodingType::Trivial);
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"first", "second"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}),
          std::make_shared<const ScalarType>(
              StreamDescriptor{2, ScalarKind::Int32}),
      });
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 3> streams{
      std::string_view{}, firstStream, secondStream};
  auto output = velox::BaseVector::create(
      velox::ROW({{"first", velox::BIGINT()}, {"second", velox::INTEGER()}}),
      2,
      pool_.get());
  const std::array<RowRange, 2> ranges{{{1, 2}, {3, 4}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* row = output->as<velox::RowVector>();
  ASSERT_NE(row, nullptr);
  const auto* first = row->childAt(0)->asFlatVector<int64_t>();
  const auto* second = row->childAt(1)->asFlatVector<int32_t>();
  const std::vector<int64_t> actualFirst{first->valueAt(0), first->valueAt(1)};
  const std::vector<int32_t> actualSecond{
      second->valueAt(0), second->valueAt(1)};
  EXPECT_EQ(actualFirst, (std::vector<int64_t>{11, 13}));
  EXPECT_EQ(actualSecond, (std::vector<int32_t>{21, 23}));
}

TEST_F(StreamReaderTest, resizesShortRowChildren) {
  const auto stream =
      encodeChunk<int64_t>(std::vector<int64_t>{10, 11}, EncodingType::Trivial);
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"value"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}),
      });
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 2> streams{std::string_view{}, stream};
  auto output = velox::BaseVector::create(
      velox::ROW("value", velox::BIGINT()), 2, pool_.get());
  auto* row = output->as<velox::RowVector>();
  ASSERT_NE(row, nullptr);
  row->childAt(0)->resize(1);
  const std::array<RowRange, 1> ranges{{{0, 2}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  EXPECT_EQ(row->childAt(0)->size(), 2);
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<int64_t>()->rawValues()[1], int64_t{11});
}

TEST_F(StreamReaderTest, readsNullableRows) {
  const auto presence = encodeChunk<bool>(
      std::array<bool, 6>{true, false, true, true, true, true},
      EncodingType::Trivial);
  const auto values = encodeChunk<int64_t>(
      std::array<int64_t, 5>{10, 12, 13, 14, 15}, EncodingType::Trivial);
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"value"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}),
      });
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 2> streams{presence, values};
  auto output = velox::BaseVector::create(
      velox::ROW("value", velox::BIGINT()), 5, pool_.get());
  const std::array<RowRange, 2> ranges{{{1, 4}, {5, 6}}};

  runRead(reader, streams, ranges, /*outputOffset=*/1, output);

  const auto* row = output->as<velox::RowVector>();
  ASSERT_NE(row, nullptr);
  const auto* child = row->childAt(0)->asFlatVector<int64_t>();
  ASSERT_NE(child, nullptr);
  std::vector<std::optional<int64_t>> actual;
  for (velox::vector_size_t i{1}; i < output->size(); ++i) {
    actual.push_back(
        row->isNullAt(i) ? std::nullopt
                         : std::optional<int64_t>{child->valueAt(i)});
  }
  EXPECT_EQ(
      actual, (std::vector<std::optional<int64_t>>{std::nullopt, 12, 13, 15}));
}

TEST_F(StreamReaderTest, readsAllNullRows) {
  const auto presence = encodeChunk<bool>(
      std::array<bool, 2>{false, false}, EncodingType::Trivial);
  const auto values =
      encodeChunk<int64_t>(std::array<int64_t, 0>{}, EncodingType::Trivial);
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"value"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}),
      });
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 2> streams{presence, values};
  auto output = velox::BaseVector::create(
      velox::ROW("value", velox::BIGINT()), 2, pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 2}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* row = output->as<velox::RowVector>();
  ASSERT_NE(row, nullptr);
  EXPECT_TRUE(row->isNullAt(0));
  EXPECT_TRUE(row->isNullAt(1));
}

TEST_F(StreamReaderTest, readsNestedNullableRows) {
  const auto innerPresence = encodeChunk<bool>(
      std::array<bool, 4>{true, false, true, true}, EncodingType::Trivial);
  const auto values = encodeChunk<int64_t>(
      std::array<int64_t, 3>{10, 12, 13}, EncodingType::Trivial);
  const auto innerType = std::make_shared<const RowType>(
      StreamDescriptor{1, ScalarKind::Bool},
      std::vector<std::string>{"value"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{2, ScalarKind::Int64}),
      });
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"nested"},
      std::vector<std::shared_ptr<const Type>>{innerType});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 3> streams{
      std::string_view{}, innerPresence, values};
  auto output = velox::BaseVector::create(
      velox::ROW("nested", velox::ROW("value", velox::BIGINT())),
      4,
      pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 4}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* outer = output->as<velox::RowVector>();
  ASSERT_NE(outer, nullptr);
  const auto* inner = outer->childAt(0)->as<velox::RowVector>();
  ASSERT_NE(inner, nullptr);
  const auto* child = inner->childAt(0)->asFlatVector<int64_t>();
  ASSERT_NE(child, nullptr);
  std::vector<std::optional<int64_t>> actual;
  for (velox::vector_size_t i{0}; i < output->size(); ++i) {
    actual.push_back(
        inner->isNullAt(i) ? std::nullopt
                           : std::optional<int64_t>{child->valueAt(i)});
  }
  EXPECT_EQ(
      actual, (std::vector<std::optional<int64_t>>{10, std::nullopt, 12, 13}));
}

TEST_F(StreamReaderTest, readsRandomNullableRowsAndProjections) {
  constexpr uint32_t kSeed{8'675'309};
  std::mt19937 rng{kSeed};
  for (uint32_t iteration{0}; iteration < 100; ++iteration) {
    SCOPED_TRACE(
        ::testing::Message()
        << "seed=" << kSeed << ", iteration=" << iteration);
    const uint32_t rowCount{1 + static_cast<uint32_t>(rng() % 128)};
    const uint32_t nullModulo{1 + static_cast<uint32_t>(rng() % 8)};
    const uint32_t projectionMask{1 + static_cast<uint32_t>(rng() % 3)};
    Vector<bool> isNonNull{pool_.get(), rowCount};
    std::vector<int64_t> firstValues;
    std::vector<int32_t> secondValues;
    for (uint32_t row{0}; row < rowCount; ++row) {
      isNonNull[row] = rng() % nullModulo != 0;
      if (isNonNull[row]) {
        firstValues.push_back(static_cast<int64_t>(row) * 10 + 1);
        secondValues.push_back(static_cast<int32_t>(row) * 10 + 2);
      }
    }

    std::vector<std::string> names;
    std::vector<std::shared_ptr<const Type>> childTypes;
    std::vector<velox::TypePtr> outputTypes;
    std::vector<std::string> encodedStreams;
    encodedStreams.push_back(
        encodeChunk<bool>(
            std::span<const bool>{isNonNull.data(), isNonNull.size()},
            EncodingType::Trivial));
    if (projectionMask & 1) {
      names.emplace_back("first");
      childTypes.push_back(
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}));
      outputTypes.push_back(velox::BIGINT());
      encodedStreams.push_back(
          encodeChunk<int64_t>(firstValues, EncodingType::Trivial));
    }
    if (projectionMask & 2) {
      names.emplace_back("second");
      childTypes.push_back(
          std::make_shared<const ScalarType>(
              StreamDescriptor{2, ScalarKind::Int32}));
      outputTypes.push_back(velox::INTEGER());
      encodedStreams.push_back(
          encodeChunk<int32_t>(secondValues, EncodingType::Trivial));
    }
    const auto type = std::make_shared<const RowType>(
        StreamDescriptor{0, ScalarKind::Bool}, names, childTypes);
    StreamReader reader{type, pool_.get(), {}};
    std::vector<std::string_view> streams;
    streams.reserve(encodedStreams.size());
    for (const auto& stream : encodedStreams) {
      streams.push_back(stream);
    }
    const auto ranges = makeRandomRanges(rowCount, rng);
    velox::vector_size_t numSelectedRows{0};
    for (const auto& range : ranges) {
      numSelectedRows += static_cast<velox::vector_size_t>(range.numRows());
    }
    const auto outputOffset = static_cast<velox::vector_size_t>(rng() % 4);
    auto output = velox::BaseVector::create(
        velox::ROW(names, outputTypes),
        outputOffset + numSelectedRows,
        pool_.get());

    runRead(reader, streams, ranges, outputOffset, output);

    const auto* row = output->as<velox::RowVector>();
    ASSERT_NE(row, nullptr);
    std::vector<std::optional<int64_t>> actualFirst;
    std::vector<std::optional<int32_t>> actualSecond;
    std::vector<std::optional<int64_t>> expectedFirst;
    std::vector<std::optional<int32_t>> expectedSecond;
    velox::vector_size_t outputRow{outputOffset};
    for (const auto& range : ranges) {
      for (uint32_t sourceRow{range.startRow}; sourceRow < range.endRow;
           ++sourceRow, ++outputRow) {
        if (projectionMask & 1) {
          expectedFirst.push_back(
              isNonNull[sourceRow]
                  ? std::optional<
                        int64_t>{static_cast<int64_t>(sourceRow) * 10 + 1}
                  : std::nullopt);
          const auto* child = row->childAt(0)->asFlatVector<int64_t>();
          actualFirst.push_back(
              row->isNullAt(outputRow)
                  ? std::nullopt
                  : std::optional<int64_t>{child->valueAt(outputRow)});
        }
        if (projectionMask & 2) {
          expectedSecond.push_back(
              isNonNull[sourceRow]
                  ? std::optional<
                        int32_t>{static_cast<int32_t>(sourceRow) * 10 + 2}
                  : std::nullopt);
          const auto childIndex = projectionMask & 1 ? 1 : 0;
          const auto* child = row->childAt(childIndex)->asFlatVector<int32_t>();
          actualSecond.push_back(
              row->isNullAt(outputRow)
                  ? std::nullopt
                  : std::optional<int32_t>{child->valueAt(outputRow)});
        }
      }
    }
    EXPECT_EQ(actualFirst, expectedFirst);
    EXPECT_EQ(actualSecond, expectedSecond);
  }
}

TEST_F(StreamReaderTest, readsSelectedTimestamps) {
  const auto micros = encodeNullableChunk<int64_t>(
      {1'000'001, std::nullopt, -1, 2'000'003}, EncodingType::Trivial);
  const auto nanos = encodeChunk<uint16_t>(
      std::array<uint16_t, 3>{2, 999, 4}, EncodingType::Trivial);
  const auto type = std::make_shared<const TimestampMicroNanoType>(
      StreamDescriptor{0, ScalarKind::Int64},
      StreamDescriptor{1, ScalarKind::UInt16});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 2> streams{micros, nanos};
  auto output = velox::BaseVector::create(velox::TIMESTAMP(), 3, pool_.get());
  const std::array<RowRange, 2> ranges{{{0, 2}, {3, 4}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* timestamps = output->asFlatVector<velox::Timestamp>();
  ASSERT_NE(timestamps, nullptr);
  EXPECT_FALSE(timestamps->isNullAt(0));
  EXPECT_TRUE(timestamps->isNullAt(1));
  EXPECT_FALSE(timestamps->isNullAt(2));
  EXPECT_EQ(timestamps->valueAt(0), velox::Timestamp(1, 1'002));
  EXPECT_EQ(timestamps->valueAt(2), velox::Timestamp(2, 3'004));
}

TEST_F(StreamReaderTest, readsSelectedArrays) {
  const auto lengths = encodeNullableChunk<uint32_t>(
      {2, std::nullopt, 0, 1, 2}, EncodingType::Trivial);
  const auto elements = encodeChunk<int64_t>(
      std::array<int64_t, 5>{10, 11, 13, 14, 15}, EncodingType::Trivial);
  const auto type = std::make_shared<const ArrayType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{1, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 2> streams{lengths, elements};
  auto output =
      velox::BaseVector::create(velox::ARRAY(velox::BIGINT()), 4, pool_.get());
  const std::array<RowRange, 2> ranges{{{1, 2}, {3, 5}}};

  runRead(reader, streams, ranges, /*outputOffset=*/1, output);

  using NullableArray = std::optional<std::vector<int64_t>>;
  EXPECT_THAT(
      readArrays(output, /*offset=*/1, /*count=*/3),
      ::testing::ElementsAre(
          NullableArray{std::nullopt},
          NullableArray{{13}},
          NullableArray{{14, 15}}));
}

TEST_F(StreamReaderTest, readsRandomArrays) {
  constexpr uint32_t kSeed{2'236'067};
  std::mt19937 rng{kSeed};
  for (uint32_t iteration{0}; iteration < 100; ++iteration) {
    SCOPED_TRACE(
        ::testing::Message()
        << "seed=" << kSeed << ", iteration=" << iteration);
    const uint32_t numRows{1 + static_cast<uint32_t>(rng() % 128)};
    std::vector<std::optional<std::vector<int64_t>>> arrays;
    std::vector<std::optional<uint32_t>> lengths;
    std::vector<int64_t> elements;
    arrays.reserve(numRows);
    lengths.reserve(numRows);
    for (uint32_t row{0}; row < numRows; ++row) {
      if (rng() % 5 == 0) {
        arrays.emplace_back(std::nullopt);
        lengths.emplace_back(std::nullopt);
        continue;
      }
      const uint32_t numElements{static_cast<uint32_t>(rng() % 6)};
      std::vector<int64_t> rowValues;
      rowValues.reserve(numElements);
      for (uint32_t i{0}; i < numElements; ++i) {
        const int64_t value{static_cast<int64_t>(row) * 10 + i};
        rowValues.push_back(value);
        elements.push_back(value);
      }
      arrays.emplace_back(std::move(rowValues));
      lengths.emplace_back(numElements);
    }

    const auto lengthsStream =
        encodeNullableChunk<uint32_t>(lengths, EncodingType::Trivial);
    const auto elementsStream =
        encodeChunk<int64_t>(elements, EncodingType::Trivial);
    const auto type = std::make_shared<const ArrayType>(
        StreamDescriptor{0, ScalarKind::UInt32},
        std::make_shared<const ScalarType>(
            StreamDescriptor{1, ScalarKind::Int64}));
    StreamReader reader{type, pool_.get(), {}};
    const std::array<std::string_view, 2> streams{
        lengthsStream, elementsStream};
    const auto ranges = makeRandomRanges(numRows, rng);
    velox::vector_size_t numSelectedRows{0};
    std::vector<std::optional<std::vector<int64_t>>> expected;
    for (const auto& range : ranges) {
      numSelectedRows += static_cast<velox::vector_size_t>(range.numRows());
      expected.insert(
          expected.end(),
          arrays.begin() + range.startRow,
          arrays.begin() + range.endRow);
    }
    const auto outputOffset = static_cast<velox::vector_size_t>(rng() % 4);
    auto output = velox::BaseVector::create(
        velox::ARRAY(velox::BIGINT()),
        outputOffset + numSelectedRows,
        pool_.get());

    runRead(reader, streams, ranges, outputOffset, output);

    EXPECT_THAT(
        readArrays(output, outputOffset, numSelectedRows),
        ::testing::ElementsAreArray(expected));
  }
}

TEST_F(StreamReaderTest, readsSelectedMaps) {
  const auto lengths = encodeNullableChunk<uint32_t>(
      {1, std::nullopt, 0, 2}, EncodingType::Trivial);
  const auto keys = encodeChunk<std::string_view>(
      std::array<std::string_view, 3>{"a", "b", "c"}, EncodingType::Trivial);
  const auto values = encodeChunk<int64_t>(
      std::array<int64_t, 3>{10, 20, 30}, EncodingType::Trivial);
  const auto type = std::make_shared<const MapType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{1, ScalarKind::String}),
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 3> streams{lengths, keys, values};
  auto output = velox::BaseVector::create(
      velox::MAP(velox::VARCHAR(), velox::BIGINT()), 3, pool_.get());
  const std::array<RowRange, 2> ranges{{{0, 2}, {3, 4}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  using MapEntries = std::vector<std::pair<std::string, int64_t>>;
  using NullableMap = std::optional<MapEntries>;
  EXPECT_THAT(
      readMaps(output, /*offset=*/0, /*count=*/3),
      ::testing::ElementsAre(
          NullableMap{MapEntries{{"a", 10}}},
          NullableMap{std::nullopt},
          NullableMap{MapEntries{{"b", 20}, {"c", 30}}}));
}

TEST_F(StreamReaderTest, readsSelectedDeduplicatedArrays) {
  const auto offsets = encodeNullableChunk<uint32_t>(
      {0, 0, std::nullopt, 1, 2}, EncodingType::Trivial);
  const auto lengths = encodeChunk<uint32_t>(
      std::array<uint32_t, 3>{2, 1, 2}, EncodingType::Trivial);
  const auto elements = encodeChunk<int64_t>(
      std::array<int64_t, 5>{10, 11, 13, 14, 15}, EncodingType::Trivial);
  const auto type = std::make_shared<const ArrayWithOffsetsType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 3> streams{lengths, offsets, elements};
  auto output =
      velox::BaseVector::create(velox::ARRAY(velox::BIGINT()), 2, pool_.get());
  const std::array<RowRange, 2> ranges{{{1, 2}, {4, 5}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  using NullableArray = std::optional<std::vector<int64_t>>;
  EXPECT_THAT(
      readArrays(output, /*offset=*/0, /*count=*/2),
      ::testing::ElementsAre(NullableArray{{10, 11}}, NullableArray{{14, 15}}));
}

TEST_F(StreamReaderTest, readsSelectedSlidingWindowMaps) {
  const auto offsets = encodeNullableChunk<uint32_t>(
      {0, 0, std::nullopt, 2, 2}, EncodingType::Trivial);
  const auto lengths = encodeChunk<uint32_t>(
      std::array<uint32_t, 4>{2, 2, 1, 1}, EncodingType::Trivial);
  const auto keys = encodeChunk<std::string_view>(
      std::array<std::string_view, 3>{"a", "b", "c"}, EncodingType::Trivial);
  const auto values = encodeChunk<int64_t>(
      std::array<int64_t, 3>{10, 20, 30}, EncodingType::Trivial);
  const auto type = std::make_shared<const SlidingWindowMapType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::String}),
      std::make_shared<const ScalarType>(
          StreamDescriptor{3, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 4> streams{offsets, lengths, keys, values};
  auto output = velox::BaseVector::create(
      velox::MAP(velox::VARCHAR(), velox::BIGINT()), 3, pool_.get());
  const std::array<RowRange, 2> ranges{{{1, 3}, {4, 5}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  using MapEntries = std::vector<std::pair<std::string, int64_t>>;
  using NullableMap = std::optional<MapEntries>;
  EXPECT_THAT(
      readMaps(output, /*offset=*/0, /*count=*/3),
      ::testing::ElementsAre(
          NullableMap{MapEntries{{"a", 10}, {"b", 20}}},
          NullableMap{std::nullopt},
          NullableMap{MapEntries{{"c", 30}}}));
}

// Selecting two rows of one deduplicated run has to place the elements once
// and point both rows at them.
TEST_F(StreamReaderTest, readsSelectedDeduplicatedArraysSharingOneRun) {
  const auto offsets = encodeNullableChunk<uint32_t>(
      {0, 0, std::nullopt, 1, 2}, EncodingType::Trivial);
  const auto lengths = encodeChunk<uint32_t>(
      std::array<uint32_t, 3>{2, 1, 2}, EncodingType::Trivial);
  const auto elements = encodeChunk<int64_t>(
      std::array<int64_t, 5>{10, 11, 13, 14, 15}, EncodingType::Trivial);
  const auto type = std::make_shared<const ArrayWithOffsetsType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 3> streams{lengths, offsets, elements};
  auto output =
      velox::BaseVector::create(velox::ARRAY(velox::BIGINT()), 3, pool_.get());
  // Rows 0 and 1 share a run; row 3 starts a later one.
  const std::array<RowRange, 2> ranges{{{0, 2}, {3, 4}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  using NullableArray = std::optional<std::vector<int64_t>>;
  EXPECT_THAT(
      readArrays(output, /*offset=*/0, /*count=*/3),
      ::testing::ElementsAre(
          NullableArray{{10, 11}},
          NullableArray{{10, 11}},
          NullableArray{{13}}));
  const auto* arrays = output->as<velox::ArrayVector>();
  ASSERT_NE(arrays, nullptr);
  // The shared run is stored once rather than copied per row.
  EXPECT_EQ(arrays->offsetAt(0), arrays->offsetAt(1));
  EXPECT_EQ(arrays->elements()->size(), 3);
}

// A run may carry no elements, and rows sharing it stay non-null and empty.
TEST_F(StreamReaderTest, readsSelectedDeduplicatedArraysWithEmptyRun) {
  const auto offsets =
      encodeNullableChunk<uint32_t>({0, 1, 1}, EncodingType::Trivial);
  const auto lengths = encodeChunk<uint32_t>(
      std::array<uint32_t, 2>{2, 0}, EncodingType::Trivial);
  const auto elements = encodeChunk<int64_t>(
      std::array<int64_t, 2>{10, 11}, EncodingType::Trivial);
  const auto type = std::make_shared<const ArrayWithOffsetsType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 3> streams{lengths, offsets, elements};
  auto output =
      velox::BaseVector::create(velox::ARRAY(velox::BIGINT()), 3, pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 3}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  using NullableArray = std::optional<std::vector<int64_t>>;
  EXPECT_THAT(
      readArrays(output, /*offset=*/0, /*count=*/3),
      ::testing::ElementsAre(
          NullableArray{{10, 11}},
          NullableArray{std::vector<int64_t>{}},
          NullableArray{std::vector<int64_t>{}}));
}

// Every selected row being null leaves no run, so the lengths stream is never
// read and the rows still come back null.
TEST_F(StreamReaderTest, readsSelectedDeduplicatedArraysWithAllNullRows) {
  const auto offsets = encodeNullableChunk<uint32_t>(
      {std::nullopt, std::nullopt, std::nullopt}, EncodingType::Trivial);
  const auto lengths =
      encodeChunk<uint32_t>(std::array<uint32_t, 1>{2}, EncodingType::Trivial);
  const auto elements = encodeChunk<int64_t>(
      std::array<int64_t, 2>{10, 11}, EncodingType::Trivial);
  const auto type = std::make_shared<const ArrayWithOffsetsType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 3> streams{lengths, offsets, elements};
  auto output =
      velox::BaseVector::create(velox::ARRAY(velox::BIGINT()), 2, pool_.get());
  const std::array<RowRange, 1> ranges{{{1, 3}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  using NullableArray = std::optional<std::vector<int64_t>>;
  EXPECT_THAT(
      readArrays(output, /*offset=*/0, /*count=*/2),
      ::testing::ElementsAre(
          NullableArray{std::nullopt}, NullableArray{std::nullopt}));
}

// Deduplicated rows are written in place, so a second read has to land past
// the rows and elements the first one placed.
TEST_F(StreamReaderTest, readsSelectedDeduplicatedArraysAtOutputOffset) {
  const auto offsets = encodeNullableChunk<uint32_t>(
      {0, 0, std::nullopt, 1, 2}, EncodingType::Trivial);
  const auto lengths = encodeChunk<uint32_t>(
      std::array<uint32_t, 3>{2, 1, 2}, EncodingType::Trivial);
  const auto elements = encodeChunk<int64_t>(
      std::array<int64_t, 5>{10, 11, 13, 14, 15}, EncodingType::Trivial);
  const auto type = std::make_shared<const ArrayWithOffsetsType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 3> streams{lengths, offsets, elements};
  auto output =
      velox::BaseVector::create(velox::ARRAY(velox::BIGINT()), 4, pool_.get());

  const std::array<RowRange, 1> firstRanges{{{0, 2}}};
  runRead(reader, streams, firstRanges, /*outputOffset=*/0, output);
  const std::array<RowRange, 2> secondRanges{{{2, 3}, {4, 5}}};
  runRead(reader, streams, secondRanges, /*outputOffset=*/2, output);

  using NullableArray = std::optional<std::vector<int64_t>>;
  EXPECT_THAT(
      readArrays(output, /*offset=*/0, /*count=*/4),
      ::testing::ElementsAre(
          NullableArray{{10, 11}},
          NullableArray{{10, 11}},
          NullableArray{std::nullopt},
          NullableArray{{14, 15}}));
}

// Skipping a whole window leaves a gap between the element ranges handed to
// the key and value readers, which only a selection that drops a middle row
// produces.
TEST_F(StreamReaderTest, readsSelectedSlidingWindowMapsSkippingAWindow) {
  const auto offsets =
      encodeNullableChunk<uint32_t>({0, 2, 4, 6}, EncodingType::Trivial);
  const auto lengths = encodeChunk<uint32_t>(
      std::array<uint32_t, 4>{2, 2, 2, 2}, EncodingType::Trivial);
  const auto keys = encodeChunk<std::string_view>(
      std::array<std::string_view, 8>{"a", "b", "c", "d", "e", "f", "g", "h"},
      EncodingType::Trivial);
  const auto values = encodeChunk<int64_t>(
      std::array<int64_t, 8>{10, 20, 30, 40, 50, 60, 70, 80},
      EncodingType::Trivial);
  const auto type = std::make_shared<const SlidingWindowMapType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::String}),
      std::make_shared<const ScalarType>(
          StreamDescriptor{3, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 4> streams{offsets, lengths, keys, values};
  auto output = velox::BaseVector::create(
      velox::MAP(velox::VARCHAR(), velox::BIGINT()), 2, pool_.get());
  // Rows 1 and 2 are skipped, so elements 2 through 5 are never read.
  const std::array<RowRange, 2> ranges{{{0, 1}, {3, 4}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  using MapEntries = std::vector<std::pair<std::string, int64_t>>;
  using NullableMap = std::optional<MapEntries>;
  EXPECT_THAT(
      readMaps(output, /*offset=*/0, /*count=*/2),
      ::testing::ElementsAre(
          NullableMap{MapEntries{{"a", 10}, {"b", 20}}},
          NullableMap{MapEntries{{"g", 70}, {"h", 80}}}));
}

// Sliding-window map rows are written straight into the caller's vector, so a
// second read has to land past the rows and entries the first one placed.
TEST_F(StreamReaderTest, readsSelectedSlidingWindowMapsAtOutputOffset) {
  const auto offsets = encodeNullableChunk<uint32_t>(
      {0, 0, std::nullopt, 2, 2}, EncodingType::Trivial);
  const auto lengths = encodeChunk<uint32_t>(
      std::array<uint32_t, 4>{2, 2, 1, 1}, EncodingType::Trivial);
  const auto keys = encodeChunk<std::string_view>(
      std::array<std::string_view, 3>{"a", "b", "c"}, EncodingType::Trivial);
  const auto values = encodeChunk<int64_t>(
      std::array<int64_t, 3>{10, 20, 30}, EncodingType::Trivial);
  const auto type = std::make_shared<const SlidingWindowMapType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::String}),
      std::make_shared<const ScalarType>(
          StreamDescriptor{3, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 4> streams{offsets, lengths, keys, values};
  auto output = velox::BaseVector::create(
      velox::MAP(velox::VARCHAR(), velox::BIGINT()), 4, pool_.get());

  const std::array<RowRange, 1> firstRanges{{{0, 2}}};
  runRead(reader, streams, firstRanges, /*outputOffset=*/0, output);
  const std::array<RowRange, 2> secondRanges{{{2, 3}, {4, 5}}};
  runRead(reader, streams, secondRanges, /*outputOffset=*/2, output);

  using MapEntries = std::vector<std::pair<std::string, int64_t>>;
  using NullableMap = std::optional<MapEntries>;
  EXPECT_THAT(
      readMaps(output, /*offset=*/0, /*count=*/4),
      ::testing::ElementsAre(
          NullableMap{MapEntries{{"a", 10}, {"b", 20}}},
          NullableMap{MapEntries{{"a", 10}, {"b", 20}}},
          NullableMap{std::nullopt},
          NullableMap{MapEntries{{"c", 30}}}));
}

// A non-null empty map keeps an offset but consumes no entry, which is the
// shape that corrupted sliding-window map reads in S505299. Covers one inside
// a selected range and one inside a skipped gap.
TEST_F(StreamReaderTest, readsSelectedSlidingWindowMapsWithEmptyMaps) {
  const auto offsets = encodeNullableChunk<uint32_t>(
      {0, 2, 2, 3, std::nullopt, 3}, EncodingType::Trivial);
  const auto lengths = encodeChunk<uint32_t>(
      std::array<uint32_t, 5>{2, 0, 1, 0, 1}, EncodingType::Trivial);
  const auto keys = encodeChunk<std::string_view>(
      std::array<std::string_view, 4>{"a", "b", "c", "d"},
      EncodingType::Trivial);
  const auto values = encodeChunk<int64_t>(
      std::array<int64_t, 4>{10, 20, 30, 40}, EncodingType::Trivial);
  const auto type = std::make_shared<const SlidingWindowMapType>(
      StreamDescriptor{0, ScalarKind::UInt32},
      StreamDescriptor{1, ScalarKind::UInt32},
      std::make_shared<const ScalarType>(
          StreamDescriptor{2, ScalarKind::String}),
      std::make_shared<const ScalarType>(
          StreamDescriptor{3, ScalarKind::Int64}));
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 4> streams{offsets, lengths, keys, values};
  auto output = velox::BaseVector::create(
      velox::MAP(velox::VARCHAR(), velox::BIGINT()), 5, pool_.get());
  // Skips row 1, the empty map sharing an offset with the following row.
  const std::array<RowRange, 2> ranges{{{0, 1}, {2, 6}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  using MapEntries = std::vector<std::pair<std::string, int64_t>>;
  using NullableMap = std::optional<MapEntries>;
  EXPECT_THAT(
      readMaps(output, /*offset=*/0, /*count=*/5),
      ::testing::ElementsAre(
          NullableMap{MapEntries{{"a", 10}, {"b", 20}}},
          NullableMap{MapEntries{{"c", 30}}},
          NullableMap{MapEntries{}},
          NullableMap{std::nullopt},
          NullableMap{MapEntries{{"d", 40}}}));
}

TEST_F(StreamReaderTest, readsSelectedFlatMaps) {
  std::vector<std::unique_ptr<StreamDescriptor>> inMapDescriptors;
  inMapDescriptors.push_back(
      std::make_unique<StreamDescriptor>(2, ScalarKind::Bool));
  inMapDescriptors.push_back(
      std::make_unique<StreamDescriptor>(4, ScalarKind::Bool));
  std::vector<std::shared_ptr<const Type>> valueTypes;
  valueTypes.push_back(
      std::make_shared<const ScalarType>(
          StreamDescriptor{3, ScalarKind::Int64}));
  valueTypes.push_back(
      std::make_shared<const ScalarType>(
          StreamDescriptor{5, ScalarKind::Int64}));
  const auto flatMapType = std::make_shared<const FlatMapType>(
      StreamDescriptor{1, ScalarKind::Bool},
      ScalarKind::String,
      std::vector<std::string>{"a", "b"},
      std::move(inMapDescriptors),
      std::move(valueTypes));
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"attributes"},
      std::vector<std::shared_ptr<const Type>>{flatMapType});
  const auto mapPresence = encodeChunk<bool>(
      std::array<bool, 4>{true, false, true, true}, EncodingType::Trivial);
  const auto inMapA = encodeChunk<bool>(
      std::array<bool, 3>{true, true, false}, EncodingType::Trivial);
  const auto valuesA = encodeChunk<int64_t>(
      std::array<int64_t, 2>{10, 12}, EncodingType::Trivial);
  const auto inMapB = encodeChunk<bool>(
      std::array<bool, 3>{false, true, false}, EncodingType::Trivial);
  const auto valuesB =
      encodeChunk<int64_t>(std::array<int64_t, 1>{22}, EncodingType::Trivial);
  const std::array<std::string_view, 6> streams{
      std::string_view{}, mapPresence, inMapA, inMapB, valuesA, valuesB};
  StreamReader reader{type, pool_.get(), {}};
  auto output = velox::BaseVector::create(
      velox::ROW("attributes", velox::MAP(velox::VARCHAR(), velox::BIGINT())),
      3,
      pool_.get());
  const std::array<RowRange, 2> ranges{{{0, 2}, {3, 4}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* row = output->as<velox::RowVector>();
  ASSERT_NE(row, nullptr);
  using MapEntries = std::vector<std::pair<std::string, int64_t>>;
  using NullableMap = std::optional<MapEntries>;
  EXPECT_THAT(
      readMaps(row->childAt(0), /*offset=*/0, /*count=*/3),
      ::testing::ElementsAre(
          NullableMap{MapEntries{{"a", 10}}},
          NullableMap{std::nullopt},
          NullableMap{MapEntries{}}));
}

TEST_F(StreamReaderTest, readsRandomNestedSchemaNodes) {
  using MapEntries = std::vector<std::pair<std::string, int64_t>>;
  constexpr uint32_t kSeed{3'141'592};
  std::mt19937 rng{kSeed};
  for (uint32_t iteration{0}; iteration < 50; ++iteration) {
    SCOPED_TRACE(
        ::testing::Message()
        << "seed=" << kSeed << ", iteration=" << iteration);
    const uint32_t numRows{1 + static_cast<uint32_t>(rng() % 64)};

    std::vector<std::optional<int64_t>> scalars;
    std::vector<std::optional<uint32_t>> arrayLengths;
    std::vector<int64_t> arrayElements;
    std::vector<std::optional<uint32_t>> mapLengths;
    std::vector<std::string> mapKeyValues;
    std::vector<int64_t> mapEntryValues;
    std::vector<std::optional<uint32_t>> outerLengths;
    std::vector<uint32_t> innerLengths;
    std::vector<int64_t> innerElements;
    std::vector<std::optional<int64_t>> micros;
    std::vector<uint16_t> nanos;
    std::vector<std::optional<std::vector<int64_t>>> expectedArrays;
    std::vector<std::optional<MapEntries>> expectedMaps;
    std::vector<std::optional<std::vector<std::vector<int64_t>>>>
        expectedNestedArrays;
    std::vector<std::optional<std::pair<int64_t, uint64_t>>> expectedTimestamps;

    for (uint32_t row{0}; row < numRows; ++row) {
      // Keep every child stream of row 0 populated so that no encoded stream
      // is empty, which the chunk encoders do not accept.
      const bool forcePresent = row == 0;

      if (!forcePresent && rng() % 5 == 0) {
        scalars.emplace_back(std::nullopt);
      } else {
        scalars.emplace_back(static_cast<int64_t>(rng() % 1'000));
      }

      if (!forcePresent && rng() % 5 == 0) {
        arrayLengths.emplace_back(std::nullopt);
        expectedArrays.emplace_back(std::nullopt);
      } else {
        const uint32_t numElements{
            forcePresent ? 2u : static_cast<uint32_t>(rng() % 5)};
        std::vector<int64_t> values;
        values.reserve(numElements);
        for (uint32_t i{0}; i < numElements; ++i) {
          const auto value = static_cast<int64_t>(row) * 100 + i;
          values.push_back(value);
          arrayElements.push_back(value);
        }
        arrayLengths.emplace_back(numElements);
        expectedArrays.emplace_back(std::move(values));
      }

      if (!forcePresent && rng() % 5 == 0) {
        mapLengths.emplace_back(std::nullopt);
        expectedMaps.emplace_back(std::nullopt);
      } else {
        const uint32_t numEntries{
            forcePresent ? 2u : static_cast<uint32_t>(rng() % 4)};
        MapEntries entries;
        entries.reserve(numEntries);
        for (uint32_t i{0}; i < numEntries; ++i) {
          auto key =
              std::string("k") + std::to_string(row) + "_" + std::to_string(i);
          const auto value = static_cast<int64_t>(row) * 10 + i;
          mapKeyValues.push_back(key);
          mapEntryValues.push_back(value);
          entries.emplace_back(std::move(key), value);
        }
        mapLengths.emplace_back(numEntries);
        expectedMaps.emplace_back(std::move(entries));
      }

      if (!forcePresent && rng() % 5 == 0) {
        outerLengths.emplace_back(std::nullopt);
        expectedNestedArrays.emplace_back(std::nullopt);
      } else {
        const uint32_t numInnerArrays{
            forcePresent ? 2u : static_cast<uint32_t>(rng() % 4)};
        std::vector<std::vector<int64_t>> outerValues;
        outerValues.reserve(numInnerArrays);
        for (uint32_t i{0}; i < numInnerArrays; ++i) {
          const uint32_t numElements{
              forcePresent ? 2u : static_cast<uint32_t>(rng() % 4)};
          std::vector<int64_t> innerValues;
          innerValues.reserve(numElements);
          for (uint32_t j{0}; j < numElements; ++j) {
            const auto value = static_cast<int64_t>(row) * 1'000 + i * 10 + j;
            innerValues.push_back(value);
            innerElements.push_back(value);
          }
          innerLengths.push_back(numElements);
          outerValues.emplace_back(std::move(innerValues));
        }
        outerLengths.emplace_back(numInnerArrays);
        expectedNestedArrays.emplace_back(std::move(outerValues));
      }

      if (!forcePresent && rng() % 5 == 0) {
        micros.emplace_back(std::nullopt);
        expectedTimestamps.emplace_back(std::nullopt);
      } else {
        // Span both signs so that the conversion's borrow-a-second path for
        // negative remainders is exercised.
        const auto microValue =
            static_cast<int64_t>(rng() % 10'000'000) - 5'000'000;
        const auto nanoValue = static_cast<uint16_t>(rng() % 1'000);
        micros.emplace_back(microValue);
        nanos.push_back(nanoValue);
        int64_t seconds = microValue / 1'000'000;
        int64_t remainder = microValue % 1'000'000;
        if (remainder < 0) {
          --seconds;
          remainder += 1'000'000;
        }
        expectedTimestamps.emplace_back(
            std::pair{
                seconds, static_cast<uint64_t>(remainder) * 1'000 + nanoValue});
      }
    }

    std::vector<std::string_view> mapKeyViews;
    mapKeyViews.reserve(mapKeyValues.size());
    for (const auto& key : mapKeyValues) {
      mapKeyViews.emplace_back(key);
    }

    const auto scalarStream =
        encodeNullableChunk<int64_t>(scalars, EncodingType::Trivial);
    const auto arrayLengthStream =
        encodeNullableChunk<uint32_t>(arrayLengths, EncodingType::Trivial);
    const auto arrayElementStream =
        encodeChunk<int64_t>(arrayElements, EncodingType::Trivial);
    const auto mapLengthStream =
        encodeNullableChunk<uint32_t>(mapLengths, EncodingType::Trivial);
    const auto mapKeyStream =
        encodeChunk<std::string_view>(mapKeyViews, EncodingType::Trivial);
    const auto mapValueStream =
        encodeChunk<int64_t>(mapEntryValues, EncodingType::Trivial);
    const auto outerLengthStream =
        encodeNullableChunk<uint32_t>(outerLengths, EncodingType::Trivial);
    const auto innerLengthStream =
        encodeChunk<uint32_t>(innerLengths, EncodingType::Trivial);
    const auto innerElementStream =
        encodeChunk<int64_t>(innerElements, EncodingType::Trivial);
    const auto microStream =
        encodeNullableChunk<int64_t>(micros, EncodingType::Trivial);
    const auto nanoStream = encodeChunk<uint16_t>(nanos, EncodingType::Trivial);

    const auto type = std::make_shared<const RowType>(
        StreamDescriptor{0, ScalarKind::Bool},
        std::vector<std::string>{
            "scalar", "array", "map", "nestedArray", "timestamp"},
        std::vector<std::shared_ptr<const Type>>{
            std::make_shared<const ScalarType>(
                StreamDescriptor{1, ScalarKind::Int64}),
            std::make_shared<const ArrayType>(
                StreamDescriptor{2, ScalarKind::UInt32},
                std::make_shared<const ScalarType>(
                    StreamDescriptor{3, ScalarKind::Int64})),
            std::make_shared<const MapType>(
                StreamDescriptor{4, ScalarKind::UInt32},
                std::make_shared<const ScalarType>(
                    StreamDescriptor{5, ScalarKind::String}),
                std::make_shared<const ScalarType>(
                    StreamDescriptor{6, ScalarKind::Int64})),
            std::make_shared<const ArrayType>(
                StreamDescriptor{7, ScalarKind::UInt32},
                std::make_shared<const ArrayType>(
                    StreamDescriptor{8, ScalarKind::UInt32},
                    std::make_shared<const ScalarType>(
                        StreamDescriptor{9, ScalarKind::Int64}))),
            std::make_shared<const TimestampMicroNanoType>(
                StreamDescriptor{10, ScalarKind::Int64},
                StreamDescriptor{11, ScalarKind::UInt16}),
        });
    const std::array<std::string_view, 12> streams{
        std::string_view{},
        scalarStream,
        arrayLengthStream,
        arrayElementStream,
        mapLengthStream,
        mapKeyStream,
        mapValueStream,
        outerLengthStream,
        innerLengthStream,
        innerElementStream,
        microStream,
        nanoStream};

    const auto ranges = makeRandomRanges(numRows, rng);
    velox::vector_size_t numSelectedRows{0};
    for (const auto& range : ranges) {
      numSelectedRows += static_cast<velox::vector_size_t>(range.numRows());
    }
    const auto select = [&](const auto& source) {
      std::decay_t<decltype(source)> selected;
      selected.reserve(numSelectedRows);
      for (const auto& range : ranges) {
        selected.insert(
            selected.end(),
            source.begin() + range.startRow,
            source.begin() + range.endRow);
      }
      return selected;
    };

    const auto outputOffset = static_cast<velox::vector_size_t>(rng() % 4);
    auto output = velox::BaseVector::create(
        velox::ROW(
            {{"scalar", velox::BIGINT()},
             {"array", velox::ARRAY(velox::BIGINT())},
             {"map", velox::MAP(velox::VARCHAR(), velox::BIGINT())},
             {"nestedArray", velox::ARRAY(velox::ARRAY(velox::BIGINT()))},
             {"timestamp", velox::TIMESTAMP()}}),
        outputOffset + numSelectedRows,
        pool_.get());
    StreamReader reader{type, pool_.get(), {}};

    runRead(reader, streams, ranges, outputOffset, output);

    const auto* rowVector = output->as<velox::RowVector>();
    ASSERT_NE(rowVector, nullptr);
    EXPECT_THAT(
        readScalars(rowVector->childAt(0), outputOffset, numSelectedRows),
        ::testing::ElementsAreArray(select(scalars)));
    EXPECT_THAT(
        readArrays(rowVector->childAt(1), outputOffset, numSelectedRows),
        ::testing::ElementsAreArray(select(expectedArrays)));
    EXPECT_THAT(
        readMaps(rowVector->childAt(2), outputOffset, numSelectedRows),
        ::testing::ElementsAreArray(select(expectedMaps)));
    EXPECT_THAT(
        readNestedArrays(rowVector->childAt(3), outputOffset, numSelectedRows),
        ::testing::ElementsAreArray(select(expectedNestedArrays)));
    EXPECT_THAT(
        readTimestamps(rowVector->childAt(4), outputOffset, numSelectedRows),
        ::testing::ElementsAreArray(select(expectedTimestamps)));
  }
}

TEST_F(StreamReaderTest, writesNullForAbsentScalarStream) {
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"value"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}),
      });
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 2> streams{};
  auto output = velox::BaseVector::create(
      velox::ROW("value", velox::BIGINT()), 2, pool_.get());
  const std::array<RowRange, 1> ranges{{{3, 5}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  const auto* row = output->as<velox::RowVector>();
  ASSERT_NE(row, nullptr);
  EXPECT_FALSE(row->isNullAt(0));
  EXPECT_FALSE(row->isNullAt(1));
  EXPECT_TRUE(row->childAt(0)->isNullAt(0));
  EXPECT_TRUE(row->childAt(0)->isNullAt(1));
}

TEST_F(StreamReaderTest, clearsReusedOutputForAbsentScalarStream) {
  const auto valueStream =
      encodeChunk<int64_t>(std::vector<int64_t>{10, 11}, EncodingType::Trivial);
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"value"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}),
      });
  StreamReader reader{type, pool_.get(), {}};
  auto output = velox::BaseVector::create(
      velox::ROW("value", velox::BIGINT()), 2, pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 2}}};

  runRead(
      reader,
      std::array<std::string_view, 2>{std::string_view{}, valueStream},
      ranges,
      /*outputOffset=*/0,
      output);
  auto* values = output->as<velox::RowVector>()->childAt(0).get();
  EXPECT_FALSE(values->isNullAt(0));
  EXPECT_FALSE(values->isNullAt(1));

  runRead(
      reader,
      std::array<std::string_view, 2>{},
      ranges,
      /*outputOffset=*/0,
      output);

  EXPECT_TRUE(values->isNullAt(0));
  EXPECT_TRUE(values->isNullAt(1));
}

TEST_F(StreamReaderTest, writesSelectedNullsIntoOutput) {
  const auto stream = encodeNullableChunk<int64_t>(
      {10, std::nullopt, 12, std::nullopt, 14}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 3, pool_.get());
  const std::array<RowRange, 2> ranges{{{0, 2}, {4, 5}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);
  const auto* flat = output->asFlatVector<int64_t>();
  ASSERT_NE(flat, nullptr);

  std::vector<std::optional<int64_t>> actual;
  actual.reserve(output->size());
  for (velox::vector_size_t row{0}; row < output->size(); ++row) {
    actual.push_back(
        flat->isNullAt(row) ? std::nullopt
                            : std::optional<int64_t>{flat->valueAt(row)});
  }
  EXPECT_EQ(
      actual, (std::vector<std::optional<int64_t>>{10, std::nullopt, 14}));
}

TEST_F(StreamReaderTest, clearsValueBufferForSelectedNulls) {
  const auto stream =
      encodeNullableChunk<int64_t>({10, std::nullopt}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::Int64});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::BIGINT(), 1, pool_.get());
  auto* flat = output->asFlatVector<int64_t>();
  ASSERT_NE(flat, nullptr);
  flat->set(0, 42);
  const std::array<RowRange, 1> ranges{{{1, 2}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  EXPECT_TRUE(flat->isNullAt(0));
  EXPECT_EQ(flat->rawValues()[0], int64_t{0});
}

TEST_F(StreamReaderTest, clearsStringValueBufferForSelectedNulls) {
  const auto stream = encodeNullableChunk<std::string_view>(
      {"value", std::nullopt}, EncodingType::Trivial);
  const auto type = std::make_shared<const ScalarType>(
      StreamDescriptor{0, ScalarKind::String});
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 1> streams{stream};
  auto output = velox::BaseVector::create(velox::VARCHAR(), 1, pool_.get());
  auto* flat = output->asFlatVector<velox::StringView>();
  ASSERT_NE(flat, nullptr);
  flat->set(0, velox::StringView{"stale"});
  const std::array<RowRange, 1> ranges{{{1, 2}}};

  runRead(reader, streams, ranges, /*outputOffset=*/0, output);

  EXPECT_TRUE(flat->isNullAt(0));
  EXPECT_TRUE(flat->rawValues()[0].empty());
}

TEST_F(StreamReaderTest, readsNullableEdgeCases) {
  const std::vector<std::vector<std::optional<int64_t>>> testCases{
      {std::nullopt, std::nullopt, std::nullopt},
      {10, 11, 12},
      {std::nullopt},
      {42},
  };

  for (const auto& values : testCases) {
    SCOPED_TRACE(values.size());
    const auto stream =
        encodeNullableChunk<int64_t>(values, EncodingType::Trivial);
    const auto type = std::make_shared<const ScalarType>(
        StreamDescriptor{0, ScalarKind::Int64});
    StreamReader reader{type, pool_.get(), {}};
    const std::array<std::string_view, 1> streams{stream};
    auto output = velox::BaseVector::create(
        velox::BIGINT(),
        static_cast<velox::vector_size_t>(values.size()),
        pool_.get());
    const std::array<RowRange, 1> ranges{
        {{0, static_cast<uint32_t>(values.size())}}};

    runRead(reader, streams, ranges, /*outputOffset=*/0, output);

    const auto* flat = output->asFlatVector<int64_t>();
    ASSERT_NE(flat, nullptr);
    std::vector<std::optional<int64_t>> actual;
    actual.reserve(values.size());
    for (velox::vector_size_t row{0}; row < output->size(); ++row) {
      actual.push_back(
          flat->isNullAt(row) ? std::nullopt
                              : std::optional<int64_t>{flat->valueAt(row)});
    }
    EXPECT_EQ(actual, values);
  }
}

} // namespace
} // namespace facebook::nimble
