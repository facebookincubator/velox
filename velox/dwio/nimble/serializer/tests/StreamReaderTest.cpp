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

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <folly/coro/BlockingWait.h>
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

TEST_F(StreamReaderTest, rejectsNullableRowStream) {
  const std::array<bool, 3> rowIsNonNull{true, false, true};
  const auto rowStream = encodeChunk<bool>(rowIsNonNull, EncodingType::Trivial);
  const auto valueStream =
      encodeChunk<int64_t>(std::vector<int64_t>{10, 12}, EncodingType::Trivial);
  const auto type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool},
      std::vector<std::string>{"value"},
      std::vector<std::shared_ptr<const Type>>{
          std::make_shared<const ScalarType>(
              StreamDescriptor{1, ScalarKind::Int64}),
      });
  StreamReader reader{type, pool_.get(), {}};
  const std::array<std::string_view, 2> streams{rowStream, valueStream};
  auto output = velox::BaseVector::create(
      velox::ROW("value", velox::BIGINT()), 1, pool_.get());
  const std::array<RowRange, 1> ranges{{{0, 1}}};

  NIMBLE_ASSERT_THROW(
      runRead(reader, streams, ranges, /*outputOffset=*/0, output),
      "Selective row decoding does not support nullable row streams");
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
