/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include "velox/serializers/PrestoSerializer.h"
#include <folly/Random.h>
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class PrestoSerializerTest
    : public ::testing::TestWithParam<common::CompressionKind> {
 protected:
  static void SetUpTestCase() {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }

  void SetUp() override {
    pool_ = memory::addDefaultLeafMemoryPool();
    serde_ = std::make_unique<serializer::presto::PrestoVectorSerde>();
    vectorMaker_ = std::make_unique<test::VectorMaker>(pool_.get());
  }

  void sanityCheckEstimateSerializedSize(const RowVectorPtr& rowVector) {
    const auto numRows = rowVector->size();

    std::vector<IndexRange> rows(numRows);
    for (int i = 0; i < numRows; i++) {
      rows[i] = IndexRange{i, 1};
    }

    std::vector<vector_size_t> rowSizes(numRows, 0);
    std::vector<vector_size_t*> rawRowSizes(numRows);
    for (auto i = 0; i < numRows; i++) {
      rawRowSizes[i] = &rowSizes[i];
    }
    serde_->estimateSerializedSize(
        rowVector, folly::Range(rows.data(), numRows), rawRowSizes.data());
  }

  serializer::presto::PrestoVectorSerde::PrestoOptions getParamSerdeOptions(
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    const bool useLosslessTimestamp =
        serdeOptions == nullptr ? false : serdeOptions->useLosslessTimestamp;
    common::CompressionKind kind = GetParam();
    serializer::presto::PrestoVectorSerde::PrestoOptions paramOptions{
        useLosslessTimestamp, kind};
    return paramOptions;
  }

  void serialize(
      const RowVectorPtr& rowVector,
      std::ostream* output,
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    auto streamInitialSize = output->tellp();
    sanityCheckEstimateSerializedSize(rowVector);

    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto rowType = asRowType(rowVector->type());
    auto numRows = rowVector->size();
    auto paramOptions = getParamSerdeOptions(serdeOptions);
    auto serializer =
        serde_->createSerializer(rowType, numRows, arena.get(), &paramOptions);

    serializer->append(rowVector);
    auto size = serializer->maxSerializedSize();
    facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
    OStreamOutputStream out(output, &listener);
    serializer->flush(&out);
    if (paramOptions.compressionKind == common::CompressionKind_NONE) {
      ASSERT_EQ(size, out.tellp() - streamInitialSize);
    } else {
      ASSERT_GE(size, out.tellp() - streamInitialSize);
    }
  }

  void serializeRle(
      const RowVectorPtr& rowVector,
      std::ostream* output,
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
    OStreamOutputStream out(output, &listener);
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto paramOptions = getParamSerdeOptions(serdeOptions);
    serde_->serializeConstants(rowVector, arena.get(), &paramOptions, &out);
  }

  std::unique_ptr<ByteStream> toByteStream(const std::string& input) {
    auto byteStream = std::make_unique<ByteStream>();
    ByteRange byteRange{
        reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())),
        (int32_t)input.length(),
        0};
    byteStream->resetInput({byteRange});
    return byteStream;
  }

  RowVectorPtr deserialize(
      const RowTypePtr& rowType,
      const std::string& input,
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    auto byteStream = toByteStream(input);
    auto paramOptions = getParamSerdeOptions(serdeOptions);
    RowVectorPtr result;
    serde_->deserialize(
        byteStream.get(), pool_.get(), rowType, &result, &paramOptions);
    return result;
  }

  RowVectorPtr makeTestVector(vector_size_t size) {
    auto a = vectorMaker_->flatVector<int64_t>(
        size, [](vector_size_t row) { return row; });
    auto b = vectorMaker_->flatVector<double>(
        size, [](vector_size_t row) { return row * 0.1; });

    std::vector<VectorPtr> childVectors = {a, b};

    return vectorMaker_->rowVector(childVectors);
  }

  void testRoundTrip(
      VectorPtr vector,
      const serializer::presto::PrestoVectorSerde::PrestoOptions* serdeOptions =
          nullptr) {
    auto rowVector = vectorMaker_->rowVector({vector});
    std::ostringstream out;
    serialize(rowVector, &out, serdeOptions);

    auto rowType = asRowType(rowVector->type());
    auto deserialized = deserialize(rowType, out.str(), serdeOptions);
    assertEqualVectors(deserialized, rowVector);
  }

  void testRleRoundTrip(
      const VectorPtr& constantVector,
      const serializer::presto::PrestoVectorSerde::PrestoOptions* serdeOptions =
          nullptr) {
    auto rowVector = vectorMaker_->rowVector({constantVector});
    std::ostringstream out;
    serializeRle(rowVector, &out, serdeOptions);

    auto rowType = asRowType(rowVector->type());
    auto deserialized = deserialize(rowType, out.str(), serdeOptions);

    assertEqualVectors(rowVector, deserialized);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<serializer::presto::PrestoVectorSerde> serde_;
  std::unique_ptr<test::VectorMaker> vectorMaker_;
};

TEST_P(PrestoSerializerTest, basic) {
  vector_size_t numRows = 1'000;
  auto rowVector = makeTestVector(numRows);
  testRoundTrip(rowVector);
}

/// Test serialization of a dictionary vector that adds nulls to the base
/// vector.
TEST_P(PrestoSerializerTest, dictionaryWithExtraNulls) {
  vector_size_t size = 1'000;

  auto base =
      vectorMaker_->flatVector<int64_t>(10, [](auto row) { return row; });

  BufferPtr nulls = AlignedBuffer::allocate<bool>(size, pool_.get());
  auto rawNulls = nulls->asMutable<uint64_t>();
  for (auto i = 0; i < size; i++) {
    bits::setNull(rawNulls, i, i % 5 == 0);
  }

  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(size, pool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (auto i = 0; i < size; i++) {
    if (i % 5 != 0) {
      rawIndices[i] = i % 10;
    }
  }

  auto dictionary = BaseVector::wrapInDictionary(nulls, indices, size, base);
  testRoundTrip(dictionary);
}

TEST_P(PrestoSerializerTest, emptyPage) {
  auto rowVector = vectorMaker_->rowVector(ROW({"a"}, {BIGINT()}), 0);

  std::ostringstream out;
  serialize(rowVector, &out, nullptr);

  auto rowType = asRowType(rowVector->type());
  auto deserialized = deserialize(rowType, out.str(), nullptr);
  assertEqualVectors(deserialized, rowVector);
}

TEST_P(PrestoSerializerTest, emptyArray) {
  auto arrayVector = vectorMaker_->arrayVector<int32_t>(
      1'000,
      [](vector_size_t row) { return row % 5; },
      [](vector_size_t row) { return row; });

  testRoundTrip(arrayVector);
}

TEST_P(PrestoSerializerTest, emptyMap) {
  auto mapVector = vectorMaker_->mapVector<int32_t, int32_t>(
      1'000,
      [](vector_size_t row) { return row % 5; },
      [](vector_size_t row) { return row; },
      [](vector_size_t row) { return row * 2; });

  testRoundTrip(mapVector);
}

TEST_P(PrestoSerializerTest, timestampWithTimeZone) {
  auto timestamp = vectorMaker_->flatVector<int64_t>(
      100, [](auto row) { return 10'000 + row; });
  auto timezone =
      vectorMaker_->flatVector<int16_t>(100, [](auto row) { return row % 37; });

  auto vector = std::make_shared<RowVector>(
      pool_.get(),
      TIMESTAMP_WITH_TIME_ZONE(),
      BufferPtr(nullptr),
      100,
      std::vector<VectorPtr>{timestamp, timezone});

  testRoundTrip(vector);

  // Add some nulls.
  for (auto i = 0; i < 100; i += 7) {
    vector->setNull(i, true);
  }
  testRoundTrip(vector);
}

TEST_P(PrestoSerializerTest, intervalDayTime) {
  auto vector = vectorMaker_->flatVector<int64_t>(
      100,
      [](auto row) { return row + folly::Random::rand32(); },
      nullptr, // nullAt
      INTERVAL_DAY_TIME());

  testRoundTrip(vector);

  // Add some nulls.
  for (auto i = 0; i < 100; i += 7) {
    vector->setNull(i, true);
  }
  testRoundTrip(vector);
}

TEST_P(PrestoSerializerTest, unknown) {
  const vector_size_t size = 123;
  auto constantVector =
      BaseVector::createNullConstant(UNKNOWN(), 123, pool_.get());
  testRoundTrip(constantVector);

  auto flatVector = BaseVector::create(UNKNOWN(), size, pool_.get());
  for (auto i = 0; i < size; i++) {
    flatVector->setNull(i, true);
  }
  testRoundTrip(flatVector);
}

TEST_P(PrestoSerializerTest, multiPage) {
  std::ostringstream out;

  // page 1
  auto a = makeTestVector(1'234);
  serialize(a, &out, nullptr);

  // page 2
  auto b = makeTestVector(538);
  serialize(b, &out, nullptr);

  // page 3
  auto c = makeTestVector(2'048);
  serialize(c, &out, nullptr);

  auto bytes = out.str();

  auto rowType = asRowType(a->type());
  auto byteStream = toByteStream(bytes);

  RowVectorPtr deserialized;
  auto paramOptions = getParamSerdeOptions(nullptr);
  serde_->deserialize(
      byteStream.get(), pool_.get(), rowType, &deserialized, &paramOptions);
  ASSERT_FALSE(byteStream->atEnd());
  assertEqualVectors(deserialized, a);

  serde_->deserialize(
      byteStream.get(), pool_.get(), rowType, &deserialized, &paramOptions);
  assertEqualVectors(deserialized, b);
  ASSERT_FALSE(byteStream->atEnd());

  serde_->deserialize(
      byteStream.get(), pool_.get(), rowType, &deserialized, &paramOptions);
  assertEqualVectors(deserialized, c);
  ASSERT_TRUE(byteStream->atEnd());
}

TEST_P(PrestoSerializerTest, timestampWithNanosecondPrecision) {
  // Verify that nanosecond precision is preserved when the right options are
  // passed to the serde.
  const serializer::presto::PrestoVectorSerde::PrestoOptions
      kUseLosslessTimestampOptions(
          true, common::CompressionKind::CompressionKind_NONE);
  auto timestamp = vectorMaker_->flatVector<Timestamp>(
      {Timestamp{0, 0},
       Timestamp{12, 0},
       Timestamp{0, 17'123'456},
       Timestamp{1, 17'123'456},
       Timestamp{-1, 17'123'456}});
  testRoundTrip(timestamp, &kUseLosslessTimestampOptions);

  // Verify that precision is lost when no option is passed to the serde.
  auto timestampMillis = vectorMaker_->flatVector<Timestamp>(
      {Timestamp{0, 0},
       Timestamp{12, 0},
       Timestamp{0, 17'000'000},
       Timestamp{1, 17'000'000},
       Timestamp{-1, 17'000'000}});
  auto inputRowVector = vectorMaker_->rowVector({timestamp});
  auto expectedOutputWithLostPrecision =
      vectorMaker_->rowVector({timestampMillis});
  std::ostringstream out;
  serialize(inputRowVector, &out, {});
  auto rowType = asRowType(inputRowVector->type());
  auto deserialized = deserialize(rowType, out.str(), {});
  assertEqualVectors(deserialized, expectedOutputWithLostPrecision);
}

TEST_P(PrestoSerializerTest, longDecimal) {
  std::vector<int128_t> decimalValues(102);
  decimalValues[0] = DecimalUtil::kLongDecimalMin;
  for (int row = 1; row < 101; row++) {
    decimalValues[row] = row - 50;
  }
  decimalValues[101] = DecimalUtil::kLongDecimalMax;
  auto vector =
      vectorMaker_->flatVector<int128_t>(decimalValues, DECIMAL(20, 5));

  testRoundTrip(vector);

  // Add some nulls.
  for (auto i = 0; i < 102; i += 7) {
    vector->setNull(i, true);
  }
  testRoundTrip(vector);
}

TEST_P(PrestoSerializerTest, rle) {
  // Test RLE vectors with non-null value.
  testRleRoundTrip(
      BaseVector::createConstant(BOOLEAN(), true, 12, pool_.get()));
  testRleRoundTrip(BaseVector::createConstant(INTEGER(), 779, 12, pool_.get()));
  testRleRoundTrip(BaseVector::createConstant(DOUBLE(), 1.23, 12, pool_.get()));
  testRleRoundTrip(
      BaseVector::createConstant(VARCHAR(), "Hello, world!", 12, pool_.get()));
  testRleRoundTrip(BaseVector::wrapInConstant(
      12, 0, vectorMaker_->arrayVector<int64_t>({{1, 2, 3}})));
  testRleRoundTrip(BaseVector::wrapInConstant(
      12,
      0,
      vectorMaker_->mapVector(
          {0},
          vectorMaker_->flatVector<int32_t>({1}),
          vectorMaker_->flatVector<int64_t>({7}))));

  // Test RLE vector with null value.
  testRleRoundTrip(BaseVector::createNullConstant(BOOLEAN(), 17, pool_.get()));
  testRleRoundTrip(BaseVector::createNullConstant(BIGINT(), 17, pool_.get()));
  testRleRoundTrip(BaseVector::createNullConstant(REAL(), 17, pool_.get()));
  testRleRoundTrip(BaseVector::createNullConstant(VARCHAR(), 17, pool_.get()));
  testRleRoundTrip(
      BaseVector::createNullConstant(ARRAY(INTEGER()), 17, pool_.get()));
  testRleRoundTrip(BaseVector::createNullConstant(
      MAP(VARCHAR(), INTEGER()), 17, pool_.get()));
}

TEST_P(PrestoSerializerTest, lazy) {
  constexpr int kSize = 1000;
  auto rowVector = makeTestVector(kSize);
  auto lazyVector = std::make_shared<LazyVector>(
      pool_.get(),
      rowVector->type(),
      kSize,
      std::make_unique<SimpleVectorLoader>([&](auto) { return rowVector; }));
  testRoundTrip(lazyVector);
}

TEST_P(PrestoSerializerTest, ioBufRoundTrip) {
  VectorFuzzer::Options opts;
  opts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
  opts.nullRatio = 0.1;
  VectorFuzzer fuzzer(opts, pool_.get());

  const size_t numRounds = 20;

  for (size_t i = 0; i < numRounds; ++i) {
    auto rowType = fuzzer.randRowType();
    auto inputRowVector = fuzzer.fuzzInputRow(rowType);
    auto outputRowVector = IOBufToRowVector(
        rowVectorToIOBuf(inputRowVector, *pool_), rowType, *pool_);

    assertEqualVectors(inputRowVector, outputRowVector);
  }
}

INSTANTIATE_TEST_SUITE_P(
    PrestoSerializerTest,
    PrestoSerializerTest,
    ::testing::Values(
        common::CompressionKind::CompressionKind_NONE,
        common::CompressionKind::CompressionKind_ZLIB,
        common::CompressionKind::CompressionKind_SNAPPY,
        common::CompressionKind::CompressionKind_ZSTD,
        common::CompressionKind::CompressionKind_LZ4));
