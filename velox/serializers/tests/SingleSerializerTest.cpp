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
#include "velox/serializers/SingleSerializer.h"
#include <folly/Random.h>
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class SingleSerializerTest : public ::testing::Test,
                             public test::VectorTestBase {
 protected:
  void SetUp() override {
    pool_ = memory::addDefaultLeafMemoryPool();
    serde_ = std::make_unique<serializer::SingleVectorSerde>();
    vectorMaker_ = std::make_unique<test::VectorMaker>(pool_.get());
  }

  void sanityCheckEstimateSerializedSize(
      RowVectorPtr rowVector,
      const folly::Range<const IndexRange*>& ranges) {
    auto numRows = rowVector->size();
    std::vector<vector_size_t> rowSizes(numRows, 0);
    std::vector<vector_size_t*> rawRowSizes(numRows);
    for (auto i = 0; i < numRows; i++) {
      rawRowSizes[i] = &rowSizes[i];
    }
    serde_->estimateSerializedSize(rowVector, ranges, rawRowSizes.data());
  }

  void serialize(
      const RowVectorPtr& rowVector,
      std::ostream* output,
      const VectorSerde::Options* serdeOptions) {
    auto numRows = rowVector->size();

    std::vector<IndexRange> rows(numRows);
    for (int i = 0; i < numRows; i++) {
      rows[i] = IndexRange{i, 1};
    }

    sanityCheckEstimateSerializedSize(
        rowVector, folly::Range(rows.data(), numRows));

    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto rowType = asRowType(rowVector->type());
    auto serializer =
        serde_->createSerializer(rowType, numRows, arena.get(), serdeOptions);

    serializer->append(rowVector, folly::Range(rows.data(), numRows));
    vector_size_t size = serializer->maxSerializedSize();
    facebook::velox::serializer::SingleOutputStreamListener listener;
    OStreamOutputStream out(output, &listener);
    serializer->flush(&out);
    ASSERT_EQ(size, out.tellp());
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
      std::shared_ptr<const RowType> rowType,
      const std::string& input,
      const VectorSerde::Options* serdeOptions) {
    auto byteStream = toByteStream(input);
    RowVectorPtr result;
    serde_->deserialize(
        byteStream.get(), pool_.get(), rowType, &result, serdeOptions);
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

  // just test rowvector with flat vector
  void testRoundTripRowVector(
      RowVectorPtr rowVector,
      const VectorSerde::Options* serdeOptions = nullptr) {
    std::ostringstream out;
    serialize(rowVector, &out, serdeOptions);

    auto rowType = asRowType(rowVector->type());
    auto deserialized = deserialize(rowType, out.str(), serdeOptions);
    assertEqualVectors(deserialized, rowVector);
  }

  void testRoundTrip(
      VectorPtr vector,
      const VectorSerde::Options* serdeOptions = nullptr) {
    auto rowVector = vectorMaker_->rowVector({vector});
    std::ostringstream out;
    serialize(rowVector, &out, serdeOptions);

    auto rowType = asRowType(rowVector->type());
    auto deserialized = deserialize(rowType, out.str(), serdeOptions);
    assertEqualVectors(deserialized, rowVector);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<serializer::SingleVectorSerde> serde_;
  std::unique_ptr<test::VectorMaker> vectorMaker_;
};

TEST_F(SingleSerializerTest, basic) {
  vector_size_t numRows = 5;
  auto rowVector = makeTestVector(numRows);
  testRoundTripRowVector(rowVector);

  auto stringVector = makeRowVector({
      makeNullableFlatVector<int64_t>({std::nullopt, 1, 2, 3, 4}),
      makeFlatVector<StringView>({
          "ALGERIA",
          "ARGENTINA",
          "BRAZIL",
          "CANADA",
          "EGYPT",
      }),
  });
  testRoundTripRowVector(stringVector);
}

TEST_F(SingleSerializerTest, constant) {
  auto constVector = makeRowVector({
      makeConstant<int64_t>(100, 5),
      makeConstant<StringView>("ALGERIA", 5),
  });
  testRoundTripRowVector(constVector);

  auto rowVector = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
      makeConstant<int64_t>(100, 5),
      makeConstant<UnscaledLongDecimal>(
          UnscaledLongDecimal(10012), 5, DECIMAL(20, 2)),
      makeConstant<UnscaledShortDecimal>(
          UnscaledShortDecimal(10012), 5, DECIMAL(10, 2)),
      makeFlatVector<int32_t>({0, 1, 2, 3, 4}),
      makeNullConstant(TypeKind::INTEGER, 5),
      makeConstant<StringView>("ALGERIA", 5),
  });
  testRoundTripRowVector(rowVector);

  auto timeVector = makeRowVector({
      makeConstant<int64_t>(100, 5),
      // If with Timestamp(100, 1), will throw exception
      // expected {100, 1970-01-01T00:01:40.000000000, 1970-01-16}, but got
      // {100, 1970-01-01T00:01:40.000000001, 1970-01-16}
      makeConstant<Timestamp>(Timestamp(100, 0), 5),
      makeConstant<Date>(Date(15), 5),
  });
  testRoundTripRowVector(timeVector);
}

TEST_F(SingleSerializerTest, emptyPage) {
  auto rowVector = vectorMaker_->rowVector(ROW({"a"}, {BIGINT()}), 0);

  std::ostringstream out;
  serialize(rowVector, &out, nullptr);

  auto rowType = asRowType(rowVector->type());
  auto deserialized = deserialize(rowType, out.str(), nullptr);
  assertEqualVectors(deserialized, rowVector);
}

TEST_F(SingleSerializerTest, unscaledLongDecimal) {
  std::vector<int128_t> decimalValues(102);
  decimalValues[0] = UnscaledLongDecimal::min().unscaledValue();
  for (int row = 1; row < 101; row++) {
    decimalValues[row] = row - 50;
  }
  decimalValues[101] = UnscaledLongDecimal::max().unscaledValue();
  auto vector =
      vectorMaker_->longDecimalFlatVector(decimalValues, DECIMAL(20, 5));

  testRoundTrip(vector);

  // Add some nulls.
  for (auto i = 0; i < 102; i += 7) {
    vector->setNull(i, true);
  }
  testRoundTrip(vector);
}
