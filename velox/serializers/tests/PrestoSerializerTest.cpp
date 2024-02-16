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
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/common/time/Timer.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

struct SerializeStats {
  int64_t actualSize{0};
  int64_t estimatedSize{0};
};

class PrestoSerializerTest
    : public ::testing::TestWithParam<common::CompressionKind>,
      public VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    serde_ = std::make_unique<serializer::presto::PrestoVectorSerde>();
  }

  void sanityCheckEstimateSerializedSize(const RowVectorPtr& rowVector) {
    Scratch scratch;
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
        rowVector,
        folly::Range(rows.data(), numRows),
        rawRowSizes.data(),
        scratch);
  }

  serializer::presto::PrestoVectorSerde::PrestoOptions getParamSerdeOptions(
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    const bool useLosslessTimestamp =
        serdeOptions == nullptr ? false : serdeOptions->useLosslessTimestamp;
    common::CompressionKind kind = GetParam();
    const bool nullsFirst =
        serdeOptions == nullptr ? false : serdeOptions->nullsFirst;
    serializer::presto::PrestoVectorSerde::PrestoOptions paramOptions{
        useLosslessTimestamp, kind, nullsFirst};

    return paramOptions;
  }

  SerializeStats serialize(
      const RowVectorPtr& rowVector,
      std::ostream* output,
      const serializer::presto::PrestoVectorSerde::PrestoOptions* serdeOptions,
      std::optional<folly::Range<const IndexRange*>> indexRanges = std::nullopt,
      std::optional<folly::Range<const vector_size_t*>> rows = std::nullopt) {
    auto streamInitialSize = output->tellp();
    sanityCheckEstimateSerializedSize(rowVector);

    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto rowType = asRowType(rowVector->type());
    auto numRows = rowVector->size();
    auto paramOptions = getParamSerdeOptions(serdeOptions);
    auto serializer = serde_->createIterativeSerializer(
        rowType, numRows, arena.get(), &paramOptions);
    vector_size_t sizeEstimate = 0;

    Scratch scratch;
    if (indexRanges.has_value()) {
      raw_vector<vector_size_t*> sizes(indexRanges.value().size());
      std::fill(sizes.begin(), sizes.end(), &sizeEstimate);
      serde_->estimateSerializedSize(
          rowVector, indexRanges.value(), sizes.data(), scratch);
      serializer->append(rowVector, indexRanges.value(), scratch);
    } else if (rows.has_value()) {
      raw_vector<vector_size_t*> sizes(rows.value().size());
      std::fill(sizes.begin(), sizes.end(), &sizeEstimate);
      serde_->estimateSerializedSize(
          rowVector, rows.value(), sizes.data(), scratch);
      serializer->append(rowVector, rows.value(), scratch);
    } else {
      vector_size_t* sizes = &sizeEstimate;
      IndexRange range{0, rowVector->size()};
      serde_->estimateSerializedSize(
          rowVector,
          folly::Range<const IndexRange*>(&range, 1),
          &sizes,
          scratch);
      serializer->append(rowVector);
    }
    auto size = serializer->maxSerializedSize();
    auto estimatePct = (100.0 * sizeEstimate) / static_cast<float>(size + 1);
    LOG(INFO) << "Size=" << size << " estimate=" << sizeEstimate << " "
              << estimatePct << "%";
    facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
    OStreamOutputStream out(output, &listener);
    serializer->flush(&out);
    if (paramOptions.compressionKind == common::CompressionKind_NONE) {
      EXPECT_EQ(size, out.tellp() - streamInitialSize);
    } else {
      EXPECT_GE(size, out.tellp() - streamInitialSize);
    }
    return {static_cast<int64_t>(size), sizeEstimate};
  }

  ByteInputStream toByteStream(const std::string& input) {
    ByteRange byteRange{
        reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())),
        (int32_t)input.length(),
        0};
    return ByteInputStream({byteRange});
  }

  void validateLexer(
      const std::string& input,
      const serializer::presto::PrestoVectorSerde::PrestoOptions&
          paramOptions) {
    if (paramOptions.useLosslessTimestamp ||
        paramOptions.compressionKind !=
            common::CompressionKind::CompressionKind_NONE ||
        paramOptions.nullsFirst) {
      // Unsupported options
      return;
    }
    auto byteStream = toByteStream(input);
    auto tokens =
        serializer::presto::PrestoVectorSerde::lex(&byteStream, &paramOptions);
    size_t tokenLengthSum = 0;
    for (auto const& token : tokens) {
      tokenLengthSum += token.length;
    }
    for (auto const& token : tokens) {
      // The lexer should not produce empty tokens
      EXPECT_NE(token.length, 0);
    }
    for (size_t i = 1; i < tokens.size(); ++i) {
      // The lexer should merge consecutive tokens of the same type
      EXPECT_NE(tokens[i].tokenType, tokens[i - 1].tokenType);
    }
    EXPECT_EQ(tokenLengthSum, input.size());
  }

  RowVectorPtr deserialize(
      const RowTypePtr& rowType,
      const std::string& input,
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    auto byteStream = toByteStream(input);
    auto paramOptions = getParamSerdeOptions(serdeOptions);
    validateLexer(input, paramOptions);
    RowVectorPtr result;
    serde_->deserialize(
        &byteStream, pool_.get(), rowType, &result, 0, &paramOptions);
    return result;
  }

  RowVectorPtr makeTestVector(vector_size_t size) {
    auto a =
        makeFlatVector<int64_t>(size, [](vector_size_t row) { return row; });
    auto b = makeFlatVector<double>(
        size, [](vector_size_t row) { return row * 0.1; });
    auto c = makeFlatVector<std::string>(size, [](vector_size_t row) {
      return row % 2 == 0 ? "LaaaaaaaaargeString" : "inlineStr";
    });

    std::vector<VectorPtr> childVectors = {a, b, c};

    return makeRowVector(childVectors);
  }

  RowVectorPtr wrapChildren(const RowVectorPtr& row) {
    auto children = row->children();
    std::vector<VectorPtr> newChildren = children;
    auto indices = makeIndices(row->size(), [](auto row) { return row; });
    for (auto& child : newChildren) {
      child = BaseVector::wrapInDictionary(
          BufferPtr(nullptr), indices, row->size(), child);
    }
    return makeRowVector(newChildren);
  }

  void testRoundTrip(
      VectorPtr vector,
      const serializer::presto::PrestoVectorSerde::PrestoOptions* serdeOptions =
          nullptr) {
    auto rowVector = makeRowVector({vector});
    std::ostringstream out;
    serialize(rowVector, &out, serdeOptions);

    auto rowType = asRowType(rowVector->type());
    auto deserialized = deserialize(rowType, out.str(), serdeOptions);
    assertEqualVectors(deserialized, rowVector);

    if (rowVector->size() < 3) {
      return;
    }

    // Split input into 3 batches. Serialize each separately. Then, deserialize
    // all into one vector.
    auto splits = split(rowVector, 3);
    std::vector<std::string> serialized;
    for (const auto& split : splits) {
      std::ostringstream out;
      serialize(split, &out, serdeOptions);
      serialized.push_back(out.str());
    }

    auto paramOptions = getParamSerdeOptions(serdeOptions);
    RowVectorPtr result;
    vector_size_t offset = 0;
    for (auto i = 0; i < serialized.size(); ++i) {
      auto byteStream = toByteStream(serialized[i]);
      serde_->deserialize(
          &byteStream, pool_.get(), rowType, &result, offset, &paramOptions);
      offset = result->size();
    }

    assertEqualVectors(result, rowVector);

    // Serialize the vector with even and odd rows in different partitions.
    auto even =
        makeIndices(rowVector->size() / 2, [&](auto row) { return row * 2; });
    auto odd = makeIndices(
        (rowVector->size() - 1) / 2, [&](auto row) { return (row * 2) + 1; });
    testSerializeRows(rowVector, even, serdeOptions);
    auto oddStats = testSerializeRows(rowVector, odd, serdeOptions);
    auto wrappedRowVector = wrapChildren(rowVector);
    auto wrappedStats = testSerializeRows(wrappedRowVector, odd, serdeOptions);
    EXPECT_EQ(oddStats.estimatedSize, wrappedStats.estimatedSize);
    EXPECT_EQ(oddStats.actualSize, wrappedStats.actualSize);
  }

  void testLexer(
      VectorPtr vector,
      const serializer::presto::PrestoVectorSerde::PrestoOptions* serdeOptions =
          nullptr) {
    auto rowVector = makeRowVector({vector});
    std::ostringstream out;
    serialize(rowVector, &out, serdeOptions);
    validateLexer(out.str(), getParamSerdeOptions(serdeOptions));
  }

  SerializeStats testSerializeRows(
      const RowVectorPtr& rowVector,
      BufferPtr indices,
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    std::ostringstream out;
    auto rows = folly::Range<const vector_size_t*>(
        indices->as<vector_size_t>(), indices->size() / sizeof(vector_size_t));
    auto stats = serialize(rowVector, &out, serdeOptions, std::nullopt, rows);

    auto rowType = asRowType(rowVector->type());
    auto deserialized = deserialize(rowType, out.str(), serdeOptions);
    assertEqualVectors(
        deserialized,
        BaseVector::wrapInDictionary(
            BufferPtr(nullptr),
            indices,
            indices->size() / sizeof(vector_size_t),
            rowVector));
    return stats;
  }

  void assertEqualEncoding(
      const RowVectorPtr& expected,
      const RowVectorPtr& actual) {
    for (auto i = 0; i < expected->childrenSize(); ++i) {
      VELOX_CHECK_EQ(
          actual->childAt(i)->encoding(), expected->childAt(i)->encoding());

      if (expected->childAt(i)->encoding() == VectorEncoding::Simple::ROW) {
        assertEqualEncoding(
            std::dynamic_pointer_cast<RowVector>(expected->childAt(i)),
            std::dynamic_pointer_cast<RowVector>(actual->childAt(i)));
      }
    }
  }

  void verifySerializedEncodedData(
      const RowVectorPtr& original,
      const std::string& serialized,
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    auto rowType = asRowType(original->type());
    auto deserialized = deserialize(rowType, serialized, serdeOptions);

    assertEqualVectors(original, deserialized);
    assertEqualEncoding(original, deserialized);

    // Deserialize 3 times while appending to a single vector.
    auto paramOptions = getParamSerdeOptions(serdeOptions);
    RowVectorPtr result;
    vector_size_t offset = 0;
    for (auto i = 0; i < 3; ++i) {
      auto byteStream = toByteStream(serialized);
      serde_->deserialize(
          &byteStream, pool_.get(), rowType, &result, offset, &paramOptions);
      offset = result->size();
    }

    auto expected =
        BaseVector::create(original->type(), original->size() * 3, pool());
    for (auto i = 0; i < 3; ++i) {
      expected->copy(original.get(), original->size() * i, 0, original->size());
    }

    assertEqualVectors(expected, result);
  }

  void makePermutations(
      const std::vector<VectorPtr>& vectors,
      int32_t size,
      std::vector<VectorPtr>& items,
      std::vector<std::vector<VectorPtr>>& result) {
    if (size == items.size()) {
      result.push_back(items);
      return;
    }
    for (const auto& vector : vectors) {
      items.push_back(vector);
      makePermutations(vectors, size, items, result);
      items.pop_back();
    }
  }

  // tests combining encodings in serialization and deserialization. Serializes
  // each of 'vectors' with encoding, then reads them back into a single vector.
  void testEncodedConcatenation(
      std::vector<VectorPtr>& vectors,
      const serializer::presto::PrestoVectorSerde::PrestoOptions* serdeOptions =
          nullptr) {
    std::vector<std::string> pieces;
    auto rowType = ROW({{"f", vectors[0]->type()}});
    auto concatenation = BaseVector::create(rowType, 0, pool_.get());
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto paramOptions = getParamSerdeOptions(serdeOptions);
    auto serializer =
        serde_->createSerializer(rowType, 10, arena.get(), &paramOptions);

    for (const auto& vector : vectors) {
      auto data = makeRowVector({"f"}, {vector});
      concatenation->append(data.get());
      std::ostringstream out;
      serializeBatch(data, &out, &paramOptions);
      pieces.push_back(out.str());
      serializer->append(data);
    }
    facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
    std::ostringstream allOut;
    OStreamOutputStream allOutStream(&allOut, &listener);
    serializer->flush(&allOutStream);

    auto allDeserialized = deserialize(rowType, allOut.str(), &paramOptions);
    assertEqualVectors(allDeserialized, concatenation);
    RowVectorPtr deserialized =
        BaseVector::create<RowVector>(rowType, 0, pool_.get());
    for (auto& piece : pieces) {
      auto byteStream = toByteStream(piece);
      serde_->deserialize(
          &byteStream,
          pool_.get(),
          rowType,
          &deserialized,
          deserialized->size(),
          &paramOptions);
    }
    assertEqualVectors(concatenation, deserialized);
  }

  void testEncodedRoundTrip(
      const RowVectorPtr& data,
      const serializer::presto::PrestoVectorSerde::PrestoOptions* serdeOptions =
          nullptr) {
    std::ostringstream out;
    serializeBatch(data, &out, serdeOptions);
    const auto serialized = out.str();

    verifySerializedEncodedData(data, serialized, serdeOptions);
  }

  void serializeBatch(
      const RowVectorPtr& rowVector,
      std::ostream* output,
      const serializer::presto::PrestoVectorSerde::PrestoOptions*
          serdeOptions) {
    facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
    OStreamOutputStream out(output, &listener);
    auto paramOptions = getParamSerdeOptions(serdeOptions);

    auto serializer = serde_->createBatchSerializer(pool_.get(), &paramOptions);
    serializer->serialize(rowVector, &out);
  }

  void testBatchVectorSerializerRoundTrip(
      const RowVectorPtr& data,
      const serializer::presto::PrestoVectorSerde::PrestoOptions* serdeOptions =
          nullptr) {
    std::ostringstream out;
    serializeBatch(data, &out, serdeOptions);
    const auto serialized = out.str();

    verifySerializedEncodedData(data, serialized, serdeOptions);
  }

  RowVectorPtr encodingsTestVector() {
    auto baseNoNulls = makeFlatVector<int64_t>({1, 2, 3, 4});
    auto baseWithNulls =
        makeNullableFlatVector<int32_t>({1, std::nullopt, 2, 3});
    auto baseArray =
        makeArrayVector<int32_t>({{1, 2, 3}, {}, {4, 5}, {6, 7, 8, 9, 10}});
    auto indices = makeIndices(8, [](auto row) { return row / 2; });

    return makeRowVector({
        BaseVector::wrapInDictionary(nullptr, indices, 8, baseNoNulls),
        BaseVector::wrapInDictionary(nullptr, indices, 8, baseWithNulls),
        BaseVector::wrapInDictionary(nullptr, indices, 8, baseArray),
        BaseVector::createConstant(INTEGER(), 123, 8, pool_.get()),
        BaseVector::createNullConstant(VARCHAR(), 8, pool_.get()),
        BaseVector::wrapInConstant(8, 1, baseArray),
        BaseVector::wrapInConstant(8, 2, baseArray),
        makeRowVector({
            BaseVector::wrapInDictionary(nullptr, indices, 8, baseNoNulls),
            BaseVector::wrapInDictionary(nullptr, indices, 8, baseWithNulls),
            BaseVector::wrapInDictionary(nullptr, indices, 8, baseArray),
            BaseVector::createConstant(INTEGER(), 123, 8, pool_.get()),
            BaseVector::createNullConstant(VARCHAR(), 8, pool_.get()),
            BaseVector::wrapInConstant(8, 1, baseArray),
            BaseVector::wrapInConstant(8, 2, baseArray),
            makeRowVector({
                BaseVector::wrapInDictionary(
                    nullptr, indices, 8, baseWithNulls),
                BaseVector::createConstant(INTEGER(), 123, 8, pool_.get()),
                BaseVector::wrapInConstant(8, 2, baseArray),
            }),
        }),
    });
  }

  RowVectorPtr encodingsArrayElementsTestVector() {
    auto baseNoNulls = makeFlatVector<int64_t>({1, 2, 3, 4});
    auto baseWithNulls =
        makeNullableFlatVector<int32_t>({1, std::nullopt, 2, 3});
    auto baseArray =
        makeArrayVector<int32_t>({{1, 2, 3}, {}, {4, 5}, {6, 7, 8, 9, 10}});
    auto elementIndices = makeIndices(16, [](auto row) { return row / 4; });
    std::vector<vector_size_t> offsets{0, 2, 4, 6, 8, 10, 12, 14, 16};

    return makeRowVector({
        makeArrayVector(
            offsets,
            BaseVector::wrapInDictionary(
                nullptr, elementIndices, 16, baseNoNulls)),
        makeArrayVector(
            offsets,
            BaseVector::wrapInDictionary(
                nullptr, elementIndices, 16, baseWithNulls)),
        makeArrayVector(
            offsets,
            BaseVector::wrapInDictionary(
                nullptr, elementIndices, 16, baseArray)),
        makeArrayVector(
            offsets,
            BaseVector::createConstant(INTEGER(), 123, 16, pool_.get())),
        makeArrayVector(
            offsets,
            BaseVector::createNullConstant(VARCHAR(), 16, pool_.get())),
        makeArrayVector(offsets, BaseVector::wrapInConstant(16, 1, baseArray)),
        makeRowVector({
            makeArrayVector(
                offsets,
                BaseVector::wrapInDictionary(
                    nullptr, elementIndices, 16, baseNoNulls)),
            makeArrayVector(
                offsets,
                BaseVector::wrapInDictionary(
                    nullptr, elementIndices, 16, baseWithNulls)),
            makeArrayVector(
                offsets,
                BaseVector::wrapInDictionary(
                    nullptr, elementIndices, 16, baseArray)),
            makeArrayVector(
                offsets,
                BaseVector::createConstant(INTEGER(), 123, 16, pool_.get())),
            makeArrayVector(
                offsets,
                BaseVector::createNullConstant(VARCHAR(), 16, pool_.get())),
            makeArrayVector(
                offsets, BaseVector::wrapInConstant(16, 1, baseArray)),
        }),
    });
  }

  RowVectorPtr encodingsMapValuesTestVector() {
    auto baseNoNulls = makeFlatVector<int64_t>({1, 2, 3, 4});
    auto baseWithNulls =
        makeNullableFlatVector<int32_t>({1, std::nullopt, 2, 3});
    auto baseArray =
        makeArrayVector<int32_t>({{1, 2, 3}, {}, {4, 5}, {6, 7, 8, 9, 10}});
    auto valueIndices = makeIndices(16, [](auto row) { return row / 4; });
    std::vector<vector_size_t> offsets{0, 2, 4, 6, 8, 10, 12, 14, 16};
    auto mapKeys = makeFlatVector<int32_t>(16, [](auto row) { return row; });

    return makeRowVector({
        makeMapVector(
            offsets,
            mapKeys,
            BaseVector::wrapInDictionary(
                nullptr, valueIndices, 16, baseNoNulls)),
        makeMapVector(
            offsets,
            mapKeys,
            BaseVector::wrapInDictionary(
                nullptr, valueIndices, 16, baseWithNulls)),
        makeMapVector(
            offsets,
            mapKeys,
            BaseVector::wrapInDictionary(nullptr, valueIndices, 16, baseArray)),
        makeMapVector(
            offsets,
            mapKeys,
            BaseVector::createConstant(INTEGER(), 123, 16, pool_.get())),
        makeMapVector(
            offsets,
            mapKeys,
            BaseVector::createNullConstant(VARCHAR(), 16, pool_.get())),
        makeMapVector(
            offsets, mapKeys, BaseVector::wrapInConstant(16, 1, baseArray)),
        makeRowVector({
            makeMapVector(
                offsets,
                mapKeys,
                BaseVector::wrapInDictionary(
                    nullptr, valueIndices, 16, baseNoNulls)),
            makeMapVector(
                offsets,
                mapKeys,
                BaseVector::wrapInDictionary(
                    nullptr, valueIndices, 16, baseWithNulls)),
            makeMapVector(
                offsets,
                mapKeys,
                BaseVector::wrapInDictionary(
                    nullptr, valueIndices, 16, baseArray)),
            makeMapVector(
                offsets,
                mapKeys,
                BaseVector::createConstant(INTEGER(), 123, 16, pool_.get())),
            makeMapVector(
                offsets,
                mapKeys,
                BaseVector::createNullConstant(VARCHAR(), 16, pool_.get())),
            makeMapVector(
                offsets, mapKeys, BaseVector::wrapInConstant(16, 1, baseArray)),
        }),
    });
  }

  std::unique_ptr<serializer::presto::PrestoVectorSerde> serde_;
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

  auto base = makeFlatVector<int64_t>(10, [](auto row) { return row; });

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
  auto rowVector = makeRowVector(ROW({"a"}, {BIGINT()}), 0);

  std::ostringstream out;
  serialize(rowVector, &out, nullptr);

  auto rowType = asRowType(rowVector->type());
  auto deserialized = deserialize(rowType, out.str(), nullptr);
  assertEqualVectors(deserialized, rowVector);
}

TEST_P(PrestoSerializerTest, emptyArray) {
  auto arrayVector = makeArrayVector<int32_t>(
      1'000,
      [](vector_size_t row) { return row % 5; },
      [](vector_size_t row) { return row; });

  testRoundTrip(arrayVector);
}

TEST_P(PrestoSerializerTest, emptyMap) {
  auto mapVector = makeMapVector<int32_t, int32_t>(
      1'000,
      [](vector_size_t row) { return row % 5; },
      [](vector_size_t row) { return row; },
      [](vector_size_t row) { return row * 2; });

  testRoundTrip(mapVector);
}

TEST_P(PrestoSerializerTest, timestampWithTimeZone) {
  auto timestamp =
      makeFlatVector<int64_t>(100, [](auto row) { return 10'000 + row; });
  auto timezone =
      makeFlatVector<int16_t>(100, [](auto row) { return row % 37; });

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
  auto vector = makeFlatVector<int64_t>(
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
  // Verify vectors of UNKNOWN type. Also verifies a special case where a
  // vector, not of UNKNOWN type and with all nulls is serialized as an UNKNOWN
  // type having BYTE_ARRAY encoding.
  auto testAllNullSerializedAsUnknown = [&](VectorPtr vector,
                                            TypePtr outputType) {
    auto rowVector = makeRowVector({vector});
    auto expected = makeRowVector(
        {BaseVector::createNullConstant(outputType, vector->size(), pool())});
    std::ostringstream out;
    serialize(rowVector, &out, nullptr);

    auto rowType = asRowType(expected->type());
    auto deserialized = deserialize(rowType, out.str(), nullptr);
    assertEqualVectors(expected, deserialized);

    if (rowVector->size() < 3) {
      return;
    }

    // Split input into 3 batches. Serialize each separately. Then, deserialize
    // all into one vector.
    auto splits = split(rowVector, 3);
    std::vector<std::string> serialized;
    for (const auto& split : splits) {
      std::ostringstream oss;
      serialize(split, &oss, nullptr);
      serialized.push_back(oss.str());
    }

    auto paramOptions = getParamSerdeOptions(nullptr);
    RowVectorPtr result;
    vector_size_t offset = 0;
    for (auto i = 0; i < serialized.size(); ++i) {
      auto byteStream = toByteStream(serialized[i]);
      serde_->deserialize(
          &byteStream, pool_.get(), rowType, &result, offset, &paramOptions);
      offset = result->size();
    }

    assertEqualVectors(expected, result);
  };

  const vector_size_t size = 123;
  auto constantVector =
      BaseVector::createNullConstant(UNKNOWN(), size, pool_.get());
  testRoundTrip(constantVector);
  testAllNullSerializedAsUnknown(constantVector, BIGINT());

  auto flatVector = BaseVector::create(UNKNOWN(), size, pool_.get());
  for (auto i = 0; i < size; i++) {
    flatVector->setNull(i, true);
  }
  testRoundTrip(flatVector);
  testAllNullSerializedAsUnknown(flatVector, BIGINT());
}

TEST_P(PrestoSerializerTest, multiPage) {
  std::ostringstream out;
  std::vector<RowVectorPtr> testVectors;
  // Note: Page of size 1250 is a slight increase in size that initiates string
  // buffer re-use.
  for (int size : {1234, 1250, 538, 2408}) {
    auto vec = makeTestVector(size);
    serialize(vec, &out, nullptr);
    testVectors.push_back(std::move(vec));
  }

  auto bytes = out.str();

  auto rowType = asRowType(testVectors[0]->type());
  auto byteStream = toByteStream(bytes);

  RowVectorPtr deserialized;
  auto paramOptions = getParamSerdeOptions(nullptr);

  for (int i = 0; i < testVectors.size(); i++) {
    RowVectorPtr& vec = testVectors[i];
    serde_->deserialize(
        &byteStream, pool_.get(), rowType, &deserialized, 0, &paramOptions);
    if (i < testVectors.size() - 1) {
      ASSERT_FALSE(byteStream.atEnd());
    } else {
      ASSERT_TRUE(byteStream.atEnd());
    }
    assertEqualVectors(deserialized, vec);
    deserialized->validate({});
  }
}

TEST_P(PrestoSerializerTest, timestampWithNanosecondPrecision) {
  // Verify that nanosecond precision is preserved when the right options are
  // passed to the serde.
  const serializer::presto::PrestoVectorSerde::PrestoOptions
      kUseLosslessTimestampOptions(
          true, common::CompressionKind::CompressionKind_NONE);
  auto timestamp = makeFlatVector<Timestamp>(
      {Timestamp{0, 0},
       Timestamp{12, 0},
       Timestamp{0, 17'123'456},
       Timestamp{1, 17'123'456},
       Timestamp{-1, 17'123'456}});
  testRoundTrip(timestamp, &kUseLosslessTimestampOptions);

  // Verify that precision is lost when no option is passed to the serde.
  auto timestampMillis = makeFlatVector<Timestamp>(
      {Timestamp{0, 0},
       Timestamp{12, 0},
       Timestamp{0, 17'000'000},
       Timestamp{1, 17'000'000},
       Timestamp{-1, 17'000'000}});
  auto inputRowVector = makeRowVector({timestamp});
  auto expectedOutputWithLostPrecision = makeRowVector({timestampMillis});
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
  auto vector = makeFlatVector<int128_t>(decimalValues, DECIMAL(20, 5));

  testRoundTrip(vector);

  // Add some nulls.
  for (auto i = 0; i < 102; i += 7) {
    vector->setNull(i, true);
  }
  testRoundTrip(vector);
}

// Test that hierarchically encoded columns (rows) have their encodings
// preserved by the PrestoBatchVectorSerializer.
TEST_P(PrestoSerializerTest, encodingsBatchVectorSerializer) {
  testBatchVectorSerializerRoundTrip(encodingsTestVector());
}

// Test that array elements have their encodings preserved by the
// PrestoBatchVectorSerializer.
TEST_P(PrestoSerializerTest, encodingsArrayElementsBatchVectorSerializer) {
  testBatchVectorSerializerRoundTrip(encodingsArrayElementsTestVector());
}

// Test that map values have their encodings preserved by the
// PrestoBatchVectorSerializer.
TEST_P(PrestoSerializerTest, encodingsMapValuesBatchVectorSerializer) {
  testBatchVectorSerializerRoundTrip(encodingsMapValuesTestVector());
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

  const size_t numRounds = 100;

  for (size_t i = 0; i < numRounds; ++i) {
    auto rowType = fuzzer.randRowType();
    auto inputRowVector = fuzzer.fuzzInputRow(rowType);
    auto outputRowVector = IOBufToRowVector(
        rowVectorToIOBuf(inputRowVector, *pool_), rowType, *pool_);

    assertEqualVectors(inputRowVector, outputRowVector);
  }
}

TEST_P(PrestoSerializerTest, roundTrip) {
  VectorFuzzer::Options opts;
  opts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
  opts.nullRatio = 0.1;
  VectorFuzzer fuzzer(opts, pool_.get());
  VectorFuzzer::Options nonNullOpts;
  nonNullOpts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
  nonNullOpts.nullRatio = 0;
  VectorFuzzer nonNullFuzzer(nonNullOpts, pool_.get());

  const size_t numRounds = 100;

  for (size_t i = 0; i < numRounds; ++i) {
    auto rowType = fuzzer.randRowType();
    auto inputRowVector = (i % 2 == 0) ? fuzzer.fuzzInputRow(rowType)
                                       : nonNullFuzzer.fuzzInputRow(rowType);
    serializer::presto::PrestoVectorSerde::PrestoOptions prestoOpts;
    // Test every 2/4 with struct nulls first.
    prestoOpts.nullsFirst = i % 4 < 2;
    testRoundTrip(inputRowVector, &prestoOpts);
  }
}

TEST_P(PrestoSerializerTest, encodedRoundtrip) {
  VectorFuzzer::Options opts;
  opts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
  opts.nullRatio = 0.1;
  opts.dictionaryHasNulls = false;
  VectorFuzzer fuzzer(opts, pool_.get());

  const size_t numRounds = 200;

  for (size_t i = 0; i < numRounds; ++i) {
    auto rowType = fuzzer.randRowType();
    auto inputRowVector = fuzzer.fuzzInputRow(rowType);
    serializer::presto::PrestoVectorSerde::PrestoOptions serdeOpts;
    serdeOpts.nullsFirst = true;
    testEncodedRoundTrip(inputRowVector, &serdeOpts);
  }
}

TEST_P(PrestoSerializerTest, encodedConcatenation) {
  // Slow test, run only for no compression.
  if (GetParam() != common::CompressionKind::CompressionKind_NONE) {
    return;
  }

  std::vector<TypePtr> types = {ROW({{"s0", VARCHAR()}}), VARCHAR()};
  VectorFuzzer::Options nonNullOpts{
      .nullRatio = 0, .dictionaryHasNulls = false};
  VectorFuzzer nonNullFuzzer(nonNullOpts, pool_.get());
  VectorFuzzer::Options nullOpts{.nullRatio = 1, .dictionaryHasNulls = false};
  VectorFuzzer nullFuzzer(nullOpts, pool_.get());
  VectorFuzzer::Options mixedOpts{
      .nullRatio = 0.5, .dictionaryHasNulls = false};
  VectorFuzzer mixedFuzzer(mixedOpts, pool_.get());

  for (const auto& type : types) {
    auto flatNull = nullFuzzer.fuzzFlat(type, 12);
    auto flatNonNull = nonNullFuzzer.fuzzFlat(type, 13);

    std::vector<VectorPtr> vectors = {
        nullFuzzer.fuzzConstant(type, 10),
        nonNullFuzzer.fuzzConstant(type, 11),
        flatNonNull,
        flatNull,
        nonNullFuzzer.fuzzDictionary(flatNonNull),
        nonNullFuzzer.fuzzDictionary(flatNull),
        nullFuzzer.fuzzDictionary(flatNull),
    };

    if (type->isRow()) {
      auto& rowType = type->as<TypeKind::ROW>();
      auto row = makeRowVector(
          {rowType.nameOf(0)},
          {nonNullFuzzer.fuzzConstant(rowType.childAt(0), 14)});
      row->setNull(2, true);
      row->setNull(10, true);
      vectors.push_back(row);
    }
    std::vector<std::vector<VectorPtr>> permutations;
    std::vector<VectorPtr> temp;
    makePermutations(vectors, 4, temp, permutations);
    for (auto i = 0; i < permutations.size(); ++i) {
      serializer::presto::PrestoVectorSerde::PrestoOptions opts;
      opts.nullsFirst = true;
      testEncodedConcatenation(permutations[i], &opts);
    }
  }
}

TEST_P(PrestoSerializerTest, emptyArrayOfRowVector) {
  // The value of nullCount_ + nonNullCount_ of the inner RowVector is 0.
  auto arrayOfRow = makeArrayOfRowVector(ROW({UNKNOWN()}), {{}});
  testRoundTrip(arrayOfRow);
}

TEST_P(PrestoSerializerTest, typeMismatch) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<std::string>({"a", "b", "c"}),
  });

  std::ostringstream out;
  serialize(data, &out, nullptr);
  auto serialized = out.str();

  // Too many columns to deserialize.
  VELOX_ASSERT_THROW(
      deserialize(ROW({BIGINT(), VARCHAR(), BOOLEAN()}), serialized, nullptr),
      "Number of columns in serialized data doesn't match "
      "number of columns requested for deserialization");

  // TMore columns in serialization than in type.
  if (GetParam() == common::CompressionKind_NONE) {
    // No throw.
    deserialize(ROW({BIGINT()}), serialized, nullptr);
  } else {
    VELOX_ASSERT_THROW(
        deserialize(ROW({BIGINT()}), serialized, nullptr),
        "Number of columns in serialized data doesn't match "
        "number of columns requested for deserialization");
  }

  // Wrong types of columns.
  VELOX_ASSERT_THROW(
      deserialize(ROW({BIGINT(), DOUBLE()}), serialized, nullptr),
      "Serialized encoding is not compatible with requested type: DOUBLE. "
      "Expected LONG_ARRAY. Got VARIABLE_WIDTH.");
}

TEST_P(PrestoSerializerTest, lexer) {
  VectorFuzzer::Options opts;
  opts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
  opts.nullRatio = 0.1;
  VectorFuzzer fuzzer(opts, pool_.get());
  VectorFuzzer::Options nonNullOpts;
  nonNullOpts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
  nonNullOpts.nullRatio = 0;
  VectorFuzzer nonNullFuzzer(nonNullOpts, pool_.get());

  const size_t numRounds = 20;

  for (size_t i = 0; i < numRounds; ++i) {
    auto rowType = fuzzer.randRowType();

    auto inputRowVector = (i % 2 == 0) ? fuzzer.fuzzInputRow(rowType)
                                       : nonNullFuzzer.fuzzInputRow(rowType);
    testLexer(inputRowVector);
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
        common::CompressionKind::CompressionKind_LZ4,
        common::CompressionKind::CompressionKind_GZIP));

TEST_F(PrestoSerializerTest, deserializeSingleColumn) {
  // Verify that deserializeSingleColumn API can handle all supported types.
  static const size_t kPrestoPageHeaderBytes = 21;
  static const size_t kNumOfColumnsSerializedBytes = sizeof(int32_t);
  static const size_t kBytesToTrim =
      kPrestoPageHeaderBytes + kNumOfColumnsSerializedBytes;

  auto testRoundTripSingleColumn = [&](const VectorPtr& vector) {
    auto rowVector = makeRowVector({vector});
    // Serialize to PrestoPage format.
    std::ostringstream output;
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto rowType = asRowType(rowVector->type());
    auto numRows = rowVector->size();
    auto serializer =
        serde_->createSerializer(rowType, numRows, arena.get(), nullptr);
    serializer->append(rowVector);
    facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
    OStreamOutputStream out(&output, &listener);
    serializer->flush(&out);

    validateLexer(output.str(), {});

    // Remove the PrestoPage header and Number of columns section from the
    // serialized data.
    std::string input = output.str().substr(kBytesToTrim);

    auto byteStream = toByteStream(input);
    VectorPtr deserialized;
    serde_->deserializeSingleColumn(
        &byteStream, pool(), vector->type(), &deserialized, nullptr);
    assertEqualVectors(vector, deserialized);
  };

  std::vector<TypePtr> typesToTest = {
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      TIMESTAMP(),
      ROW({VARCHAR(), INTEGER()}),
      ARRAY(INTEGER()),
      ARRAY(INTEGER()),
      MAP(VARCHAR(), INTEGER()),
      MAP(VARCHAR(), ARRAY(INTEGER())),
  };

  VectorFuzzer::Options opts;
  opts.vectorSize = 5;
  opts.nullRatio = 0.1;
  opts.dictionaryHasNulls = false;
  opts.stringVariableLength = true;
  opts.stringLength = 20;
  opts.containerVariableLength = false;
  opts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
  opts.containerLength = 10;

  auto seed = 0;

  LOG(ERROR) << "Seed: " << seed;
  SCOPED_TRACE(fmt::format("seed: {}", seed));
  VectorFuzzer fuzzer(opts, pool_.get(), seed);

  for (const auto& type : typesToTest) {
    SCOPED_TRACE(fmt::format("Type: {}", type->toString()));
    auto data = fuzzer.fuzz(type);
    testRoundTripSingleColumn(data);
  }
}
