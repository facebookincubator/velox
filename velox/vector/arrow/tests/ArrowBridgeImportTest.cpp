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

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/testing/gtest_util.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/Nulls.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/PageBeforeGuard.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/arrow/Bridge.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {
namespace {

void mockSchemaRelease(ArrowSchema*) {}
void mockArrayRelease(ArrowArray*) {}

template <typename T>
struct VeloxToArrowType {
  using type = T;
};

template <>
struct VeloxToArrowType<Timestamp> {
  using type = int64_t;
};

// Runs each test with both import functions: importFromArrowAsViewer() when
// the parameter is true, importFromArrowAsOwner() when it is false.
class ArrowBridgeImportTest : public testing::TestWithParam<bool>,
                              public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  VectorPtr importFromArrow(
      ArrowSchema& arrowSchema,
      ArrowArray& arrowArray,
      memory::MemoryPool* pool) {
    return isViewer() ? importFromArrowAsViewer(arrowSchema, arrowArray, pool)
                      : importFromArrowAsOwner(arrowSchema, arrowArray, pool);
  }

  bool isViewer() const {
    return GetParam();
  }

  ArrowSchema makeArrowSchema(const char* format) {
    return ArrowSchema{
        .format = format,
        .name = nullptr,
        .metadata = nullptr,
        .flags = 0,
        .n_children = 0,
        .children = nullptr,
        .dictionary = nullptr,
        .release = mockSchemaRelease,
        .private_data = nullptr,
    };
  }

  ArrowArray makeArrowArray(
      const void** buffers,
      int64_t nBuffers,
      int64_t length,
      int64_t nullCount) {
    return ArrowArray{
        .length = length,
        .null_count = nullCount,
        .offset = 0,
        .n_buffers = nBuffers,
        .n_children = 0,
        .buffers = buffers,
        .children = nullptr,
        .dictionary = nullptr,
        .release = mockArrayRelease,
        .private_data = nullptr,
    };
  }

  void exportToArrow(const TypePtr& type, ArrowSchema& out) {
    velox::exportToArrow(
        BaseVector::create(type, 0, pool()), out, ArrowOptions{});
  }

  // Helper structure to hold buffers required by an ArrowArray.
  struct ArrowContextHolder {
    BufferPtr values;
    BufferPtr nulls;
    BufferPtr offsets;

    // Tests might not use the whole array.
    const void* buffers[3] = {nullptr, nullptr, nullptr};
    ArrowArray* children[10];
  };

  template <typename T>
  ArrowArray fillArrowArray(
      const std::vector<std::optional<T>>& inputValues,
      ArrowContextHolder& holder) {
    using TArrow = typename VeloxToArrowType<T>::type;
    int64_t length = inputValues.size();
    int64_t nullCount = 0;

    holder.values = AlignedBuffer::allocate<TArrow>(length, pool());
    holder.nulls = AlignedBuffer::allocate<uint64_t>(length, pool());

    auto rawValues = holder.values->asMutable<TArrow>();
    auto rawNulls = holder.nulls->asMutable<uint64_t>();

    for (size_t i = 0; i < length; ++i) {
      if (inputValues[i] == std::nullopt) {
        bits::setNull(rawNulls, i);
        nullCount++;
      } else {
        bits::clearNull(rawNulls, i);
        if constexpr (std::is_same_v<T, bool>) {
          bits::setBit(rawValues, i, *inputValues[i]);
        } else {
          rawValues[i] = *inputValues[i];
        }
      }
    }

    holder.buffers[0] = (length == 0) ? nullptr : (const void*)rawNulls;
    holder.buffers[1] = (length == 0) ? nullptr : (const void*)rawValues;
    return makeArrowArray(holder.buffers, 2, length, nullCount);
  }

  template <typename TOffsets = int32_t>
  ArrowArray fillArrowArray(
      const std::vector<std::optional<std::string>>& inputValues,
      ArrowContextHolder& holder) {
    int64_t length = inputValues.size();
    int64_t nullCount = 0;

    // Calculate overall buffer size.
    int64_t bufferSize = 0;
    for (const auto& it : inputValues) {
      if (it.has_value()) {
        bufferSize += it->size();
      }
    }

    holder.nulls = AlignedBuffer::allocate<uint64_t>(length, pool());
    holder.offsets = AlignedBuffer::allocate<TOffsets>(length + 1, pool());
    holder.values = AlignedBuffer::allocate<char>(bufferSize, pool());

    auto rawNulls = holder.nulls->asMutable<uint64_t>();
    auto rawOffsets = holder.offsets->asMutable<TOffsets>();
    auto rawValues = holder.values->asMutable<char>();
    *rawOffsets = 0;

    holder.buffers[2] = (length == 0) ? nullptr : (const void*)rawValues;
    holder.buffers[1] = (length == 0) ? nullptr : (const void*)rawOffsets;

    for (size_t i = 0; i < length; ++i) {
      if (inputValues[i] == std::nullopt) {
        bits::setNull(rawNulls, i);
        nullCount++;
        *(rawOffsets + 1) = *rawOffsets;
        ++rawOffsets;
      } else {
        bits::clearNull(rawNulls, i);
        const auto& val = *inputValues[i];

        std::memcpy(rawValues, val.data(), val.size());
        rawValues += val.size();
        *(rawOffsets + 1) = *rawOffsets + val.size();
        ++rawOffsets;
      }
    }

    holder.buffers[0] = (length == 0) ? nullptr : (const void*)rawNulls;
    return makeArrowArray(holder.buffers, 3, length, nullCount);
  }

  bool is64Offsets(const char* format) {
    return format != nullptr && (format[0] == 'U' || format[0] == 'Z');
  }

  // Takes a vector with input data, generates an input ArrowArray and Velox
  // Vector (using vector maker). Then converts ArrowArray into Velox vector and
  // assert that both Velox vectors are semantically the same.
  template <typename TOutput, typename TInput = TOutput>
  void testArrowImport(
      const char* format,
      const std::vector<std::optional<TInput>>& inputValues) {
    ArrowContextHolder holder;
    auto arrowArray = [&] {
      if constexpr (std::is_same_v<TInput, std::string>) {
        if (is64Offsets(format)) {
          return fillArrowArray<int64_t>(inputValues, holder);
        }
      }
      return fillArrowArray(inputValues, holder);
    }();

    // for format U or Z, the offsets buffer is int64_t
    if (is64Offsets(format)) {
      EXPECT_EQ(arrowArray.n_buffers, 3);
      EXPECT_EQ(
          holder.offsets->size(), (inputValues.size() + 1) * sizeof(int64_t));
    }

    auto arrowSchema = makeArrowSchema(format);
    auto output = importFromArrow(arrowSchema, arrowArray, pool());
    if constexpr (
        std::is_same_v<TInput, int64_t> && std::is_same_v<TOutput, Timestamp>) {
      assertTimestampVectorContent(
          inputValues, output, arrowArray.null_count, format);
    } else if constexpr (
        std::is_same_v<TInput, int128_t> && std::is_same_v<TOutput, int64_t>) {
      assertShortDecimalVectorContent(
          inputValues, output, arrowArray.null_count);
    } else if constexpr (
        std::is_same_v<TOutput, int64_t> &&
        (std::is_same_v<TInput, int32_t> || std::is_same_v<TInput, int64_t>)) {
      // TIME: Arrow time32 (int32) or time64 (int64) to Velox TIME (int64
      // millis) Check if format starts with "tt" to distinguish from regular
      // int32/int64.
      if (format[0] == 't' && format[1] == 't') {
        assertTimeVectorContent(
            inputValues, output, arrowArray.null_count, format);
      } else {
        assertVectorContent(inputValues, output, arrowArray.null_count);
      }
    } else {
      assertVectorContent(inputValues, output, arrowArray.null_count);
    }

    // Buffer views are not reusable. Strings might need to create an additional
    // buffer, depending on the string sizes, in which case the buffers could be
    // reusable. So we don't check them in here.
    if constexpr (!std::is_same_v<TInput, std::string>) {
      EXPECT_FALSE(BaseVector::isVectorWritable(output));
    } else {
      size_t totalLength = 0;
      bool isInline = true;
      for (const auto& value : inputValues) {
        if (value.has_value()) {
          auto view = StringView(value.value());
          totalLength += view.size();
          isInline = isInline && view.isInline();
        }
      }
      totalLength = isInline ? 0 : totalLength;
      auto buffers =
          dynamic_cast<const FlatVector<StringView>*>(output->wrappedVector())
              ->stringBuffers();
      size_t realLength = 0;
      for (auto& buffer : buffers) {
        realLength += buffer->capacity();
      }
      EXPECT_EQ(totalLength, realLength);
    }
  }

  template <typename T>
  void assertVectorContent(
      const std::vector<std::optional<T>>& inputValues,
      const VectorPtr& convertedVector,
      size_t nullCount) {
    EXPECT_EQ((nullCount > 0), convertedVector->mayHaveNulls());
    EXPECT_EQ(nullCount, *convertedVector->getNullCount());
    EXPECT_EQ(inputValues.size(), convertedVector->size());

    auto expected = makeNullableFlatVector(inputValues);

    // Assert new vector contents.
    for (vector_size_t i = 0; i < convertedVector->size(); ++i) {
      ASSERT_TRUE(expected->equalValueAt(convertedVector.get(), i, i))
          << "at " << i << ": " << expected->toString(i) << " vs. "
          << convertedVector->toString(i);
    }
  }

  template <typename TOutput, typename TInput>
  void testImportWithoutNullsBuffer(
      std::vector<std::optional<TInput>> inputValues,
      const char* format) {
    auto length = inputValues.size();

    // Construct Arrow array without nulls value and nulls buffer
    ArrowContextHolder holder1;
    holder1.values = AlignedBuffer::allocate<TInput>(length, pool());
    auto rawValues1 = holder1.values->asMutable<TInput>();
    for (size_t i = 0; i < length; ++i) {
      rawValues1[i] = *inputValues[i];
    }

    holder1.buffers[0] = nullptr;
    holder1.buffers[1] = (const void*)rawValues1;
    auto arrowArray1 = makeArrowArray(holder1.buffers, 2, length, 0);
    auto arrowSchema1 = makeArrowSchema(format);

    auto output = importFromArrow(arrowSchema1, arrowArray1, pool());
    if constexpr (
        std::is_same_v<TInput, int64_t> && std::is_same_v<TOutput, Timestamp>) {
      assertTimestampVectorContent(
          inputValues, output, arrowArray1.null_count, format);
    } else if constexpr (
        std::is_same_v<TOutput, int64_t> &&
        (std::is_same_v<TInput, int32_t> || std::is_same_v<TInput, int64_t>)) {
      // TIME: Arrow time32 (int32) or time64 (int64) to Velox TIME (int64
      // millis).
      if (format[0] == 't' && format[1] == 't') {
        assertTimeVectorContent(
            inputValues, output, arrowArray1.null_count, format);
      } else {
        assertVectorContent(inputValues, output, arrowArray1.null_count);
      }
    } else {
      assertVectorContent(inputValues, output, arrowArray1.null_count);
    }

    // However, convert from an Arrow array without nulls buffer but non-zero
    // null count should fail
    ArrowContextHolder holder2;
    holder2.values = AlignedBuffer::allocate<TInput>(length, pool());
    auto rawValues2 = holder2.values->asMutable<TInput>();
    for (size_t i = 0; i < length; ++i) {
      rawValues2[i] = *inputValues[i];
    }

    holder2.buffers[0] = nullptr;
    holder2.buffers[1] = (const void*)rawValues2;
    auto arrowSchema2 = makeArrowSchema(format);
    auto arrowArray2 = makeArrowArray(holder2.buffers, 2, length, 1);
    EXPECT_THROW(
        importFromArrow(arrowSchema2, arrowArray2, pool()), VeloxUserError);
  }

  void testImportNullType(int64_t nullCount) {
    ArrowSchema arrowSchema = makeArrowSchema("n");
    ArrowArray arrowArray = makeArrowArray(nullptr, 0, 4, nullCount);

    auto output = importFromArrow(arrowSchema, arrowArray, pool());

    EXPECT_TRUE(output->type()->kindEquals(UNKNOWN()));
    EXPECT_EQ(4, output->size());
    ASSERT_TRUE(output->mayHaveNulls());
    EXPECT_EQ(4, *output->getNullCount());
    for (vector_size_t i = 0; i < output->size(); ++i) {
      EXPECT_TRUE(output->isNullAt(i));
    }
  }

  // Creates short decimals from int128 and asserts the content of actual vector
  // with the expected values.
  void assertShortDecimalVectorContent(
      const std::vector<std::optional<int128_t>>& expectedValues,
      const VectorPtr& actual,
      size_t nullCount) {
    std::vector<std::optional<int64_t>> decValues;
    decValues.reserve(expectedValues.size());
    for (const auto& value : expectedValues) {
      if (value) {
        decValues.emplace_back(static_cast<int64_t>(*value));
      } else {
        decValues.emplace_back(std::nullopt);
      }
    }
    assertVectorContent(decValues, actual, nullCount);
  }

  // Creates TIME from int32/int64 and asserts the content of actual vector with
  // the expected TIME values (in milliseconds).
  template <typename TInput>
  void assertTimeVectorContent(
      const std::vector<std::optional<TInput>>& expectedValues,
      const VectorPtr& actual,
      size_t nullCount,
      const char* format) {
    VELOX_USER_CHECK_GE(
        strlen(format), 3, "At least three characters are expected.");
    std::vector<std::optional<int64_t>> timeValues;
    timeValues.reserve(expectedValues.size());
    for (const auto& value : expectedValues) {
      if (!value.has_value()) {
        timeValues.emplace_back(std::nullopt);
      } else {
        int64_t millis;
        // Convert from Arrow time unit to Velox TIME (milliseconds).
        switch (format[2]) {
          case 's':
            millis = static_cast<int64_t>(value.value()) * 1'000;
            break;
          case 'm':
            millis = static_cast<int64_t>(value.value());
            break;
          case 'u':
            millis = static_cast<int64_t>(value.value()) / 1'000;
            break;
          case 'n':
            millis = static_cast<int64_t>(value.value()) / 1'000'000;
            break;
          default:
            VELOX_UNREACHABLE();
        }
        timeValues.emplace_back(millis);
      }
    }
    auto expected = makeNullableFlatVector(timeValues, TIME());
    assertVectorContent(timeValues, actual, nullCount);
  }

  // Creates timestamp from bigint and asserts the content of actual vector with
  // the expected timestamp values.
  void assertTimestampVectorContent(
      const std::vector<std::optional<int64_t>>& expectedValues,
      const VectorPtr& actual,
      size_t nullCount,
      const char* format) {
    VELOX_USER_CHECK_GE(
        strlen(format), 3, "At least three characters are expected.");
    std::vector<std::optional<Timestamp>> tsValues;
    tsValues.reserve(expectedValues.size());
    for (const auto& value : expectedValues) {
      if (!value.has_value()) {
        tsValues.emplace_back(std::nullopt);
      } else {
        Timestamp ts;
        switch (format[2]) {
          case 's':
            ts = Timestamp(value.value(), 0);
            break;
          case 'm':
            ts = Timestamp::fromMillis(value.value());
            break;
          case 'u':
            ts = Timestamp::fromMicros(value.value());
            break;
          case 'n':
            ts = Timestamp::fromNanos(value.value());
            break;
          default:
            VELOX_UNREACHABLE();
        }
        tsValues.emplace_back(ts);
      }
    }
    assertVectorContent(tsValues, actual, nullCount);
  }

  void toVeloxVector(const arrow::Array& array, VectorPtr& outVector) {
    ArrowSchema schema;
    ArrowArray data;
    ASSERT_OK(arrow::ExportType(*array.type(), &schema));
    ASSERT_OK(arrow::ExportArray(array, &data));
    outVector = importFromArrow(schema, data, pool());
    if (isViewer()) {
      // These release calls just decrease the refcount; there are still
      // references to keep the data alive from the original Arrow array.
      schema.release(&schema);
      data.release(&data);
    } else {
      EXPECT_FALSE(schema.release);
      EXPECT_FALSE(data.release);
    }
  }

  template <typename F>
  void testArrowRoundTrip(
      const arrow::Array& array,
      F validateVector,
      const ArrowOptions& options = ArrowOptions{}) {
    VectorPtr vec;
    toVeloxVector(array, vec);
    validateVector(*vec);

    ArrowSchema schema;
    ArrowArray data;
    velox::exportToArrow(vec, schema, options);
    velox::exportToArrow(vec, data, pool(), options);
    ASSERT_OK_AND_ASSIGN(auto arrowType, arrow::ImportType(&schema));
    ASSERT_OK_AND_ASSIGN(auto array2, arrow::ImportArray(&data, arrowType));
    ASSERT_OK(array2->ValidateFull());
    EXPECT_TRUE(array2->Equals(array));
  }

  void validateImportREESingleRun(
      const arrow::Array& array,
      const TypePtr& expectedType,
      size_t expectedSize) {
    testArrowRoundTrip(array, [&](const BaseVector& vec) {
      ASSERT_EQ(*vec.type(), *expectedType);
      EXPECT_EQ(vec.encoding(), VectorEncoding::Simple::CONSTANT);
      EXPECT_EQ(vec.size(), expectedSize);
    });
  }

  void assertRoundTrip(const VectorPtr& vector) {
    ArrowSchema schema;
    ArrowArray data;
    velox::exportToArrow(vector, schema, ArrowOptions{});
    velox::exportToArrow(vector, data, pool(), ArrowOptions{});

    auto output = importFromArrow(schema, data, pool());
    assertEqualVectors(vector, output);

    if (isViewer()) {
      schema.release(&schema);
      data.release(&data);
    } else {
      EXPECT_EQ(nullptr, schema.release);
      EXPECT_EQ(nullptr, data.release);
    }
  }
};

TEST_P(ArrowBridgeImportTest, scalar) {
  testArrowImport<bool>("b", {});
  testArrowImport<bool>("b", {true});
  testArrowImport<bool>("b", {false});
  testArrowImport<bool>("b", {true, false, true});
  testArrowImport<bool>("b", {true, std::nullopt, true});

  testArrowImport<int8_t>("c", {});
  testArrowImport<int8_t>("c", {101});
  testArrowImport<int8_t>("c", {5, 4, 3, 1, 2});
  testArrowImport<int8_t>("c", {8, std::nullopt, std::nullopt});
  testArrowImport<int8_t>("c", {std::nullopt, std::nullopt});

  testArrowImport<int16_t>("s", {5, 4, 3, 1, 2});
  testArrowImport<int32_t>("i", {5, 4, 3, 1, 2});

  testArrowImport<int32_t>("tdD", {5, 4, 3, 1, 2});

  testArrowImport<int64_t>("l", {});
  testArrowImport<int64_t>("l", {std::nullopt});
  testArrowImport<int64_t>("l", {-99, 4, 318321631, 1211, -12});
  testArrowImport<int64_t>("l", {std::nullopt, 12345678, std::nullopt});
  testArrowImport<int64_t>("l", {std::nullopt, std::nullopt});

  testArrowImport<double>("g", {});
  testArrowImport<double>("g", {std::nullopt});
  testArrowImport<double>("g", {-99.9, 4.3, 31.1, 129.11, -12});
  testArrowImport<float>("f", {-99.9, 4.3, 31.1, 129.11, -12});

  for (const auto tsString : {"tss:", "tsm:", "tsu:", "tsn:"}) {
    testArrowImport<Timestamp, int64_t>(
        tsString, {0, std::nullopt, Timestamp::kMaxSeconds});
  }

  testArrowImport<int64_t, int32_t>(
      "tts", {0, std::nullopt, 1, 60, 3600, 50402, 86399});
  testArrowImport<int64_t, int32_t>(
      "ttm", {0, std::nullopt, 1000, 60000, 3600000, 50402000, 86399999});
  testArrowImport<int64_t, int64_t>(
      "ttu",
      {0,
       std::nullopt,
       1'000'000,
       60'000'000,
       3'600'000'000,
       50'402'000'000,
       86'399'999'000});
  testArrowImport<int64_t, int64_t>(
      "ttn",
      {0,
       std::nullopt,
       1'000'000'000,
       60'000'000'000,
       3'600'000'000'000,
       50'402'000'000'000,
       86'399'999'000'000});

  testArrowImport<int64_t, int128_t>(
      "d:5,2", {1, -1, 0, 12345, -12345, std::nullopt});
  testArrowImport<int128_t, int128_t>(
      "d:36,2",
      {HugeInt::parse("20000000000000000"),
       HugeInt::parse("50000000000000000"),
       0,
       HugeInt::parse("50000000000000000000"),
       HugeInt::parse("-40000000000000000000"),
       std::nullopt});
}

TEST_P(ArrowBridgeImportTest, decimal32) {
  const auto type = DECIMAL(9, 2);
  const std::vector<std::vector<std::optional<int32_t>>> inputs = {
      {},
      {0, 1, -1, 999'999'999, -999'999'999},
      {12'345, std::nullopt, -12'345},
      {std::nullopt, std::nullopt}};
  for (const auto& input : inputs) {
    ArrowContextHolder holder;
    auto array = fillArrowArray(input, holder);
    if (array.null_count == array.length) {
      holder.buffers[1] = nullptr;
    }
    auto schema = makeArrowSchema("d:9,2,32");
    auto actual = importFromArrow(schema, array, pool());
    std::vector<std::optional<int64_t>> expectedValues(
        input.begin(), input.end());
    auto expected = makeNullableFlatVector(expectedValues, type);
    facebook::velox::test::assertEqualVectors(expected, actual);
    const auto* rawValues = actual->asFlatVector<int64_t>()->rawValues();
    for (vector_size_t i = 0; i < actual->size(); ++i) {
      if (!input[i].has_value()) {
        EXPECT_EQ(rawValues[i], 0);
      }
    }
  }
}

TEST_P(ArrowBridgeImportTest, timestampUtc) {
  auto flatVector = makeNullableFlatVector<Timestamp>(
      {
          Timestamp(0, 0),
          std::nullopt,
          Timestamp(1699300965, 12'349),
          Timestamp(-2208960000, 0),
          Timestamp(3155788800, 999'999'999),
      },
      TIMESTAMP_UTC());
  assertRoundTrip(flatVector);

  auto dictionaryValues = makeFlatVector<Timestamp>(
      {
          Timestamp(0, 0),
          Timestamp(1699300965, 12'349),
          Timestamp(3155788800, 999'999'999),
      },
      TIMESTAMP_UTC());
  auto dictionaryVector = BaseVector::wrapInDictionary(
      nullptr, makeIndicesInReverse(3), 3, dictionaryValues);
  assertRoundTrip(dictionaryVector);

  using NullableTimestampArray =
      std::optional<std::vector<std::optional<Timestamp>>>;
  std::vector<NullableTimestampArray> nestedValues = {
      std::vector<std::optional<Timestamp>>{
          Timestamp(0, 0),
          std::nullopt,
          Timestamp(2, 200),
      },
      std::nullopt,
      std::vector<std::optional<Timestamp>>{Timestamp(-2208960000, 0)},
      std::vector<std::optional<Timestamp>>{},
  };
  auto nestedVector =
      makeNullableArrayVector<Timestamp>(nestedValues, ARRAY(TIMESTAMP_UTC()));
  assertRoundTrip(nestedVector);
}

TEST_P(ArrowBridgeImportTest, withoutNullsBuffer) {
  std::vector<std::optional<int64_t>> inputValues = {1, 2, 3, 4, 5};
  testImportWithoutNullsBuffer<int64_t>(inputValues, "l");
  testImportWithoutNullsBuffer<Timestamp>(inputValues, "tsn:");
}

TEST_P(ArrowBridgeImportTest, string) {
  testArrowImport<std::string>("u", {});
  testArrowImport<std::string>("u", {"single"});
  testArrowImport<std::string>(
      "u",
      {
          "hello world",
          "larger string which should not be inlined...",
          std::nullopt,
          "hello",
          "from",
          "the",
          "other",
          "side",
          std::nullopt,
          std::nullopt,
      });

  testArrowImport<std::string>(
      "z",
      {
          std::nullopt,
          "testing",
          "a",
          std::nullopt,
          "varbinary",
          "vector",
          std::nullopt,
      });
}

TEST_P(ArrowBridgeImportTest, string64) {
  testArrowImport<std::string>("U", {});
  testArrowImport<std::string>("U", {"single"});
  testArrowImport<std::string>(
      "U",
      {
          "hello world",
          "larger string which should not be inlined...",
          std::nullopt,
          "hello",
          "from",
          "the",
          "other",
          "side",
          std::nullopt,
          std::nullopt,
      });

  testArrowImport<std::string>(
      "Z",
      {
          std::nullopt,
          "testing",
          "a",
          std::nullopt,
          "varbinary",
          "vector",
          std::nullopt,
      });
}

TEST_P(ArrowBridgeImportTest, stringview) {
  arrow::StringViewBuilder sb(arrow::default_memory_pool());
  ASSERT_OK(sb.Append("hello world", 11));
  ASSERT_OK(sb.Append("larger string which should not be inlined...", 44));
  ASSERT_OK(sb.AppendNull());
  ASSERT_OK(sb.Append("hello", 5));
  ASSERT_OK(sb.Append("", 0));
  ASSERT_OK(sb.Append("my string", 9));
  ASSERT_OK(sb.Append("another slightly longer string", 30));
  ASSERT_OK(sb.AppendNull());
  ASSERT_OK(sb.Append("a", 1));
  ASSERT_OK(sb.Append(
      "another even longer string to ensure it's for sure not stored inline!!!",
      71));
  ASSERT_OK(sb.AppendNull());
  ASSERT_OK(sb.AppendNull());
  ASSERT_OK_AND_ASSIGN(auto array, sb.Finish());
  const void* stringBuffers = array->data()->buffers[2]->data();

  testArrowRoundTrip(
      *array,
      [stringBuffers](const BaseVector& vec) {
        // assert zero-copy import to velox for stringBuffers
        ASSERT_EQ(
            vec.asFlatVector<StringView>()
                ->stringBuffers()
                .data()
                ->get()
                ->as<void*>(),
            stringBuffers);
        ASSERT_EQ(*vec.type(), *VARCHAR());
        EXPECT_EQ(vec.size(), 12);
      },
      ArrowOptions{.varTypeLayout = ArrowOptions::VarTypeLayout::kStringView});
}

// Widths 4 and 16 cover inlined and buffer-backed StringViews.
TEST_P(ArrowBridgeImportTest, fixedSizeBinary) {
  for (const int32_t width : {4, 16}) {
    const std::vector<std::optional<std::string>> values = {
        std::string(width, 'a'), std::nullopt, std::string(width, 'b')};
    arrow::FixedSizeBinaryBuilder builder(arrow::fixed_size_binary(width));
    for (const auto& value : values) {
      ASSERT_OK(value ? builder.Append(*value) : builder.AppendNull());
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));

    VectorPtr vector;
    toVeloxVector(*array, vector);
    test::assertEqualVectors(
        makeNullableFlatVector(values, VARBINARY()), vector);
  }
}

TEST_P(ArrowBridgeImportTest, nullType) {
  testImportNullType(4);
  testImportNullType(-1);
}

TEST_P(ArrowBridgeImportTest, rowFull) {
  // Manually create a ROW type.
  ArrowSchema arrowSchema;
  exportToArrow(
      ROW({"col1", "col2", "col3"}, {BIGINT(), DOUBLE(), VARCHAR()}),
      arrowSchema);

  // Some test data.
  std::vector<std::optional<int64_t>> col1 = {1, 2, 3, 4};
  std::vector<std::optional<double>> col2 = {99.9, 88.8, 77.7, std::nullopt};
  std::vector<std::optional<std::string>> col3 = {
      "my", "string", "column", "Longer string so it's not inlined."};

  // Create the 3 children arrays.
  ArrowContextHolder childHolder1;
  ArrowContextHolder childHolder2;
  ArrowContextHolder childHolder3;

  auto childArray1 = fillArrowArray(col1, childHolder1);
  auto childArray2 = fillArrowArray(col2, childHolder2);
  auto childArray3 = fillArrowArray(col3, childHolder3);

  // Create parent array and set the child pointers.
  ArrowContextHolder parentHolder;
  auto arrowArray = makeArrowArray(parentHolder.buffers, 0, col1.size(), 0);
  arrowArray.buffers[0] = nullptr;

  arrowArray.n_children = 3;
  arrowArray.children = parentHolder.children;
  arrowArray.children[0] = &childArray1;
  arrowArray.children[1] = &childArray2;
  arrowArray.children[2] = &childArray3;

  // Import and validate output.
  auto outputVector = importFromArrow(arrowSchema, arrowArray, pool());
  auto rowVector = std::dynamic_pointer_cast<RowVector>(outputVector);

  EXPECT_TRUE(rowVector != nullptr);
  EXPECT_EQ(arrowArray.n_children, rowVector->childrenSize());
  EXPECT_EQ(arrowArray.length, rowVector->size());

  assertVectorContent(col1, rowVector->childAt(0), childArray1.null_count);
  assertVectorContent(col2, rowVector->childAt(1), childArray2.null_count);
  assertVectorContent(col3, rowVector->childAt(2), childArray3.null_count);

  if (isViewer()) {
    arrowArray.release(&arrowArray);
    arrowSchema.release(&arrowSchema);
  } else {
    EXPECT_EQ(arrowArray.release, nullptr);
    EXPECT_EQ(arrowSchema.release, nullptr);
  }
}

TEST_P(ArrowBridgeImportTest, rowEmpty) {
  // Manually create a ROW type.
  ArrowSchema arrowSchema;
  exportToArrow(ROW({}), arrowSchema);

  // Create parent array and set the child pointers.
  ArrowContextHolder parentHolder;
  auto arrowArray = makeArrowArray(parentHolder.buffers, 0, 0, 0);

  // Import and validate output.
  auto outputVector = importFromArrow(arrowSchema, arrowArray, pool());
  auto rowVector = std::dynamic_pointer_cast<RowVector>(outputVector);

  EXPECT_TRUE(rowVector != nullptr);
  EXPECT_EQ(0, rowVector->childrenSize());
  EXPECT_EQ(0, rowVector->size());

  if (isViewer()) {
    arrowArray.release(&arrowArray);
    arrowSchema.release(&arrowSchema);
  } else {
    EXPECT_EQ(arrowArray.release, nullptr);
    EXPECT_EQ(arrowSchema.release, nullptr);
  }
}

TEST_P(ArrowBridgeImportTest, array) {
  auto vb = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder lb(arrow::default_memory_pool(), vb);
  ASSERT_OK(lb.Append());
  ASSERT_OK(vb->Append(1));
  ASSERT_OK(lb.AppendNull());
  ASSERT_OK(lb.Append());
  ASSERT_OK(vb->Append(2));
  ASSERT_OK(vb->Append(3));
  ASSERT_OK(lb.AppendEmptyValue());
  ASSERT_OK_AND_ASSIGN(auto array, lb.Finish());
  testArrowRoundTrip(*array, [](const BaseVector& vec) {
    ASSERT_EQ(*vec.type(), *ARRAY(INTEGER()));
    EXPECT_EQ(vec.size(), 4);
  });
}

TEST_P(ArrowBridgeImportTest, map) {
  auto kb = std::make_shared<arrow::Int32Builder>(); // key builder
  auto ib = std::make_shared<arrow::Int32Builder>(); // item builder
  arrow::MapBuilder mb(arrow::default_memory_pool(), kb, ib);
  ASSERT_OK(mb.Append());
  ASSERT_OK(kb->Append(1));
  ASSERT_OK(ib->Append(2));
  ASSERT_OK(mb.AppendNull());
  ASSERT_OK(mb.Append());
  ASSERT_OK(kb->Append(2));
  ASSERT_OK(ib->Append(4));
  ASSERT_OK(kb->Append(3));
  ASSERT_OK(ib->Append(6));
  ASSERT_OK(mb.AppendEmptyValue());
  ASSERT_OK_AND_ASSIGN(auto array, mb.Finish());
  testArrowRoundTrip(*array, [](const BaseVector& vec) {
    ASSERT_EQ(*vec.type(), *MAP(INTEGER(), INTEGER()));
    EXPECT_EQ(vec.size(), 4);
  });
}

// Imports BOOLEAN values that end right before an unmapped page. The values
// buffer covers the bitmap, one bit per row, so copying it reads only the
// bitmap.
TEST_P(ArrowBridgeImportTest, booleanValuesSize) {
  constexpr int64_t kLength = 4'096;
  common::testutil::PageBeforeGuard page;
  auto bitmap = page.atEnd<uint8_t>(bits::nbytes(kLength));
  for (int64_t i = 0; i < kLength; ++i) {
    bits::setBit(bitmap.data(), i, i % 3 == 0);
  }

  const void* buffers[2] = {nullptr, bitmap.data()};
  auto arrowArray = makeArrowArray(buffers, 2, kLength, 0);
  auto arrowSchema = makeArrowSchema("b");
  auto output = importFromArrow(arrowSchema, arrowArray, pool_.get());

  auto* flat = output->asFlatVector<bool>();
  EXPECT_EQ(flat->values()->size(), bits::nbytes(kLength));
  auto copy = AlignedBuffer::copy(pool_.get(), flat->values());
  for (int64_t i = 0; i < kLength; ++i) {
    EXPECT_EQ(bits::isBitSet(copy->as<uint64_t>(), i), i % 3 == 0) << i;
  }
}

TEST_P(ArrowBridgeImportTest, dictionary) {
  arrow::Dictionary32Builder<arrow::Int64Type> b;
  for (int i = 0; i < 60; ++i) {
    if (i % 7 == 0) {
      ASSERT_OK(b.AppendNull());
    } else {
      ASSERT_OK(b.Append(i % 11));
    }
  }
  ASSERT_OK_AND_ASSIGN(auto array, b.Finish());
  testArrowRoundTrip(*array, [](const BaseVector& vec) {
    ASSERT_EQ(*vec.type(), *BIGINT());
    EXPECT_EQ(vec.encoding(), VectorEncoding::Simple::DICTIONARY);
    EXPECT_EQ(vec.size(), 60);
  });
}

TEST_P(ArrowBridgeImportTest, nestedDictionary) {
  // Export writes a dictionary over a dictionary as an Arrow dictionary whose
  // values are dictionary encoded too. Structs, arrays and maps take the
  // types of their children from the schema.
  auto dictionaryOverDictionary = wrapInDictionary(
      makeIndices({2, 0, 1}),
      wrapInDictionary(
          makeIndices({1, 0, 1}), makeFlatVector<int64_t>({10, 20})));

  // Struct child.
  assertRoundTrip(makeRowVector({dictionaryOverDictionary}));

  // Array elements.
  assertRoundTrip(makeArrayVector({0, 1}, dictionaryOverDictionary));

  // Map values.
  assertRoundTrip(makeMapVector(
      {0, 1}, makeFlatVector<int64_t>({1, 2, 3}), dictionaryOverDictionary));

  // Three levels of dictionaries.
  assertRoundTrip(makeRowVector(
      {wrapInDictionary(makeIndices({1, 2, 0}), dictionaryOverDictionary)}));
}

TEST_P(ArrowBridgeImportTest, dictionaryEncodedNullType) {
  const int32_t indices[] = {0, 0, 0, 0};
  const void* buffers[] = {nullptr, indices};

  ArrowSchema arrowSchema = makeArrowSchema("i");
  ArrowSchema dictionarySchema = makeArrowSchema("n");
  arrowSchema.dictionary = &dictionarySchema;

  ArrowArray arrowArray = makeArrowArray(buffers, 2, 4, 0);
  ArrowArray dictionaryArray = makeArrowArray(nullptr, 0, 1, 1);
  arrowArray.dictionary = &dictionaryArray;

  auto output = importFromArrow(arrowSchema, arrowArray, pool());

  ASSERT_EQ(*UNKNOWN(), *output->type());
  EXPECT_EQ(4, output->size());

  DecodedVector decoded(*output);
  EXPECT_TRUE(decoded.mayHaveNulls());
  for (vector_size_t i = 0; i < output->size(); ++i) {
    EXPECT_TRUE(decoded.isNullAt(i));
  }
}

TEST_P(ArrowBridgeImportTest, reeNoRuns) {
  auto pool = arrow::default_memory_pool();

  arrow::RunEndEncodedBuilder reeInt32(
      pool,
      std::make_shared<arrow::Int32Builder>(pool),
      std::make_shared<arrow::Int32Builder>(pool),
      run_end_encoded(arrow::int32(), arrow::int32()));
  ASSERT_OK_AND_ASSIGN(auto array, reeInt32.Finish());

  VectorPtr vector;
  toVeloxVector(*array, vector);

  ASSERT_EQ(*vector->type(), *INTEGER());
  EXPECT_EQ(vector->encoding(), VectorEncoding::Simple::CONSTANT);
  EXPECT_EQ(vector->size(), 0);
}

TEST_P(ArrowBridgeImportTest, reeSingleRun) {
  auto pool = arrow::default_memory_pool();

  // Simple integer column.
  arrow::RunEndEncodedBuilder reeInt32(
      pool,
      std::make_shared<arrow::Int32Builder>(pool),
      std::make_shared<arrow::Int32Builder>(pool),
      run_end_encoded(arrow::int32(), arrow::int32()));
  ASSERT_OK(reeInt32.AppendScalar(*arrow::MakeScalar<int32_t>(123), 20));
  ASSERT_OK_AND_ASSIGN(auto array, reeInt32.Finish());

  validateImportREESingleRun(*array, INTEGER(), 20);

  // String column.
  arrow::RunEndEncodedBuilder reeString(
      pool,
      std::make_shared<arrow::Int32Builder>(pool),
      std::make_shared<arrow::StringBuilder>(pool),
      run_end_encoded(arrow::int32(), arrow::utf8()));
  ASSERT_OK(reeString.AppendScalar(*arrow::MakeScalar("bla"), 199));
  ASSERT_OK_AND_ASSIGN(array, reeString.Finish());

  validateImportREESingleRun(*array, VARCHAR(), 199);

  // Array/List.
  auto valuesBuilder = std::make_shared<arrow::FloatBuilder>(pool);
  auto listBuilder = std::make_shared<arrow::ListBuilder>(pool, valuesBuilder);
  ASSERT_OK(listBuilder->Append());
  ASSERT_OK(valuesBuilder->Append(1.1));
  ASSERT_OK(valuesBuilder->Append(2.2));
  ASSERT_OK(valuesBuilder->Append(3.3));
  ASSERT_OK(valuesBuilder->Append(4.4));
  ASSERT_OK_AND_ASSIGN(auto listArray, listBuilder->Finish());

  auto runEndBuilder = std::make_shared<arrow::Int32Builder>(pool);
  ASSERT_OK(runEndBuilder->Append(123));
  ASSERT_OK_AND_ASSIGN(auto runEndsArray, runEndBuilder->Finish());

  // Create a list containing [1.1, 2.2, 3.3, 4.4] and repeat it 123 times.
  ASSERT_OK_AND_ASSIGN(
      array, arrow::RunEndEncodedArray::Make(123, runEndsArray, listArray));
  validateImportREESingleRun(*array, ARRAY(REAL()), 123);
}

TEST_P(ArrowBridgeImportTest, reeMultipleRuns) {
  auto pool = arrow::default_memory_pool();

  // Simple integer column.
  arrow::RunEndEncodedBuilder reeInt32(
      pool,
      std::make_shared<arrow::Int32Builder>(pool),
      std::make_shared<arrow::Int32Builder>(pool),
      run_end_encoded(arrow::int32(), arrow::int32()));
  ASSERT_OK(reeInt32.AppendScalar(*arrow::MakeScalar<int32_t>(123), 20));
  ASSERT_OK(reeInt32.AppendScalar(*arrow::MakeScalar<int32_t>(321), 2));
  ASSERT_OK(reeInt32.AppendNulls(10));
  ASSERT_OK(reeInt32.AppendScalar(*arrow::MakeScalar<int32_t>(50), 30));
  ASSERT_OK_AND_ASSIGN(auto array, reeInt32.Finish());

  VectorPtr vector;
  toVeloxVector(*array, vector);

  ASSERT_EQ(*vector->type(), *INTEGER());
  EXPECT_EQ(vector->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(vector->size(), 62);

  DecodedVector decoded(*vector);
  EXPECT_TRUE(decoded.mayHaveNulls());
  EXPECT_FALSE(decoded.isNullAt(0));
  EXPECT_EQ(decoded.valueAt<int32_t>(0), 123);
  EXPECT_EQ(decoded.valueAt<int32_t>(1), 123);
  EXPECT_EQ(decoded.valueAt<int32_t>(19), 123);
  EXPECT_EQ(decoded.valueAt<int32_t>(20), 321);
  EXPECT_EQ(decoded.valueAt<int32_t>(21), 321);
  EXPECT_TRUE(decoded.isNullAt(22));
  EXPECT_TRUE(decoded.isNullAt(31));
  EXPECT_EQ(decoded.valueAt<int32_t>(32), 50);
  EXPECT_EQ(decoded.valueAt<int32_t>(33), 50);
  EXPECT_EQ(decoded.valueAt<int32_t>(61), 50);
}

TEST_P(ArrowBridgeImportTest, failures) {
  ArrowSchema arrowSchema;
  ArrowArray arrowArray;

  const int32_t values[] = {1, 2, 3, 4};
  const void* buffers[] = {nullptr, values};

  // Unsupported:

  // Offset not yet supported.
  arrowSchema = makeArrowSchema("i");
  arrowArray = makeArrowArray(buffers, 2, 4, 0);
  arrowArray.offset = 1;
  EXPECT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()), VeloxUserError);

  // Broken input.

  // Null release callback indicates a released structure and should be
  // error-ed out
  arrowSchema = makeArrowSchema("i");
  arrowArray = makeArrowArray(buffers, 2, 4, 0);
  arrowSchema.release = nullptr;
  EXPECT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()), VeloxUserError);

  arrowSchema = makeArrowSchema("i");
  arrowArray = makeArrowArray(buffers, 2, 4, 0);
  arrowArray.release = nullptr;
  EXPECT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()), VeloxUserError);

  // Expect two buffers.
  arrowSchema = makeArrowSchema("i");
  arrowArray = makeArrowArray(buffers, 2, 4, 0);
  arrowArray.n_buffers = 1;
  EXPECT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()), VeloxUserError);

  // Can't have nulls without null buffer.
  arrowSchema = makeArrowSchema("i");
  arrowArray = makeArrowArray(buffers, 2, 4, 1);
  arrowArray.null_count = 1;
  EXPECT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()), VeloxUserError);

  // Non-existing type.
  arrowSchema = makeArrowSchema("a");
  arrowArray = makeArrowArray(buffers, 2, 4, 0);
  EXPECT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()), VeloxUserError);

  // Ensure the baseline works.
  arrowSchema = makeArrowSchema("i");
  arrowArray = makeArrowArray(buffers, 2, 4, 0);
  EXPECT_NO_THROW(importFromArrow(arrowSchema, arrowArray, pool()));
}

// Verify that importing a Utf8View array with an out-of-bounds buffer index
// in a non-inline string view throws instead of reading past the buffers
// array.
TEST_P(ArrowBridgeImportTest, utf8ViewOobBufferIndex) {
  // Arrow Utf8View layout (format "vu"):
  //   buffer[0] = nulls
  //   buffer[1] = views (16 bytes each)
  //   buffer[2 .. n_buffers-2] = data buffers
  //   buffer[n_buffers-1] = buffer sizes (uint64_t per data buffer)
  //
  // We set up one data buffer (index 2) and one sizes buffer (index 3), so
  // n_buffers = 4 and the only valid data-buffer index is 0.  The crafted
  // view references buffer index 99 — well out of range.

  const char dataBuffer[] = "some data string";
  const uint64_t bufferSizesArr[] = {sizeof(dataBuffer)};

  // Arrow Utf8View: [4B length][4B prefix][4B buffer_index][4B buffer_offset]
  struct Utf8ViewEntry {
    uint32_t length;
    char prefix[4];
    uint32_t bufferIndex;
    uint32_t bufferOffset;
  };
  Utf8ViewEntry malformedView{};
  malformedView.length = 16; // > 12, so non-inline.
  std::memcpy(malformedView.prefix, "some", 4);
  malformedView.bufferIndex = 99; // Out of bounds.
  malformedView.bufferOffset = 0;

  const void* buffers[] = {
      nullptr, // nulls
      &malformedView, // views
      dataBuffer, // data buffer 0
      bufferSizesArr, // buffer sizes
  };

  auto arrowSchema = makeArrowSchema("vu");
  auto arrowArray = makeArrowArray(buffers, 4, 1, 0);

  VELOX_ASSERT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()),
      "Arrow Utf8View buffer index out of range");
}

TEST_P(ArrowBridgeImportTest, inputsMarkedReleased) {
  const int32_t values[] = {1, 2, 3, 4};
  const void* buffers[] = {nullptr, values};

  ArrowSchema arrowSchema = makeArrowSchema("i");
  ArrowArray arrowArray = makeArrowArray(buffers, 2, 4, 0);

  auto _ = importFromArrow(arrowSchema, arrowArray, pool());

  // The owner marks the inputs released. The viewer leaves them alone.
  if (isViewer()) {
    EXPECT_EQ(arrowSchema.release, mockSchemaRelease);
    EXPECT_EQ(arrowArray.release, mockArrayRelease);
  } else {
    EXPECT_EQ(arrowSchema.release, nullptr);
    EXPECT_EQ(arrowArray.release, nullptr);
  }
}

// Counts the releases of a structure in the int its private data points to.
// Like a producer's release callback, releases the children that are not yet
// released first.
void countSchemaRelease(ArrowSchema* schema) {
  for (int64_t i = 0; i < schema->n_children; ++i) {
    if (schema->children[i]->release != nullptr) {
      schema->children[i]->release(schema->children[i]);
    }
  }
  ++*static_cast<int*>(schema->private_data);
  schema->release = nullptr;
}

void countArrayRelease(ArrowArray* array) {
  for (int64_t i = 0; i < array->n_children; ++i) {
    if (array->children[i]->release != nullptr) {
      array->children[i]->release(array->children[i]);
    }
  }
  ++*static_cast<int*>(array->private_data);
  array->release = nullptr;
}

TEST_P(ArrowBridgeImportTest, releaseCalled) {
  const int32_t values[] = {1, 2, 3, 4};
  const void* buffers[] = {nullptr, values};
  int numSchemaReleases{0};
  int numArrayReleases{0};

  ArrowSchema arrowSchema = makeArrowSchema("i");
  arrowSchema.release = countSchemaRelease;
  arrowSchema.private_data = &numSchemaReleases;
  ArrowArray arrowArray = makeArrowArray(buffers, 2, 4, 0);
  arrowArray.release = countArrayRelease;
  arrowArray.private_data = &numArrayReleases;

  // Create a Velox Vector from Arrow and then destruct it. The owner calls
  // the release callbacks then, and the viewer leaves them to the caller.
  {
    auto _ = importFromArrow(arrowSchema, arrowArray, pool());
  }

  if (isViewer()) {
    EXPECT_EQ(numSchemaReleases, 0);
    EXPECT_EQ(numArrayReleases, 0);
  } else {
    EXPECT_EQ(numSchemaReleases, 1);
    EXPECT_EQ(numArrayReleases, 1);
  }
}

TEST_P(ArrowBridgeImportTest, releaseAfterFailedImport) {
  const int64_t values[] = {1, 2, 3, 4};
  const void* buffers[] = {nullptr, values};
  int numSchemaReleases{0};
  int numArrayReleases{0};

  // Velox has no type for Arrow durations.
  ArrowSchema arrowSchema = makeArrowSchema("tDu");
  arrowSchema.release = countSchemaRelease;
  arrowSchema.private_data = &numSchemaReleases;
  ArrowArray arrowArray = makeArrowArray(buffers, 2, 4, 0);
  arrowArray.release = countArrayRelease;
  arrowArray.private_data = &numArrayReleases;

  VELOX_ASSERT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()),
      "Unable to convert 'tDu' ArrowSchema format type to Velox");

  // The owner marks the inputs released and releases each structure once. The
  // viewer leaves the release callbacks alone and releases nothing.
  if (isViewer()) {
    EXPECT_EQ(arrowSchema.release, countSchemaRelease);
    EXPECT_EQ(arrowArray.release, countArrayRelease);
    EXPECT_EQ(numSchemaReleases, 0);
    EXPECT_EQ(numArrayReleases, 0);
  } else {
    EXPECT_EQ(arrowSchema.release, nullptr);
    EXPECT_EQ(arrowArray.release, nullptr);
    EXPECT_EQ(numSchemaReleases, 1);
    EXPECT_EQ(numArrayReleases, 1);
  }
}

TEST_P(ArrowBridgeImportTest, releaseAfterFailedChildImport) {
  const int32_t values[] = {1, 2, 3, 4};
  const void* childBuffers[] = {nullptr, values};
  const void* rowBuffers[] = {nullptr};
  // Releases of the row, then of its two children.
  std::vector<int> numSchemaReleases(3, 0);
  std::vector<int> numArrayReleases(3, 0);

  ArrowSchema childSchemas[] = {makeArrowSchema("i"), makeArrowSchema("i")};
  ArrowArray childArrays[] = {
      makeArrowArray(childBuffers, 2, 4, 0),
      makeArrowArray(childBuffers, 2, 4, 0),
  };
  // The second child misses its values buffer, so the import fails after
  // importing the first child.
  childArrays[1].n_buffers = 1;
  const char* names[] = {"a", "b"};
  ArrowSchema* schemaChildren[2];
  ArrowArray* arrayChildren[2];
  for (int i = 0; i < 2; ++i) {
    childSchemas[i].name = names[i];
    childSchemas[i].release = countSchemaRelease;
    childSchemas[i].private_data = &numSchemaReleases[i + 1];
    childArrays[i].release = countArrayRelease;
    childArrays[i].private_data = &numArrayReleases[i + 1];
    schemaChildren[i] = &childSchemas[i];
    arrayChildren[i] = &childArrays[i];
  }

  ArrowSchema arrowSchema = makeArrowSchema("+s");
  arrowSchema.n_children = 2;
  arrowSchema.children = schemaChildren;
  arrowSchema.release = countSchemaRelease;
  arrowSchema.private_data = &numSchemaReleases[0];
  ArrowArray arrowArray = makeArrowArray(rowBuffers, 1, 4, 0);
  arrowArray.n_children = 2;
  arrowArray.children = arrayChildren;
  arrowArray.release = countArrayRelease;
  arrowArray.private_data = &numArrayReleases[0];

  VELOX_ASSERT_THROW(
      importFromArrow(arrowSchema, arrowArray, pool()),
      "Primitive types expect two buffers as input");

  if (isViewer()) {
    EXPECT_EQ(arrowSchema.release, countSchemaRelease);
    EXPECT_EQ(arrowArray.release, countArrayRelease);
    EXPECT_THAT(numSchemaReleases, testing::ElementsAre(0, 0, 0));
    EXPECT_THAT(numArrayReleases, testing::ElementsAre(0, 0, 0));
  } else {
    EXPECT_EQ(arrowSchema.release, nullptr);
    EXPECT_EQ(arrowArray.release, nullptr);
    EXPECT_THAT(numSchemaReleases, testing::ElementsAre(1, 1, 1));
    EXPECT_THAT(numArrayReleases, testing::ElementsAre(1, 1, 1));
  }
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    ArrowBridgeImportTest,
    ArrowBridgeImportTest,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "viewer" : "owner";
    });

} // namespace
} // namespace facebook::velox::test
