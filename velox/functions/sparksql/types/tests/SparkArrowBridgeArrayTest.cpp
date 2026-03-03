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
#include <gtest/gtest.h>

#include "velox/common/base/Nulls.h"
#include "velox/core/QueryCtx.h"
#include "velox/functions/sparksql/types/TimestampNTZRegistration.h"
#include "velox/functions/sparksql/types/TimestampNTZType.h"
#include "velox/vector/arrow/Bridge.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

void mockSchemaRelease(ArrowSchema*) {}
void mockArrayRelease(ArrowArray*) {}

template <typename T>
struct VeloxToArrowType {
  using type = T;
};

class SparkArrowBridgeArrayExportTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    registerTimestampNTZType();
  }

  template <typename T>
  void testFlatVector(
      const std::vector<std::optional<T>>& inputData,
      const TypePtr& type = CppToType<T>::create()) {
    auto flatVector = vectorMaker_.flatVectorNullable(inputData, type);
    ArrowArray arrowArray;
    velox::exportToArrow(flatVector, arrowArray, pool_.get(), options_);

    validateArray(inputData, arrowArray);

    arrowArray.release(&arrowArray);
    EXPECT_EQ(nullptr, arrowArray.release);
    EXPECT_EQ(nullptr, arrowArray.private_data);
  }

  // Boiler plate structures required by vectorMaker.
  ArrowOptions options_;
  std::shared_ptr<core::QueryCtx> queryCtx_{velox::core::QueryCtx::create()};
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  core::ExecCtx execCtx_{pool_.get(), queryCtx_.get()};
  facebook::velox::test::VectorMaker vectorMaker_{execCtx_.pool()};

 private:
  // Helper functions for verification. Only numeric types are supported.
  template <typename T>
  void validateArray(
      const std::vector<std::optional<T>>& inputData,
      const ArrowArray& arrowArray) {
    EXPECT_EQ(inputData.size(), arrowArray.length);
    EXPECT_EQ(0, arrowArray.offset);
    EXPECT_EQ(0, arrowArray.n_children);

    EXPECT_EQ(nullptr, arrowArray.children);
    EXPECT_EQ(nullptr, arrowArray.dictionary);
    EXPECT_NE(nullptr, arrowArray.release);

    validateNulls(inputData, arrowArray);
    validateNumericalArray(inputData, arrowArray);
  }

  template <typename T>
  void validateNumericalArray(
      const std::vector<std::optional<T>>& inputData,
      const ArrowArray& arrowArray) {
    using TArrow = typename VeloxToArrowType<T>::type;

    // Null and values buffers.
    ASSERT_EQ(2, arrowArray.n_buffers);
    ASSERT_NE(nullptr, arrowArray.buffers);

    const uint64_t* nulls = static_cast<const uint64_t*>(arrowArray.buffers[0]);
    const TArrow* values = static_cast<const TArrow*>(arrowArray.buffers[1]);

    EXPECT_NE(values, nullptr);

    for (size_t i = 0; i < inputData.size(); ++i) {
      if (inputData[i] == std::nullopt) {
        EXPECT_TRUE(bits::isBitNull(nulls, i));
      } else {
        if (nulls) {
          EXPECT_FALSE(bits::isBitNull(nulls, i));
        }

        // Boolean is packed in a single bit so it needs special treatment.
        if constexpr (std::is_same_v<T, bool>) {
          EXPECT_EQ(
              inputData[i],
              bits::isBitSet(reinterpret_cast<const uint64_t*>(values), i))
              << "mismatch at index " << i;
        } else if constexpr (std::is_same_v<T, Timestamp>) {
          EXPECT_TRUE(validateTimestamp(
              inputData[i].value(), options_.timestampUnit, values[i]))
              << "mismatch at index " << i;
        } else {
          EXPECT_EQ(inputData[i], values[i]) << "mismatch at index " << i;
        }
      }
    }
  }

  // Validate nulls and returns whether this array uses Arrow's Null Layout.
  template <typename T>
  void validateNulls(
      const std::vector<std::optional<T>>& inputData,
      const ArrowArray& arrowArray) {
    size_t nullCount =
        std::count(inputData.begin(), inputData.end(), std::nullopt);
    EXPECT_EQ(arrowArray.null_count, nullCount);

    if (arrowArray.null_count == 0) {
      EXPECT_EQ(arrowArray.buffers[0], nullptr);
    } else {
      EXPECT_NE(arrowArray.buffers[0], nullptr);
      EXPECT_EQ(
          arrowArray.null_count,
          bits::countNulls(
              static_cast<const uint64_t*>(arrowArray.buffers[0]),
              0,
              inputData.size()));
    }
  }
};

TEST_F(SparkArrowBridgeArrayExportTest, flatTimestampNTZ) {
  testFlatVector<int64_t>(
      {
          std::nullopt,
          99876,
          std::nullopt,
          12345678,
          std::numeric_limits<int64_t>::max(),
          std::numeric_limits<int64_t>::min(),
          std::nullopt,
      },
      TIMESTAMP_NTZ());
}

class SparkArrowBridgeArrayImportTest : public SparkArrowBridgeArrayExportTest {
 protected:
  // Used by this base test class to import Arrow data and create Velox Vector.
  // Derived test classes should call the import function under test.
  virtual VectorPtr importFromArrow(
      ArrowSchema& arrowSchema,
      ArrowArray& arrowArray,
      memory::MemoryPool* pool) = 0;

  virtual bool isViewer() const = 0;

  // Takes a vector with input data, generates an input ArrowArray and Velox
  // Vector (using vector maker). Then converts ArrowArray into Velox vector and
  // assert that both Velox vectors are semantically the same. Only numeric
  // types are supported.
  template <typename TOutput, typename TInput = TOutput>
  void testArrowImport(
      const char* format,
      const std::vector<std::optional<TInput>>& inputValues) {
    ArrowContextHolder holder;
    auto arrowArray = [&] { return fillArrowArray(inputValues, holder); }();

    auto arrowSchema = makeArrowSchema(format);
    auto output = importFromArrow(arrowSchema, arrowArray, pool_.get());

    assertVectorContent(inputValues, output, arrowArray.null_count);
    EXPECT_FALSE(BaseVector::isVectorWritable(output));
  }

 private:
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

    holder.values = AlignedBuffer::allocate<TArrow>(length, pool_.get());
    holder.nulls = AlignedBuffer::allocate<uint64_t>(length, pool_.get());

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

  template <typename T>
  void assertVectorContent(
      const std::vector<std::optional<T>>& inputValues,
      const VectorPtr& convertedVector,
      size_t nullCount) {
    EXPECT_EQ((nullCount > 0), convertedVector->mayHaveNulls());
    EXPECT_EQ(nullCount, *convertedVector->getNullCount());
    EXPECT_EQ(inputValues.size(), convertedVector->size());

    auto expected = vectorMaker_.flatVectorNullable(inputValues);

    // Assert new vector contents.
    for (vector_size_t i = 0; i < convertedVector->size(); ++i) {
      ASSERT_TRUE(expected->equalValueAt(convertedVector.get(), i, i))
          << "at " << i << ": " << expected->toString(i) << " vs. "
          << convertedVector->toString(i);
    }
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
};

class SparkArrowBridgeArrayImportAsViewerTest
    : public SparkArrowBridgeArrayImportTest {
  bool isViewer() const override {
    return true;
  }

  VectorPtr importFromArrow(
      ArrowSchema& arrowSchema,
      ArrowArray& arrowArray,
      memory::MemoryPool* pool) override {
    return facebook::velox::importFromArrowAsViewer(
        arrowSchema, arrowArray, pool);
  }
};

class SparkArrowBridgeArrayImportAsOwnerTest
    : public SparkArrowBridgeArrayImportAsViewerTest {
  bool isViewer() const override {
    return false;
  }

  VectorPtr importFromArrow(
      ArrowSchema& arrowSchema,
      ArrowArray& arrowArray,
      memory::MemoryPool* pool) override {
    return facebook::velox::importFromArrowAsOwner(
        arrowSchema, arrowArray, pool);
  }
};

TEST_F(SparkArrowBridgeArrayImportAsOwnerTest, timestampNTZ) {
  testArrowImport<int64_t>("ts_ntz", {});
  testArrowImport<int64_t>("ts_ntz", {std::nullopt});
  testArrowImport<int64_t>("ts_ntz", {-99, 4, 318321631, 1211, -12});
  testArrowImport<int64_t>("ts_ntz", {std::nullopt, 12345678, std::nullopt});
  testArrowImport<int64_t>("ts_ntz", {std::nullopt, std::nullopt});
}

TEST_F(SparkArrowBridgeArrayImportAsViewerTest, timestampNTZ) {
  testArrowImport<int64_t>("ts_ntz", {});
  testArrowImport<int64_t>("ts_ntz", {std::nullopt});
  testArrowImport<int64_t>("ts_ntz", {-99, 4, 318321631, 1211, -12});
  testArrowImport<int64_t>("ts_ntz", {std::nullopt, 12345678, std::nullopt});
  testArrowImport<int64_t>("ts_ntz", {std::nullopt, std::nullopt});
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
