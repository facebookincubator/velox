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

#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <limits>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

constexpr int kBitsPerWord = 8 * sizeof(cudf::bitmask_type);

std::pair<rmm::device_buffer, cudf::size_type> makeNullMask(
    const std::vector<bool>& valid,
    rmm::cuda_stream_view stream) {
  auto numBits = static_cast<cudf::size_type>(valid.size());
  if (numBits == 0) {
    return {rmm::device_buffer{}, 0};
  }
  auto maskBytes = cudf::bitmask_allocation_size_bytes(numBits);
  auto numWords = maskBytes / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host(numWords, 0);
  cudf::size_type nullCount = 0;
  for (cudf::size_type i = 0; i < numBits; ++i) {
    if (valid[i]) {
      auto word = i / kBitsPerWord;
      auto bit = i % kBitsPerWord;
      host[word] |= (cudf::bitmask_type{1} << bit);
    } else {
      ++nullCount;
    }
  }
  rmm::device_buffer mask(maskBytes, stream);
  if (!host.empty()) {
    auto status = cudaMemcpyAsync(
        mask.data(),
        host.data(),
        host.size() * sizeof(cudf::bitmask_type),
        cudaMemcpyHostToDevice,
        stream.value());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.synchronize();
  }
  return {std::move(mask), nullCount};
}

template <typename T>
std::unique_ptr<cudf::column> makeFixedWidthColumn(
    cudf::data_type type,
    const std::vector<T>& values,
    const std::vector<bool>* valid,
    rmm::cuda_stream_view stream) {
  auto col = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream);
  if (!values.empty()) {
    auto status = cudaMemcpyAsync(
        col->mutable_view().data<T>(),
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream.value());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.synchronize();
  }
  if (valid) {
    auto [mask, nullCount] = makeNullMask(*valid, stream);
    col->set_null_mask(std::move(mask), nullCount);
  }
  return col;
}

std::unique_ptr<cudf::column> makeDecimal32Column(
    const std::vector<int32_t>& values,
    int32_t scale,
    const std::vector<bool>* valid,
    rmm::cuda_stream_view stream) {
  cudf::data_type type{cudf::type_id::DECIMAL32, -scale};
  return makeFixedWidthColumn(type, values, valid, stream);
}

template <typename T>
std::vector<T> copyColumnData(
    const cudf::column_view& view,
    rmm::cuda_stream_view stream) {
  std::vector<T> host(view.size());
  if (view.size() == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.data<T>(),
      view.size() * sizeof(T),
      cudaMemcpyDeviceToHost,
      stream.value());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.synchronize();
  return host;
}

std::vector<cudf::bitmask_type> copyNullMask(
    const cudf::column_view& view,
    rmm::cuda_stream_view stream) {
  auto numWords = cudf::num_bitmask_words(view.size());
  std::vector<cudf::bitmask_type> host(numWords, 0);
  if (!view.nullable() || numWords == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.null_mask(),
      host.size() * sizeof(cudf::bitmask_type),
      cudaMemcpyDeviceToHost,
      stream.value());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.synchronize();
  return host;
}

bool isValidAt(const std::vector<cudf::bitmask_type>& mask, size_t idx) {
  if (mask.empty()) {
    return true;
  }
  auto word = idx / kBitsPerWord;
  auto bit = idx % kBitsPerWord;
  return (mask[word] >> bit) & 1;
}

void skipIfDecimal32To64CastUnsupported(int32_t scale) {
  const cudf::data_type src{cudf::type_id::DECIMAL32, -scale};
  const cudf::data_type dst{cudf::type_id::DECIMAL64, -scale};
  if (!cudf::is_supported_cast(src, dst)) {
    GTEST_SKIP() << "libcudf does not support DECIMAL32 to DECIMAL64 cast";
  }
}

class DecimalWideningTest : public ::testing::Test,
                            public facebook::velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
      GTEST_SKIP() << "cudaGetDeviceCount failed: " << static_cast<int>(status)
                   << " (" << cudaGetErrorString(status) << ")";
    }
    if (deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices visible (check CUDA_VISIBLE_DEVICES)";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
  }
};

TEST_F(DecimalWideningTest, castDecimal32InputToDecimal64PreservesValues) {
  skipIfDecimal32To64CastUnsupported(2);

  auto stream = cudf::get_default_stream();
  const std::vector<int32_t> inputValues = {
      1111, -2222, 0, 99999, std::numeric_limits<int32_t>::min()};
  const std::vector<bool> valid = {true, true, true, true, false};
  auto inputCol =
      makeDecimal32Column(inputValues, 2, &valid, stream);

  std::unique_ptr<cudf::column> holder;
  auto widened =
      castDecimal32InputToDecimal64(inputCol->view(), holder, stream);
  ASSERT_NE(holder, nullptr);
  EXPECT_EQ(widened.type().id(), cudf::type_id::DECIMAL64);
  EXPECT_EQ(widened.type().scale(), inputCol->type().scale());

  auto outputValues = copyColumnData<int64_t>(widened, stream);
  auto outputMask = copyNullMask(widened, stream);
  ASSERT_EQ(outputValues.size(), inputValues.size());
  for (size_t i = 0; i < inputValues.size(); ++i) {
    if (!isValidAt(outputMask, i)) {
      continue;
    }
    EXPECT_EQ(outputValues[i], static_cast<int64_t>(inputValues[i])) << "at " << i;
  }
}

TEST_F(DecimalWideningTest, castDecimal32InputToDecimal64NoOpForDecimal64) {
  auto stream = cudf::get_default_stream();
  const std::vector<int64_t> values = {100, -200};
  cudf::data_type type{cudf::type_id::DECIMAL64, -2};
  auto inputCol = makeFixedWidthColumn(type, values, nullptr, stream);

  std::unique_ptr<cudf::column> holder;
  auto result = castDecimal32InputToDecimal64(inputCol->view(), holder, stream);
  EXPECT_EQ(holder, nullptr);
  EXPECT_EQ(result.type().id(), cudf::type_id::DECIMAL64);
  EXPECT_EQ(copyColumnData<int64_t>(result, stream), values);
}

TEST_F(DecimalWideningTest, alignTableColumnsToOutputTypeWidensDecimal32) {
  skipIfDecimal32To64CastUnsupported(2);

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  const std::vector<int32_t> decimalValues = {1111, 2222, 3333};
  const std::vector<int64_t> bigintValues = {10, 20, 30};
  auto decimalCol = makeDecimal32Column(decimalValues, 2, nullptr, stream);
  auto bigintCol = makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT64}, bigintValues, nullptr, stream);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(decimalCol));
  columns.push_back(std::move(bigintCol));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto outputType = ROW({"a", "b"}, {DECIMAL(7, 2), BIGINT()});
  auto aligned =
      alignTableColumnsToOutputType(std::move(table), outputType, stream, mr);
  ASSERT_EQ(aligned->num_columns(), 2);
  EXPECT_EQ(aligned->view().column(0).type().id(), cudf::type_id::DECIMAL64);
  EXPECT_EQ(aligned->view().column(1).type().id(), cudf::type_id::INT64);

  auto widenedValues =
      copyColumnData<int64_t>(aligned->view().column(0), stream);
  ASSERT_EQ(widenedValues.size(), decimalValues.size());
  for (size_t i = 0; i < decimalValues.size(); ++i) {
    EXPECT_EQ(widenedValues[i], static_cast<int64_t>(decimalValues[i]));
  }

  auto bigintOut =
      copyColumnData<int64_t>(aligned->view().column(1), stream);
  EXPECT_EQ(bigintOut, bigintValues);
}

TEST_F(DecimalWideningTest, alignTableColumnsToOutputTypeNoOpWhenAlreadyAligned) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  const std::vector<int64_t> values = {100, 200};
  cudf::data_type type{cudf::type_id::DECIMAL64, -2};
  auto col = makeFixedWidthColumn(type, values, nullptr, stream);
  auto* rawData = col->view().data<int64_t>();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto outputType = ROW({"a"}, {DECIMAL(5, 2)});
  auto aligned =
      alignTableColumnsToOutputType(std::move(table), outputType, stream, mr);
  ASSERT_EQ(aligned->num_columns(), 1);
  EXPECT_EQ(aligned->view().column(0).type().id(), cudf::type_id::DECIMAL64);
  EXPECT_EQ(aligned->view().column(0).data<int64_t>(), rawData);
}

TEST_F(DecimalWideningTest, alignedDecimal32RoundTripsToVelox) {
  skipIfDecimal32To64CastUnsupported(2);

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  const std::vector<int32_t> values = {1111, -1111, 0};
  const std::vector<bool> valid = {true, true, false};
  auto decimalCol = makeDecimal32Column(values, 2, &valid, stream);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(decimalCol));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto outputType = ROW({"a"}, {DECIMAL(7, 2)});
  auto aligned =
      alignTableColumnsToOutputType(std::move(table), outputType, stream, mr);

  auto expected = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int64_t>(
          {1111, -1111, std::nullopt}, DECIMAL(7, 2))});
  auto result = with_arrow::toVeloxColumn(
      aligned->view(), pool_.get(), outputType, stream, mr);
  stream.synchronize();
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(DecimalWideningTest, alignTableColumnsToOutputTypeWidensDictionaryDecimal32) {
  skipIfDecimal32To64CastUnsupported(2);

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  const std::vector<int32_t> values = {1111, 1111, 2222, 2222, 3333, 3333};
  auto decimalCol = makeDecimal32Column(values, 2, nullptr, stream);
  auto dictionaryCol = cudf::dictionary::encode(
      decimalCol->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
  VELOX_CHECK(cudf::is_dictionary(dictionaryCol->type()));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(dictionaryCol));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto outputType = ROW({"a"}, {DECIMAL(7, 2)});
  auto aligned =
      alignTableColumnsToOutputType(std::move(table), outputType, stream, mr);
  ASSERT_EQ(aligned->num_columns(), 1);
  EXPECT_EQ(aligned->view().column(0).type().id(), cudf::type_id::DECIMAL64);

  const std::vector<int64_t> expectedValues = {
      1111, 1111, 2222, 2222, 3333, 3333};
  auto widenedValues =
      copyColumnData<int64_t>(aligned->view().column(0), stream);
  EXPECT_EQ(widenedValues, expectedValues);
}

} // namespace
} // namespace facebook::velox::cudf_velox
