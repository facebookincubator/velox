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

#include "velox/experimental/cudf/expression/NullMask.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;

namespace {

cudf::size_type checkedSize(const std::vector<bool>& values) {
  VELOX_CHECK_LE(
      values.size(),
      static_cast<size_t>(std::numeric_limits<cudf::size_type>::max()));
  return static_cast<cudf::size_type>(values.size());
}

cudf::size_type countNulls(const std::vector<bool>& valid) {
  auto nullCount = std::count(valid.begin(), valid.end(), false);
  VELOX_CHECK_LE(
      nullCount,
      static_cast<decltype(nullCount)>(
          std::numeric_limits<cudf::size_type>::max()));
  return static_cast<cudf::size_type>(nullCount);
}

rmm::device_buffer makeNullMask(
    const std::vector<bool>& valid,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto size = checkedSize(valid);
  auto nullMask =
      cudf::create_null_mask(size, cudf::mask_state::ALL_NULL, stream, mr);
  std::vector<cudf::bitmask_type> hostMask(cudf::num_bitmask_words(size), 0);
  for (cudf::size_type i = 0; i < size; ++i) {
    if (valid[i]) {
      cudf::set_bit_unsafe(hostMask.data(), i);
    }
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      nullMask.data(),
      hostMask.data(),
      hostMask.size() * sizeof(cudf::bitmask_type),
      cudaMemcpyHostToDevice,
      stream.value()));
  return nullMask;
}

std::unique_ptr<cudf::column> makeNullableIntColumn(
    const std::vector<bool>& valid,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto size = checkedSize(valid);
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      size,
      rmm::device_buffer(size * sizeof(int32_t), stream, mr),
      makeNullMask(valid, stream, mr),
      countNulls(valid));
}

std::vector<cudf::bitmask_type> copyNullMaskToHost(
    cudf::column_view column,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::bitmask_type> hostMask(
      cudf::num_bitmask_words(column.size()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostMask.data(),
      column.null_mask(),
      hostMask.size() * sizeof(cudf::bitmask_type),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return hostMask;
}

void assertValidity(
    cudf::column_view column,
    const std::vector<bool>& expectedValid,
    rmm::cuda_stream_view stream) {
  ASSERT_EQ(column.size(), checkedSize(expectedValid));
  EXPECT_EQ(column.null_count(), countNulls(expectedValid));
  if (!column.has_nulls()) {
    EXPECT_FALSE(column.nullable());
    return;
  }

  ASSERT_TRUE(column.nullable());
  auto hostMask = copyNullMaskToHost(column, stream);
  for (cudf::size_type i = 0; i < column.size(); ++i) {
    EXPECT_EQ(cudf::bit_is_set(hostMask.data(), i), expectedValid[i])
        << "at row " << i;
  }
}

class NullMaskTest : public ::testing::Test {
 protected:
  rmm::cuda_stream_view stream_{cudf::get_default_stream()};
  rmm::device_async_resource_ref mr_{cudf::get_current_device_resource_ref()};
};

TEST_F(NullMaskTest, sourceWithoutNullsLeavesResultUnchanged) {
  std::vector<bool> resultValid{true, false, true, false, true};
  auto result = makeNullableIntColumn(resultValid, stream_, mr_);
  auto source = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      checkedSize(resultValid),
      cudf::mask_state::UNALLOCATED,
      stream_,
      mr_);

  mergeNullSourceNullsIntoResult(*result, source->view(), stream_, mr_);

  assertValidity(result->view(), resultValid, stream_);
}

TEST_F(NullMaskTest, copiesSourceNullMaskIntoNonNullableResult) {
  std::vector<bool> sourceValid{true, false, true, false, true};
  auto result = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      checkedSize(sourceValid),
      cudf::mask_state::UNALLOCATED,
      stream_,
      mr_);
  auto source = makeNullableIntColumn(sourceValid, stream_, mr_);

  mergeNullSourceNullsIntoResult(*result, source->view(), stream_, mr_);

  assertValidity(result->view(), sourceValid, stream_);
}

TEST_F(NullMaskTest, mergesExistingAndSourceNullMasks) {
  std::vector<bool> resultValid{true, false, true, false, true};
  std::vector<bool> sourceValid{true, true, false, false, true};
  std::vector<bool> expectedValid{true, false, false, false, true};
  auto result = makeNullableIntColumn(resultValid, stream_, mr_);
  auto source = makeNullableIntColumn(sourceValid, stream_, mr_);

  mergeNullSourceNullsIntoResult(*result, source->view(), stream_, mr_);

  assertValidity(result->view(), expectedValid, stream_);
}

TEST_F(NullMaskTest, mergesMultipleSourceNullMasks) {
  std::vector<bool> resultValid{true, true, false, true, true};
  std::vector<bool> firstSourceValid{true, false, true, true, true};
  std::vector<bool> secondSourceValid{true, true, true, false, true};
  std::vector<bool> expectedValid{true, false, false, false, true};
  auto result = makeNullableIntColumn(resultValid, stream_, mr_);
  auto firstSource = makeNullableIntColumn(firstSourceValid, stream_, mr_);
  auto secondSource = makeNullableIntColumn(secondSourceValid, stream_, mr_);

  mergeNullSourceNullsIntoResult(
      *result, {firstSource->view(), secondSource->view()}, stream_, mr_);

  assertValidity(result->view(), expectedValid, stream_);
}

} // namespace
