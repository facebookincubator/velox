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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/DecimalAggregationDevice.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>

#include <limits>

namespace facebook::velox::cudf_velox {

DecimalSumStateColumns deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(
      stateCol.type().id() == cudf::type_id::STRING,
      "Decimal sum state requires STRING/VARBINARY column (type is {})",
      cudf::type_to_name(stateCol.type()));
  // The decoded sum/count columns are consumed by the next groupby/reduce and
  // never leave the operator and should use the temporary memory resource.
  auto const mr = get_temp_mr();
  auto numRows = stateCol.size();
  if (numRows == 0) {
    DecimalSumStateColumns empty;
    empty.sum = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::DECIMAL128, -scale},
        0,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    empty.count = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        0,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    return empty;
  }

  // For fully-null state columns there is nothing to deserialize. Avoid
  // launching unpack kernels over string payload buffers that may be empty.
  if (stateCol.nullable() && stateCol.null_count() == numRows) {
    DecimalSumStateColumns allNull;
    allNull.sum = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::DECIMAL128, -scale},
        numRows,
        cudf::mask_state::ALL_NULL,
        stream,
        mr);
    allNull.count = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        numRows,
        cudf::mask_state::ALL_NULL,
        stream,
        mr);
    return allNull;
  }

  cudf::strings_column_view strings(stateCol);

  auto offsetsView = strings.offsets();
  auto charsPtr = reinterpret_cast<const uint8_t*>(strings.chars_begin(stream));

  auto sumCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::DECIMAL128, -scale},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto countCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);

  auto sumView = sumCol->mutable_view();
  auto countView = countCol->mutable_view();

  // numRows is guaranteed positive here
  auto const offsetsType = offsetsView.type().id();
  VELOX_CHECK(
      offsetsType == cudf::type_id::INT32 ||
          offsetsType == cudf::type_id::INT64,
      "Decimal sum state requires INT32 or INT64 offsets (offset type is {})",
      cudf::type_to_name(offsetsView.type()));
  detail::unpackDecimalSumState(
      offsetsType,
      offsetsView,
      charsPtr,
      sumView,
      countView,
      numRows,
      stream);

  if (stateCol.nullable()) {
    auto nullMask = cudf::copy_bitmask(stateCol, stream, mr);
    auto nullCount = stateCol.null_count();
    sumCol->set_null_mask(std::move(nullMask), nullCount);
    auto countMask = cudf::copy_bitmask(stateCol, stream, mr);
    countCol->set_null_mask(std::move(countMask), nullCount);
  }

  DecimalSumStateColumns result;
  result.sum = std::move(sumCol);
  result.count = std::move(countCol);
  return result;
}

std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal sum state requires INT64 count column (type is {})",
      static_cast<int>(countCol.type().id()));
  auto numRows = sumCol.size();
  VELOX_CHECK_EQ(
      numRows,
      countCol.size(),
      "Decimal sum state requires sum and count to be same size (sum size is {}, count size is {})",
      sumCol.size(),
      countCol.size());
  VELOX_CHECK_LE(
      numRows,
      static_cast<cudf::size_type>(std::numeric_limits<int32_t>::max()),
      "Too many rows to serialize decimal sum state (row count is {})",
      numRows);

  if (numRows == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  auto const rowCount = static_cast<int32_t>(numRows);

  auto const charsBytes =
      static_cast<int64_t>(numRows) * detail::kDecimalSumStateSize;
  auto const threshold = cudf::strings::get_offset64_threshold();
  auto const useLargeOffsets = charsBytes >= threshold;
  // Previously this guard threw std::overflow_error; Velox uses
  // VeloxRuntimeError for this guard.
  VELOX_CHECK(
      !useLargeOffsets || cudf::strings::is_large_strings_enabled(),
      "Size of output ({}) exceeds the column size limit ({})",
      charsBytes,
      threshold);

  auto const offsetsType =
      useLargeOffsets ? cudf::type_id::INT64 : cudf::type_id::INT32;
  auto offsetsCol = cudf::make_fixed_width_column(
      cudf::data_type{offsetsType},
      numRows + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto offsetsView = offsetsCol->mutable_view();

  rmm::device_buffer charsBuf(
      static_cast<size_t>(numRows) * detail::kDecimalSumStateSize, stream, mr);

  auto charsPtr = reinterpret_cast<uint8_t*>(charsBuf.data());
  detail::fillOffsetsForDecimalSumState(
      offsetsType, offsetsView, rowCount, stream);

  const auto sumType = sumCol.type().id();
  VELOX_CHECK(
      sumType == cudf::type_id::DECIMAL64 ||
          sumType == cudf::type_id::DECIMAL128,
      "Unsupported decimal sum column type (type is {})",
      cudf::type_to_name(sumCol.type()));
  detail::packDecimalSumState(
      sumType,
      offsetsType,
      sumCol,
      countCol.data<int64_t>(),
      offsetsView,
      charsPtr,
      rowCount,
      stream);

  auto [nullMask, nullCount] =
      detail::buildStateValidityMask(sumCol, countCol, stream, mr);
  return cudf::make_strings_column(
      static_cast<cudf::size_type>(numRows),
      std::move(offsetsCol),
      std::move(charsBuf),
      nullCount,
      std::move(nullMask));
}

std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal average requires INT64 count column (type is {})",
      cudf::type_to_name(countCol.type()));
  VELOX_CHECK(
      sumCol.type().id() == cudf::type_id::DECIMAL64 ||
          sumCol.type().id() == cudf::type_id::DECIMAL128,
      "Decimal average requires DECIMAL64 or DECIMAL128 sum column (type is {})",
      cudf::type_to_name(sumCol.type()));
  VELOX_CHECK_EQ(
      sumCol.size(),
      countCol.size(),
      "Decimal average requires sum and count to be same size (sum size is {}, count size is {})",
      sumCol.size(),
      countCol.size());

  auto numRows = sumCol.size();
  auto out = cudf::make_fixed_width_column(
      sumCol.type(), numRows, cudf::mask_state::UNALLOCATED, stream, mr);

  if (numRows > 0) {
    auto const rowCount = static_cast<int32_t>(numRows);
    const auto sumType = sumCol.type().id();
    detail::averageRoundDecimalSum(
        sumType,
        sumCol,
        countCol.data<int64_t>(),
        out->mutable_view(),
        rowCount,
        stream);
  }

  auto [nullMask, nullCount] =
      detail::buildStateValidityMask(sumCol, countCol, stream, mr);
  if (nullCount > 0) {
    out->set_null_mask(std::move(nullMask), nullCount);
  }
  return out;
}

} // namespace facebook::velox::cudf_velox
