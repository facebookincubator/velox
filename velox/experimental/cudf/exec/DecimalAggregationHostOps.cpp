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
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

namespace facebook::velox::cudf_velox {

namespace {

cudf::data_type physicalColumnType(cudf::column_view column) {
  if (cudf::is_dictionary(column.type())) {
    return cudf::dictionary_column_view{column}.keys().type();
  }
  return column.type();
}

} // namespace

void validateIntermediateColumnType(cudf::column_view const& column) {
  // fmt does not understand cudf::type_id enum class
  auto const colType = static_cast<int>(column.type().id());
  VELOX_CHECK_EQ(
      colType,
      static_cast<int>(cudf::type_id::STRING),
      "Expected serialized decimal aggregation state: Velox VARBINARY represented as cuDF STRING (got type {})",
      colType);
}

cudf::column_view castDecimal32InputToDecimal64(
    cudf::column_view inputCol,
    std::unique_ptr<cudf::column>& holder,
    rmm::cuda_stream_view stream) {
  if (inputCol.type().id() != cudf::type_id::DECIMAL32) {
    return inputCol;
  }
  holder = cudf::cast(
      inputCol,
      cudf::data_type{cudf::type_id::DECIMAL64, inputCol.type().scale()},
      stream,
      get_temp_mr());
  return holder->view();
}

cudf::column_view castDecimal64InputToDecimal128(
    cudf::column_view inputCol,
    std::unique_ptr<cudf::column>& holder,
    rmm::cuda_stream_view stream) {
  if (inputCol.type().id() != cudf::type_id::DECIMAL64) {
    return inputCol;
  }
  holder = cudf::cast(
      inputCol,
      cudf::data_type{cudf::type_id::DECIMAL128, inputCol.type().scale()},
      stream,
      get_temp_mr());
  return holder->view();
}

std::unique_ptr<cudf::column> castDecimalColumnIfNeeded(
    cudf::column_view inputCol,
    cudf::data_type expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  cudf::column_view valueCol = inputCol;
  std::unique_ptr<cudf::column> decoded;
  if (cudf::is_dictionary(inputCol.type())) {
    decoded = cudf::dictionary::decode(
        cudf::dictionary_column_view{inputCol}, stream, mr);
    valueCol = decoded->view();
  }

  if (valueCol.type() == expectedType) {
    return decoded;
  }
  if (!cudf::is_fixed_point(valueCol.type()) &&
      !cudf::is_fixed_point(expectedType)) {
    return decoded;
  }
  return cudf::cast(valueCol, expectedType, stream, mr);
}

std::unique_ptr<cudf::table> alignTableColumnsToOutputType(
    std::unique_ptr<cudf::table> table,
    const RowTypePtr& outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto columns = table->release();
  const auto numColumns = columns.size();
  const auto numOutputColumns =
      std::min(numColumns, static_cast<size_t>(outputType->size()));

  bool needsRebuild = false;
  for (size_t i = 0; i < numOutputColumns; ++i) {
    const auto columnView = columns[i]->view();
    const auto expectedType = veloxToCudfDataType(outputType->childAt(i));
    const auto actualType = physicalColumnType(columnView);
    if (cudf::is_dictionary(columnView.type()) ||
        (actualType != expectedType &&
         (cudf::is_fixed_point(actualType) ||
          cudf::is_fixed_point(expectedType)))) {
      needsRebuild = true;
      break;
    }
  }

  if (!needsRebuild) {
    return std::make_unique<cudf::table>(std::move(columns));
  }

  std::vector<std::unique_ptr<cudf::column>> alignedColumns;
  alignedColumns.reserve(numColumns);
  for (size_t i = 0; i < numColumns; ++i) {
    if (i < numOutputColumns) {
      const auto expectedType = veloxToCudfDataType(outputType->childAt(i));
      if (auto casted = castDecimalColumnIfNeeded(
              columns[i]->view(), expectedType, stream, mr)) {
        alignedColumns.push_back(std::move(casted));
        continue;
      }
    }
    alignedColumns.push_back(std::move(columns[i]));
  }
  return std::make_unique<cudf::table>(std::move(alignedColumns));
}

std::unique_ptr<cudf::column> castCountColumnToInt64(
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream) {
  if (count->type().id() != cudf::type_id::INT64) {
    count = cudf::cast(
        *count, cudf::data_type{cudf::type_id::INT64}, stream, get_temp_mr());
  }
  return count;
}

std::unique_ptr<cudf::column> serializeDecimalPartialOrIntermediateState(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  count = castCountColumnToInt64(std::move(count), stream);
  return serializeDecimalSumState(sum->view(), count->view(), stream, mr);
}

std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  count = castCountColumnToInt64(std::move(count), stream);
  auto avgCol = computeDecimalAverage(sum->view(), count->view(), stream, mr);
  auto const cudfOutType = veloxToCudfDataType(resultType);
  if (avgCol->type() != cudfOutType) {
    avgCol = cudf::cast(avgCol->view(), cudfOutType, stream, mr);
  }
  return avgCol;
}

} // namespace facebook::velox::cudf_velox
