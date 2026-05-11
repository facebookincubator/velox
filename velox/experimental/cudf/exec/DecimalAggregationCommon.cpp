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

#include "velox/experimental/cudf/exec/DecimalAggregationCommon.h"

#include "velox/common/base/Exceptions.h"
#include "velox/experimental/cudf/exec/DecimalAggregationKernels.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox {

void validateIntermediateColumnType(cudf::column_view const& column) {
  VELOX_CHECK(
      column.type().id() == cudf::type_id::STRING,
      "Expected serialized decimal aggregation state: Velox VARBINARY represented as cuDF STRING");
}

std::unique_ptr<cudf::column> castCountColumnToInt64(
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream) {
  if (count->type().id() != cudf::type_id::INT64) {
    count = cudf::cast(
        *count,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        get_output_mr());
  }
  return count;
}

std::unique_ptr<cudf::column> serializeDecimalPartialOrIntermediateState(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream) {
  count = castCountColumnToInt64(std::move(count), stream);
  return serializeDecimalSumState(
      sum->view(), count->view(), stream, get_output_mr());
}

std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    rmm::cuda_stream_view stream) {
  count = castCountColumnToInt64(std::move(count), stream);
  auto avgCol = computeDecimalAverage(
      sum->view(), count->view(), stream, get_output_mr());
  auto const cudfOutType = veloxToCudfDataType(resultType);
  if (avgCol->type() != cudfOutType) {
    avgCol = cudf::cast(
        avgCol->view(), cudfOutType, stream, get_output_mr());
  }
  return avgCol;
}

} // namespace facebook::velox::cudf_velox
