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
#pragma once

#include "velox/type/Type.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

/// Checks that the column is STRING-encoded serialized decimal aggregation state.
void validateIntermediateColumnType(cudf::column_view const& column);

std::unique_ptr<cudf::column> castCountColumnToInt64(
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream);

std::unique_ptr<cudf::column> serializeDecimalPartialOrIntermediateState(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream);

std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox
