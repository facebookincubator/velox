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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>

namespace facebook::velox::cudf_velox::sparksql {

/// Spark legacy numeric casts which differ from cudf::cast. Supports floating
/// point to integral, and numeric/decimal to DECIMAL64 or DECIMAL128. Decimal
/// casts round half away from zero and return null on precision overflow;
/// floating-to-integral casts saturate, narrow, and map NaN to zero as Spark
/// does. targetPrecision is required only for decimal results.
std::unique_ptr<cudf::column> legacyCastNumeric(
    cudf::column_view input,
    cudf::data_type targetType,
    int32_t targetPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::sparksql
