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
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <string>

namespace facebook::velox::cudf_velox {

std::unique_ptr<cudf::column> integralCheckedModulus(
    cudf::data_type type,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> integralCheckedModulus(
    cudf::data_type type,
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> integralCheckedModulus(
    cudf::data_type type,
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Register Presto-specific CUDF functions.
void registerPrestoFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox
