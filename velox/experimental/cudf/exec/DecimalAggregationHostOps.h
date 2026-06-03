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

// Asserts that a column holds serialized decimal aggregate state in the form
// Velox uses for VARBINARY: a cuDF STRING column whose bytes are the packed
// sum/count payloads (see serializeDecimalSumState).
void validateIntermediateColumnType(cudf::column_view const& column);

// Casts a DECIMAL64 column up to DECIMAL128 (scale preserved) so a subsequent
// SUM accumulates in 128 bits instead of wrapping. Allocates the casted column
// from the temporary memory resource into holder and returns its view.
// Lifetime stays valid only while holder is alive.
cudf::column_view castDecimal64InputToDecimal128(
    cudf::column_view inputCol,
    std::unique_ptr<cudf::column>& holder,
    rmm::cuda_stream_view stream);

// Ensures the partial-row count column is INT64, casting with the temporary
// memory resource (the result is consumed internally, not part of operator
// output) when the incoming type differs.
std::unique_ptr<cudf::column> castCountColumnToInt64(
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream);

// Normalizes the count column to INT64, then encodes sum and count into a
// single STRING column of fixed-width per-row payloads (delegates to
// serializeDecimalSumState). Used when emitting or persisting partial /
// intermediate decimal SUM state for the cuDF path.
std::unique_ptr<cudf::column> serializeDecimalPartialOrIntermediateState(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream);

// Normalizes the count column to INT64, computes a per-row decimal average
// from intermediate sum/count (delegates to computeDecimalAverage), then casts
// the result to the Velox result type when its cuDF decimal encoding differs
// from the average column's type.
std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox
