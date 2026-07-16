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

/**
 * Asserts that a column holds serialized decimal aggregate state in the form
 * Velox uses for VARBINARY: a cuDF STRING column whose bytes are the packed
 * sum/count payloads (see serializeDecimalSumState). The payload does not carry
 * scale, so VARBINARY intermediate steps decode at scale 0; the real scale is
 * applied at final cast time.
 *
 * @param column column to validate.
 */
void validateIntermediateColumnType(cudf::column_view const& column);

/**
 * Casts a DECIMAL64 column up to DECIMAL128 (scale preserved) so a subsequent
 * SUM accumulates in 128 bits instead of wrapping. Allocates the casted column
 * from the temporary memory resource into holder and returns its view. Lifetime
 * stays valid only while holder is alive.
 *
 * @param inputCol DECIMAL64 input column.
 * @param holder receives ownership of the casted column when inputCol is
 *        DECIMAL64; unchanged otherwise.
 * @param stream CUDA stream for device work.
 * @return view of inputCol or of the column stored in holder.
 */
cudf::column_view castDecimal64InputToDecimal128(
    cudf::column_view inputCol,
    std::unique_ptr<cudf::column>& holder,
    rmm::cuda_stream_view stream);

/**
 * Ensures the partial-row count column is INT64, casting with the temporary
 * memory resource (the result is consumed internally, not part of operator
 * output) when the incoming type differs.
 *
 * @param count partial-row count column.
 * @param stream CUDA stream for device work.
 * @return INT64 count column (moved through when already INT64).
 */
std::unique_ptr<cudf::column> castCountColumnToInt64(
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream);

/**
 * Normalizes the count column to INT64, then encodes sum and count into a
 * single STRING column of fixed-width per-row payloads (delegates to
 * serializeDecimalSumState). Used when emitting or persisting partial /
 * intermediate decimal SUM state for the cuDF path.
 *
 * @param sum partial sum column (DECIMAL64 or DECIMAL128).
 * @param count partial-row count column.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return STRING column of serialized state.
 */
std::unique_ptr<cudf::column> serializeDecimalPartialOrIntermediateState(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * Normalizes the count column to INT64, computes a per-row decimal average
 * from intermediate sum/count (delegates to computeDecimalAverage), then casts
 * the result to the Velox result type when its cuDF decimal encoding differs
 * from the average column's type.
 *
 * @param sum intermediate sum column (DECIMAL64 or DECIMAL128).
 * @param count intermediate count column.
 * @param resultType Velox type of the finalized average.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return finalized average column.
 */
std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
