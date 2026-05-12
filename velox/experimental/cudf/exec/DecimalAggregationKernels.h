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

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

struct DecimalSumStateColumns {
  std::unique_ptr<cudf::column> sum;
  std::unique_ptr<cudf::column> count;
};

// Decodes intermediate decimal SUM aggregate state stored as a cuDF STRING
// column (fixed-size packed bytes per row, converted from Velox VARBINARY) into
// two device columns: a DECIMAL128 sum with scale -scale (matching Velox
// intermediate state) and an INT64 partial row count. Handles empty input,
// all-null state without touching payload buffers, and propagates the source
// null mask to both outputs when present.
DecimalSumStateColumns deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Encodes partial decimal SUM state (DECIMAL64 or DECIMAL128 sums plus
// INT64 counts) into a single STRING column (later converted to Velox
// VARBINARY): per-row fixed-width payloads and string offsets (INT32 or INT64
// depending on total char size and cuDF large-strings settings). The output
// null mask matches buildStateValidityMask: a row is invalid if the sum or
// count is null, or the count is zero.
std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Finalizes AVG from intermediate SUM state: divides each sum by its count
// on device with decimal-specific rounding (see averageRoundDecimalSum),
// producing a column of the same decimal type as the sum. Rows are null
// where buildStateValidityMask marks them invalid (null sum/count or zero
// count), matching serializeDecimalSumState.
std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
