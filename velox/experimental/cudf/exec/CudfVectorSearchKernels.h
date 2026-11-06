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

#include <cudf/types.hpp>

#include <cuda/stream>

#include <cstdint>
#include <vector>

namespace facebook::velox::cudf_velox::vector_search {

/// The number of the 'numLists' lists, given by 'numLists' + 1 'offsets', that
/// do not hold 'dimension' values.
int64_t countWrongLengths(
    const cudf::size_type* offsets,
    cudf::size_type numLists,
    int32_t dimension,
    cuda::stream_ref stream);

/// Scales each of the 'numRows' rows of 'dimension' floats in 'values' to
/// length 1, leaving a zero row as it is, as faiss::fvec_renorm_L2 does.
void normalizeRows(
    float* values,
    int64_t numRows,
    int32_t dimension,
    cuda::stream_ref stream);

/// For 'numRuns' runs numbered from 0, the position in 'sortedRuns', 'size'
/// run numbers in ascending order, where each run starts, then 'size'.
std::vector<cudf::size_type> runStarts(
    const cudf::size_type* sortedRuns,
    cudf::size_type size,
    cudf::size_type numRuns,
    cuda::stream_ref stream);

/// Turns the 'numQueries' x 'k' results of a device search into join
/// results. Maps each id through 'idMap', of 'idMapSize' ids, if set, and
/// writes it back to 'ids'. Writes to 'probeRows' the probe row of each
/// result's query, from 'queryProbeRows', and to 'found' whether the result
/// is an embedding rather than padding. Sets 'hasResults', if set, for the
/// probe row of each found result.
void emitResults(
    int64_t* ids,
    int64_t numQueries,
    int64_t k,
    const int64_t* idMap,
    int64_t idMapSize,
    const cudf::size_type* queryProbeRows,
    cudf::size_type* probeRows,
    bool* found,
    bool* hasResults,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox::vector_search
