/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>

namespace facebook::velox::cudf_velox::vector_search {

using Size = int32_t;
constexpr int32_t kInvalidLength = 1;
constexpr int32_t kNullElement = 2;

void selectRows(
    Size count,
    Size first,
    const Size* rowOffsets,
    Size inputOffset,
    const uint32_t* embeddingMask,
    const bool* hasMatch,
    const uint32_t* matchMask,
    const int32_t* hasResults,
    Size* rows,
    bool* selected,
    cudaStream_t stream);

void normalize(
    Size count,
    const Size* rows,
    const Size* offsets,
    const float* elements,
    const uint32_t* elementMask,
    Size elementOffset,
    Size dimension,
    float* queries,
    int32_t* validation,
    cudaStream_t stream);

void mapResults(
    Size count,
    Size k,
    Size sliceStart,
    Size buildRow,
    const Size* queries,
    const int64_t* idMap,
    int64_t indexSize,
    int64_t paddingId,
    int64_t* ids,
    Size* rows,
    Size* buildRows,
    bool* selected,
    int32_t* hasResults,
    cudaStream_t stream);

void groupMatches(
    Size count,
    Size* build,
    Size* probe,
    Size* counts,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

void gatherQueries(
    Size count,
    Size dimension,
    const Size* indices,
    const Size* rows,
    const float* queries,
    Size* selectedRows,
    float* selectedQueries,
    cudaStream_t stream);

} // namespace facebook::velox::cudf_velox::vector_search
