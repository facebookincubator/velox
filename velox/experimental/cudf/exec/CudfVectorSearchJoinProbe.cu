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

#include "velox/experimental/cudf/exec/CudfVectorSearchJoinProbeKernels.h"

#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <thrust/sort.h>

namespace facebook::velox::cudf_velox::vector_search {
namespace {

constexpr int32_t kThreads = 128;

__device__ bool isValid(const uint32_t* mask, Size row) {
  return mask == nullptr || (mask[row / 32] & (uint32_t{1} << (row % 32)));
}

__global__ void selectRowsKernel(
    Size count,
    Size first,
    const Size* rowOffsets,
    Size inputOffset,
    const uint32_t* embeddingMask,
    const bool* hasMatch,
    const uint32_t* matchMask,
    const int32_t* hasResults,
    Size* rows,
    bool* selected) {
  const auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    rows[i] = first + (rowOffsets ? rowOffsets[i] : i);
    if (selected != nullptr) {
      selected[i] = hasResults != nullptr
          ? hasResults[i] == 0
          : isValid(embeddingMask, inputOffset + rows[i]) &&
              (hasMatch == nullptr || (isValid(matchMask, i) && hasMatch[i]));
    }
  }
}

__global__ void normalizeKernel(
    const Size* rows,
    const Size* offsets,
    const float* elements,
    const uint32_t* elementMask,
    Size elementOffset,
    Size dimension,
    float* queries,
    int32_t* validation) {
  const auto row = rows[blockIdx.x];
  const auto offset = offsets[row];
  if (offsets[row + 1] - offset != dimension) {
    if (threadIdx.x == 0) {
      atomicOr(validation, kInvalidLength);
    }
    return;
  }
  float sum = 0;
  for (Size i = threadIdx.x; i < dimension; i += blockDim.x) {
    if (!isValid(elementMask, elementOffset + offset + i)) {
      atomicOr(validation, kNullElement);
      continue;
    }
    const float value = elements[offset + i];
    sum += value * value;
  }
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float scale;
  const float norm = Reduce(storage).Sum(sum);
  if (threadIdx.x == 0) {
    scale = norm > 0 ? 1.0f / sqrtf(norm) : 1.0f;
  }
  __syncthreads();
  for (Size i = threadIdx.x; i < dimension; i += blockDim.x) {
    queries[static_cast<int64_t>(blockIdx.x) * dimension + i] =
        elements[offset + i] * scale;
  }
}

__global__ void mapResultsKernel(
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
    int32_t* hasResults) {
  const auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  auto id = ids[i];
  if (id != paddingId && idMap != nullptr) {
    id = id >= 0 && id < indexSize ? idMap[id] : paddingId;
    ids[i] = id;
  }
  rows[i] = queries[i / k];
  buildRows[i] = buildRow;
  selected[i] = id != paddingId;
  if (hasResults != nullptr && selected[i]) {
    atomicExch(hasResults + rows[i] - sliceStart, 1);
  }
}

int64_t blocks(Size rows) {
  return (int64_t{rows} + kThreads - 1) / kThreads;
}

__global__ void countMatches(Size count, const Size* build, Size* counts) {
  const auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    atomicAdd(counts + build[i], 1);
  }
}

__global__ void gatherQueriesKernel(
    Size dimension,
    const Size* indices,
    const Size* rows,
    const float* queries,
    Size* selectedRows,
    float* selectedQueries) {
  const auto i = blockIdx.x;
  const auto source = indices[i];
  if (threadIdx.x == 0) {
    selectedRows[i] = rows[source];
  }
  for (Size dim = threadIdx.x; dim < dimension; dim += blockDim.x) {
    selectedQueries[static_cast<int64_t>(i) * dimension + dim] =
        queries[static_cast<int64_t>(source) * dimension + dim];
  }
}

} // namespace

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
    cudaStream_t stream) {
  selectRowsKernel<<<blocks(count), kThreads, 0, stream>>>(
      count,
      first,
      rowOffsets,
      inputOffset,
      embeddingMask,
      hasMatch,
      matchMask,
      hasResults,
      rows,
      selected);
  CUDF_CUDA_TRY(cudaGetLastError());
}

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
    cudaStream_t stream) {
  normalizeKernel<<<count, kThreads, 0, stream>>>(
      rows,
      offsets,
      elements,
      elementMask,
      elementOffset,
      dimension,
      queries,
      validation);
  CUDF_CUDA_TRY(cudaGetLastError());
}

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
    cudaStream_t stream) {
  mapResultsKernel<<<blocks(count), kThreads, 0, stream>>>(
      count,
      k,
      sliceStart,
      buildRow,
      queries,
      idMap,
      indexSize,
      paddingId,
      ids,
      rows,
      buildRows,
      selected,
      hasResults);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void groupMatches(
    Size count,
    Size* build,
    Size* probe,
    Size* counts,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  thrust::sort_by_key(
      rmm::exec_policy_nosync{stream, mr}, build, build + count, probe);
  countMatches<<<blocks(count), kThreads, 0, stream.get()>>>(
      count, build, counts);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void gatherQueries(
    Size count,
    Size dimension,
    const Size* indices,
    const Size* rows,
    const float* queries,
    Size* selectedRows,
    float* selectedQueries,
    cudaStream_t stream) {
  gatherQueriesKernel<<<count, kThreads, 0, stream>>>(
      dimension, indices, rows, queries, selectedRows, selectedQueries);
  CUDF_CUDA_TRY(cudaGetLastError());
}

} // namespace facebook::velox::cudf_velox::vector_search
