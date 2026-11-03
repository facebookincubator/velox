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

#include "velox/experimental/cudf/exec/CudfVectorSearchKernels.h"

#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <limits>

namespace facebook::velox::cudf_velox::vector_search {
namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 8;

struct WrongLength {
  const cudf::size_type* offsets;
  int32_t dimension;

  __device__ bool operator()(cudf::size_type i) const {
    return offsets[i + 1] - offsets[i] != dimension;
  }
};

// One warp per row: sums the squares, then scales by the inverse norm.
__global__ void normalizeRowsKernel(
    float* values,
    int64_t numRows,
    int32_t dimension) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + threadIdx.y;
  if (row >= numRows) {
    return;
  }
  float* rowValues = values + row * dimension;
  float sum = 0;
  for (int32_t i = threadIdx.x; i < dimension; i += kWarpSize) {
    sum += rowValues[i] * rowValues[i];
  }
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  if (sum <= 0) {
    return;
  }
  const float inverse = 1.0 / sqrtf(sum);
  for (int32_t i = threadIdx.x; i < dimension; i += kWarpSize) {
    rowValues[i] *= inverse;
  }
}

__global__ void emitResultsKernel(
    int64_t* ids,
    int64_t numQueries,
    int64_t k,
    const int64_t* idMap,
    int64_t idMapSize,
    const cudf::size_type* queryProbeRows,
    cudf::size_type* probeRows,
    bool* found,
    bool* hasResults) {
  const int64_t numResults = numQueries * k;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < numResults;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const auto probeRow = queryProbeRows[i / k];
    probeRows[i] = probeRow;
    auto id = ids[i];
    bool isFound;
    if (idMap != nullptr) {
      isFound = id >= 0 && id < idMapSize;
      if (isFound) {
        id = idMap[id];
      }
    } else {
      // cuVS writes the largest id where FAISS writes -1.
      isFound = id >= 0 && id != std::numeric_limits<int64_t>::max();
    }
    ids[i] = isFound ? id : -1;
    found[i] = isFound;
    if (isFound && hasResults != nullptr) {
      hasResults[probeRow] = true;
    }
  }
}

} // namespace

int64_t countWrongLengths(
    const cudf::size_type* offsets,
    cudf::size_type numLists,
    int32_t dimension,
    cuda::stream_ref stream) {
  if (numLists == 0) {
    return 0;
  }
  return thrust::count_if(
      rmm::exec_policy(stream, rmm::mr::get_current_device_resource_ref()),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(numLists),
      WrongLength{offsets, dimension});
}

void normalizeRows(
    float* values,
    int64_t numRows,
    int32_t dimension,
    cuda::stream_ref stream) {
  if (numRows == 0) {
    return;
  }
  const dim3 block(kWarpSize, kRowsPerBlock);
  const auto blocks = (numRows + kRowsPerBlock - 1) / kRowsPerBlock;
  normalizeRowsKernel<<<blocks, block, 0, stream.get()>>>(
      values, numRows, dimension);
  CUDF_CUDA_TRY(cudaGetLastError());
}

std::vector<cudf::size_type> runStarts(
    const cudf::size_type* sortedRuns,
    cudf::size_type size,
    cudf::size_type numRuns,
    cuda::stream_ref stream) {
  rmm::device_uvector<cudf::size_type> starts(
      numRuns, stream, rmm::mr::get_current_device_resource_ref());
  thrust::lower_bound(
      rmm::exec_policy_nosync(
          stream, rmm::mr::get_current_device_resource_ref()),
      sortedRuns,
      sortedRuns + size,
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(numRuns),
      starts.begin());
  std::vector<cudf::size_type> result(numRuns + 1);
  if (numRuns > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        result.data(),
        starts.data(),
        numRuns * sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
  }
  result[numRuns] = size;
  return result;
}

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
    cuda::stream_ref stream) {
  const int64_t numResults = numQueries * k;
  if (numResults == 0) {
    return;
  }
  constexpr int kThreads = 256;
  const auto blocks = static_cast<int>(
      std::min<int64_t>((numResults + kThreads - 1) / kThreads, 65535));
  emitResultsKernel<<<blocks, kThreads, 0, stream.get()>>>(
      ids,
      numQueries,
      k,
      idMap,
      idMapSize,
      queryProbeRows,
      probeRows,
      found,
      hasResults);
  CUDF_CUDA_TRY(cudaGetLastError());
}

} // namespace facebook::velox::cudf_velox::vector_search
