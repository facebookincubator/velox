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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/RuntimeVectorIndex.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/device_memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <thrust/scan.h>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace facebook::velox::cudf_velox {

[[noreturn]] void failRuntime(const std::string& message);
[[noreturn]] void failUser(const std::string& message);

namespace {

constexpr int32_t kMaxCandidateK{2048};

struct BatchDevice {
  cudf::column_device_view vectors;
  cudf::column_device_view buildIds;
  int64_t rowStart;
  int32_t rows;
};

using DeviceColumn = std::unique_ptr<
    cudf::column_device_view,
    std::function<void(cudf::column_device_view*)>>;

struct Classified {
  int32_t validRows{0};
  int32_t dimension{-1};
  rmm::device_uvector<int32_t> valid;
  rmm::device_uvector<int32_t> positions;

  Classified(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
      : valid(0, stream, mr), positions(0, stream, mr) {}
};

void checkCuda(cudaError_t error) {
  if (error != cudaSuccess) {
    failRuntime(std::string("CUDA error: ") + cudaGetErrorString(error));
  }
}

raft::device_resources makeResources(
    cuda::stream_ref stream,
    std::size_t workspaceBytes) {
  raft::device_resources resources(stream);
  raft::resource::set_workspace_to_global_resource(resources, workspaceBytes);
  return resources;
}

__global__ void classifyRows(
    cudf::lists_column_device_view lists,
    int64_t rowStart,
    int32_t numRows,
    int32_t expectedDimension,
    bool cosine,
    int32_t* valid,
    int32_t* dimension,
    int32_t* mismatch) {
  const auto row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= numRows) {
    return;
  }
  if (lists.is_null(row)) {
    valid[row] = 0;
    return;
  }
  const auto start = lists.offset_at(row);
  const auto size = lists.offset_at(row + 1) - start;
  const auto child = lists.child();
  bool nullElement = false;
  for (int32_t i = 0; i < size; ++i) {
    nullElement |= child.is_null(start + i);
  }
  if (nullElement || size <= 0) {
    valid[row] = 0;
    return;
  }
  const int32_t seen = atomicCAS(dimension, -1, size);
  if ((seen != -1 && seen != size) ||
      (expectedDimension >= 0 && size != expectedDimension)) {
    atomicExch(mismatch, 1);
    valid[row] = 0;
    return;
  }
  if (cosine) {
    float squares = 0;
    for (int32_t i = 0; i < size; ++i) {
      const auto value = child.element<float>(start + i);
      squares += value * value;
    }
    if (squares == 0) {
      valid[row] = 0;
      return;
    }
  }
  valid[row] = 1;
  (void)rowStart;
}

__global__ void scatterPacked(
    cudf::lists_column_device_view lists,
    int64_t rowStart,
    int32_t numRows,
    int32_t dimension,
    const int32_t* valid,
    const int32_t* positions,
    float* packed,
    int64_t* rowIds) {
  const auto row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= numRows || valid[row] == 0) {
    return;
  }
  const auto position = positions[row];
  const auto start = lists.offset_at(row);
  const auto child = lists.child();
  auto* destination = packed + static_cast<int64_t>(position) * dimension;
  for (int32_t i = 0; i < dimension; ++i) {
    destination[i] = child.element<float>(start + i);
  }
  rowIds[position] = rowStart + row;
}

__global__ void normalizeRows(float* values, int64_t rows, int32_t dimension) {
  const auto row = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) {
    return;
  }
  auto* vector = values + row * dimension;
  float squares = 0;
  for (int32_t i = 0; i < dimension; ++i) {
    squares += vector[i] * vector[i];
  }
  const auto inverse = 1.0f / cuda::std::sqrt(squares);
  for (int32_t i = 0; i < dimension; ++i) {
    vector[i] *= inverse;
  }
}

__device__ const BatchDevice*
findBatch(const BatchDevice* batches, int32_t numBatches, int64_t rowId) {
  int32_t low = 0;
  int32_t high = numBatches;
  while (low + 1 < high) {
    const auto mid = (low + high) / 2;
    if (batches[mid].rowStart <= rowId) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return &batches[low];
}

__device__ bool betterHit(
    float score,
    int64_t buildId,
    float otherScore,
    int64_t otherBuildId,
    bool largerIsBetter,
    bool otherEmpty) {
  if (otherEmpty) {
    return true;
  }
  if (score == otherScore) {
    return buildId < otherBuildId;
  }
  return largerIsBetter ? score > otherScore : score < otherScore;
}

__device__ float exactScore(
    const float* query,
    const BatchDevice& batch,
    int32_t localRow,
    int32_t dimension,
    bool cosine,
    bool largerIsBetter,
    int64_t* buildId) {
  auto lists = cudf::lists_column_device_view(batch.vectors);
  const auto start = lists.offset_at(localRow);
  const auto child = lists.child();
  float dot = 0;
  float leftSquares = 0;
  float rightSquares = 0;
  float squared = 0;
  for (int32_t i = 0; i < dimension; ++i) {
    const auto left = query[i];
    const auto right = child.element<float>(start + i);
    const auto difference = left - right;
    dot += left * right;
    leftSquares += left * left;
    rightSquares += right * right;
    squared += difference * difference;
  }
  if (batch.buildIds.is_null(localRow)) {
    *buildId = cuda::std::numeric_limits<int64_t>::max();
  } else {
    *buildId = batch.buildIds.element<int64_t>(localRow);
  }
  if (cosine) {
    const auto denominator =
        cuda::std::sqrt(leftSquares) * cuda::std::sqrt(rightSquares);
    if (denominator == 0) {
      return -cuda::std::numeric_limits<float>::infinity();
    }
    return dot / denominator;
  }
  return largerIsBetter ? dot : squared;
}

__global__ void initHits(
    int64_t numQueries,
    int32_t k,
    bool largerIsBetter,
    const int32_t* queryRows,
    VectorJoinHit* hits) {
  const auto index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= numQueries * k) {
    return;
  }
  hits[index].probeRow = queryRows[index / k];
  hits[index].buildRow = -1;
  hits[index].buildId = cuda::std::numeric_limits<int64_t>::max();
  hits[index].score = largerIsBetter
      ? -cuda::std::numeric_limits<float>::infinity()
      : cuda::std::numeric_limits<float>::infinity();
}

__device__ void consider(
    VectorJoinHit* hits,
    int32_t k,
    int64_t query,
    int32_t probeRow,
    int64_t buildRow,
    int64_t buildId,
    float score,
    bool largerIsBetter) {
  if (!cuda::std::isfinite(score)) {
    return;
  }
  auto* rowHits = hits + query * k;
  for (int32_t i = 0; i < k; ++i) {
    if (rowHits[i].buildRow == buildRow) {
      return;
    }
  }
  int32_t slot = k;
  for (int32_t i = 0; i < k; ++i) {
    if (betterHit(
            score,
            buildId,
            rowHits[i].score,
            rowHits[i].buildId,
            largerIsBetter,
            rowHits[i].buildRow < 0)) {
      slot = i;
      break;
    }
  }
  if (slot == k) {
    return;
  }
  for (int32_t i = k - 1; i > slot; --i) {
    rowHits[i] = rowHits[i - 1];
  }
  rowHits[slot] = VectorJoinHit{probeRow, buildRow, buildId, score};
}

__global__ void rerankCandidates(
    const float* queries,
    const int32_t* queryRows,
    int64_t numQueries,
    const int64_t* candidates,
    int64_t numCandidates,
    bool candidatesPerQuery,
    const BatchDevice* batches,
    int32_t numBatches,
    int32_t dimension,
    int32_t k,
    bool cosine,
    bool largerIsBetter,
    VectorJoinHit* hits) {
  const auto query =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query >= numQueries) {
    return;
  }
  const auto* queryVector = queries + query * dimension;
  const auto width = candidatesPerQuery ? numCandidates : numCandidates;
  for (int64_t candidate = 0; candidate < width; ++candidate) {
    const auto buildRow = candidatesPerQuery
        ? candidates[query * numCandidates + candidate]
        : candidates[candidate];
    if (buildRow < 0) {
      continue;
    }
    const auto* batch = findBatch(batches, numBatches, buildRow);
    const auto local = static_cast<int32_t>(buildRow - batch->rowStart);
    if (local < 0 || local >= batch->rows) {
      continue;
    }
    int64_t buildId = 0;
    const auto score = exactScore(
        queryVector,
        *batch,
        local,
        dimension,
        cosine,
        largerIsBetter,
        &buildId);
    consider(
        hits,
        k,
        query,
        queryRows[query],
        buildRow,
        buildId,
        score,
        largerIsBetter);
  }
}

int32_t launchWidth(int64_t n) {
  return static_cast<int32_t>(std::min<int64_t>(n, 256));
}

int32_t launchGrid(int64_t n, int32_t block) {
  return static_cast<int32_t>((n + block - 1) / block);
}

Classified classifyBatch(
    const cudf::lists_column_view& lists,
    int64_t rowStart,
    int32_t expectedDimension,
    bool cosine,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto numRows = lists.size();
  Classified classified(stream, mr);
  if (numRows == 0) {
    return classified;
  }
  classified.valid = rmm::device_uvector<int32_t>(numRows, stream, mr);
  classified.positions = rmm::device_uvector<int32_t>(numRows, stream, mr);
  auto view = cudf::column_device_view::create(lists.parent(), stream, mr);
  cudf::detail::device_scalar<int32_t> dimension{-1, stream, mr};
  cudf::detail::device_scalar<int32_t> mismatch{0, stream, mr};
  const auto block = launchWidth(numRows);
  classifyRows<<<launchGrid(numRows, block), block, 0, stream.get()>>>(
      cudf::lists_column_device_view(*view),
      rowStart,
      numRows,
      expectedDimension,
      cosine,
      classified.valid.data(),
      dimension.data(),
      mismatch.data());
  checkCuda(cudaGetLastError());
  if (mismatch.value(stream) != 0) {
    failUser("Vector join inputs must be ARRAY(REAL) values of one dimension");
  }
  const auto foundDimension = dimension.value(stream);
  if (foundDimension >= 0) {
    classified.dimension = foundDimension;
  }
  thrust::exclusive_scan(
      rmm::exec_policy_nosync(stream, mr),
      classified.valid.begin(),
      classified.valid.end(),
      classified.positions.begin());
  int32_t lastValid = 0;
  int32_t lastPosition = 0;
  checkCuda(cudaMemcpyAsync(
      &lastValid,
      classified.valid.data() + numRows - 1,
      sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.get()));
  checkCuda(cudaMemcpyAsync(
      &lastPosition,
      classified.positions.data() + numRows - 1,
      sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  classified.validRows = lastPosition + lastValid;
  return classified;
}

struct DenseBatch {
  rmm::device_uvector<float> values;
  rmm::device_uvector<int64_t> rowIds;
  int64_t rows{0};

  DenseBatch(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
      : values(0, stream, mr), rowIds(0, stream, mr) {}
};

DenseBatch packBatch(
    const cudf::lists_column_view& lists,
    const Classified& classified,
    int64_t rowStart,
    int32_t dimension,
    bool cosine,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  DenseBatch packed(stream, mr);
  packed.rows = classified.validRows;
  if (packed.rows == 0) {
    return packed;
  }
  packed.values =
      rmm::device_uvector<float>(packed.rows * dimension, stream, mr);
  packed.rowIds = rmm::device_uvector<int64_t>(packed.rows, stream, mr);
  auto view = cudf::column_device_view::create(lists.parent(), stream, mr);
  const auto block = launchWidth(lists.size());
  scatterPacked<<<launchGrid(lists.size(), block), block, 0, stream.get()>>>(
      cudf::lists_column_device_view(*view),
      rowStart,
      lists.size(),
      dimension,
      classified.valid.data(),
      classified.positions.data(),
      packed.values.data(),
      packed.rowIds.data());
  checkCuda(cudaGetLastError());
  if (cosine) {
    const auto normBlock = launchWidth(packed.rows);
    normalizeRows<<<
        launchGrid(packed.rows, normBlock),
        normBlock,
        0,
        stream.get()>>>(packed.values.data(), packed.rows, dimension);
    checkCuda(cudaGetLastError());
  }
  return packed;
}

int32_t choosePqDim(int32_t dimension, int32_t pqBits, int32_t requested) {
  if (pqBits == 8) {
    return requested > 0 ? requested : std::max<int32_t>(dimension, 1);
  }
  // pq_dim must be a multiple of 8 when the code width is not a byte.
  auto dim = requested > 0 ? requested : dimension;
  if (dim % 8 != 0) {
    dim += 8 - (dim % 8);
  }
  return std::max<int32_t>(dim, 8);
}

} // namespace

struct RuntimeVectorIndex::Impl {
  raft::device_resources resources;
  std::optional<cuvs::neighbors::ivf_pq::index<int64_t>> ivf;
  std::vector<DeviceColumn> vectorViews;
  std::vector<DeviceColumn> buildIdViews;
  rmm::device_uvector<BatchDevice> batches;
  rmm::device_uvector<int64_t> bruteIds;
  int64_t indexedRows{0};
  int32_t dimension{0};
  int32_t nLists{1};
  bool approximate{false};
  bool cosine{false};
  bool largerIsBetter{false};
  mutable std::mutex mutex;

  Impl(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
      : resources(makeResources(stream, std::size_t{1} << 28)),
        batches(0, stream, mr),
        bruteIds(0, stream, mr) {}
};

RuntimeVectorIndex::RuntimeVectorIndex(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RuntimeVectorIndex::~RuntimeVectorIndex() = default;

int64_t RuntimeVectorIndex::indexedRows() const {
  return impl_->indexedRows;
}

int32_t RuntimeVectorIndex::dimension() const {
  return impl_->dimension;
}

int32_t RuntimeVectorIndex::nLists() const {
  return impl_->nLists;
}

bool RuntimeVectorIndex::approximate() const {
  return impl_->approximate;
}

std::vector<VectorJoinHit> RuntimeVectorIndex::search(
    const float* queries,
    const int32_t* queryRows,
    int64_t numQueries,
    int32_t k,
    int32_t candidateK,
    int32_t nProbes,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  if (k <= 0) {
    failRuntime("Vector join k must be positive");
  }
  std::vector<VectorJoinHit> hits(static_cast<size_t>(numQueries) * k);
  if (numQueries == 0 || impl_->indexedRows == 0 || k == 0) {
    return hits;
  }
  const std::lock_guard<std::mutex> guard{impl_->mutex};
  auto resources = makeResources(stream, std::size_t{1} << 28);
  rmm::device_uvector<VectorJoinHit> deviceHits(hits.size(), stream, mr);
  const auto block = launchWidth(deviceHits.size());
  initHits<<<launchGrid(deviceHits.size(), block), block, 0, stream.get()>>>(
      numQueries, k, impl_->largerIsBetter, queryRows, deviceHits.data());
  checkCuda(cudaGetLastError());

  const int64_t* candidates = nullptr;
  int64_t numCandidates = 0;
  bool perQuery = false;
  rmm::device_uvector<int64_t> ivfNeighbors(0, stream, mr);
  std::unique_ptr<rmm::device_uvector<float>> normalized;
  const float* searchQueries = queries;
  if (impl_->cosine) {
    normalized = std::make_unique<rmm::device_uvector<float>>(
        numQueries * impl_->dimension, stream, mr);
    checkCuda(cudaMemcpyAsync(
        normalized->data(),
        queries,
        sizeof(float) * normalized->size(),
        cudaMemcpyDeviceToDevice,
        stream.get()));
    const auto normBlock = launchWidth(numQueries);
    normalizeRows<<<
        launchGrid(numQueries, normBlock),
        normBlock,
        0,
        stream.get()>>>(normalized->data(), numQueries, impl_->dimension);
    checkCuda(cudaGetLastError());
    searchQueries = normalized->data();
  }

  if (impl_->approximate) {
    candidateK = static_cast<int32_t>(std::min<int64_t>(
        std::max<int32_t>(candidateK, k), impl_->indexedRows));
    candidateK = std::min(candidateK, kMaxCandidateK);
    ivfNeighbors =
        rmm::device_uvector<int64_t>(numQueries * candidateK, stream, mr);
    rmm::device_uvector<float> distances(ivfNeighbors.size(), stream, mr);
    cuvs::neighbors::ivf_pq::search_params params;
    params.n_probes =
        static_cast<uint32_t>(std::max(1, std::min(nProbes, impl_->nLists)));
    auto queryView =
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
            searchQueries, numQueries, impl_->dimension);
    auto neighborView =
        raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
            ivfNeighbors.data(), numQueries, candidateK);
    auto distanceView =
        raft::make_device_matrix_view<float, int64_t, raft::row_major>(
            distances.data(), numQueries, candidateK);
    try {
      cuvs::neighbors::ivf_pq::search(
          resources,
          params,
          *impl_->ivf,
          queryView,
          neighborView,
          distanceView);
    } catch (const std::exception& error) {
      failRuntime(std::string("cuVS IVF-PQ search failed: ") + error.what());
    }
    candidates = ivfNeighbors.data();
    numCandidates = candidateK;
    perQuery = true;
    searchQueries = queries;
  } else {
    candidates = impl_->bruteIds.data();
    numCandidates = impl_->indexedRows;
    perQuery = false;
    searchQueries = queries;
  }

  const auto queryBlock = launchWidth(numQueries);
  rerankCandidates<<<
      launchGrid(numQueries, queryBlock),
      queryBlock,
      0,
      stream.get()>>>(
      searchQueries,
      queryRows,
      numQueries,
      candidates,
      numCandidates,
      perQuery,
      impl_->batches.data(),
      static_cast<int32_t>(impl_->batches.size()),
      impl_->dimension,
      k,
      impl_->cosine,
      impl_->largerIsBetter,
      deviceHits.data());
  checkCuda(cudaGetLastError());
  checkCuda(cudaMemcpyAsync(
      hits.data(),
      deviceHits.data(),
      sizeof(VectorJoinHit) * hits.size(),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  return hits;
}

PackedVectors packQueryVectors(
    const cudf::column_view& vectors,
    int32_t expectedDimension,
    bool cosine,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  cudf::lists_column_view lists(vectors);
  if (lists.child().type().id() != cudf::type_id::FLOAT32) {
    failUser("Vector join supports ARRAY(REAL) vectors");
  }
  auto classified =
      classifyBatch(lists, 0, expectedDimension, cosine, stream, mr);
  if (!(classified.dimension < 0 || classified.dimension == expectedDimension ||
        expectedDimension < 0)) {
    failUser("Vector join inputs must be ARRAY(REAL) values of one dimension");
  }
  auto packed = packBatch(
      lists,
      classified,
      0,
      expectedDimension >= 0 ? expectedDimension : classified.dimension,
      false,
      stream,
      mr);
  PackedVectors result;
  result.count = packed.rows;
  result.dimension =
      expectedDimension >= 0 ? expectedDimension : classified.dimension;
  if (packed.rows == 0) {
    return result;
  }
  result.values =
      std::make_unique<rmm::device_uvector<float>>(std::move(packed.values));
  std::vector<int64_t> hostIds(static_cast<size_t>(packed.rows));
  checkCuda(cudaMemcpyAsync(
      hostIds.data(),
      packed.rowIds.data(),
      sizeof(int64_t) * hostIds.size(),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  std::vector<int32_t> hostRows(hostIds.size());
  for (size_t i = 0; i < hostIds.size(); ++i) {
    hostRows[i] = static_cast<int32_t>(hostIds[i]);
  }
  result.rowIds =
      std::make_unique<rmm::device_uvector<int32_t>>(packed.rows, stream, mr);
  checkCuda(cudaMemcpyAsync(
      result.rowIds->data(),
      hostRows.data(),
      sizeof(int32_t) * hostRows.size(),
      cudaMemcpyHostToDevice,
      stream.get()));
  return result;
}

std::unique_ptr<RuntimeVectorIndex> buildRuntimeVectorIndex(
    const std::vector<RuntimeVectorIndex::Batch>& batches,
    const RuntimeVectorIndex::Options& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto impl = std::make_unique<RuntimeVectorIndex::Impl>(stream, mr);
  impl->cosine = options.cosine;
  impl->largerIsBetter = options.largerIsBetter;
  std::vector<Classified> classified;
  classified.reserve(batches.size());
  int32_t dimension = -1;
  int64_t validRows = 0;
  for (const auto& batch : batches) {
    cudf::lists_column_view lists(batch.vectors);
    if (lists.child().type().id() != cudf::type_id::FLOAT32) {
      failUser("Vector join supports ARRAY(REAL) vectors");
    }
    if (batch.buildIds.type().id() != cudf::type_id::INT64) {
      failUser("Vector join build id must be BIGINT");
    }
    auto count = classifyBatch(
        lists, batch.rowStart, dimension, options.cosine, stream, mr);
    if (count.dimension >= 0) {
      dimension = count.dimension;
    }
    validRows += count.validRows;
    classified.push_back(std::move(count));
  }
  impl->dimension = std::max(dimension, 0);
  impl->indexedRows = validRows;
  if (validRows == 0) {
    return std::unique_ptr<RuntimeVectorIndex>(
        new RuntimeVectorIndex(std::move(impl)));
  }

  std::vector<BatchDevice> hostBatches;
  hostBatches.reserve(batches.size());
  for (const auto& batch : batches) {
    impl->vectorViews.push_back(
        cudf::column_device_view::create(batch.vectors, stream, mr));
    impl->buildIdViews.push_back(
        cudf::column_device_view::create(batch.buildIds, stream, mr));
    hostBatches.push_back(
        BatchDevice{
            *impl->vectorViews.back(),
            *impl->buildIdViews.back(),
            batch.rowStart,
            batch.vectors.size()});
  }
  impl->batches =
      rmm::device_uvector<BatchDevice>(hostBatches.size(), stream, mr);
  checkCuda(cudaMemcpyAsync(
      impl->batches.data(),
      hostBatches.data(),
      sizeof(BatchDevice) * hostBatches.size(),
      cudaMemcpyHostToDevice,
      stream.get()));

  const auto requestedLists = std::max<int32_t>(options.nLists, 1);
  auto pqBits = options.pqBits == 0 ? 8 : options.pqBits;
  if (pqBits < 4) {
    pqBits = 4;
  } else if (pqBits > 8) {
    pqBits = 8;
  }
  // Each PQ codebook has 2^pq_bits centroids and cannot train on fewer rows.
  while (pqBits > 4 && validRows < (int64_t{1} << pqBits)) {
    --pqBits;
  }
  const auto codebookRows = int64_t{1} << pqBits;
  const auto sampleCapacity =
      std::min<int64_t>(std::max<int32_t>(options.trainRows, 1), validRows);
  // Compare against the requested list count. Shrinking n_lists to
  // validRows/8 would make every shard of two or more rows look large
  // enough for IVF-PQ.
  const bool approximate = impl->dimension > 0 && validRows >= 2 &&
      validRows >= static_cast<int64_t>(requestedLists) * 2 &&
      sampleCapacity >= codebookRows;
  auto lists = int32_t{1};
  if (approximate) {
    lists = static_cast<int32_t>(std::min<int64_t>(
        requestedLists,
        std::min(sampleCapacity, std::max<int64_t>(1, validRows / 8))));
  }
  impl->nLists = lists;
  impl->approximate = approximate && lists >= 1 && lists <= sampleCapacity;
  if (!impl->approximate) {
    impl->bruteIds = rmm::device_uvector<int64_t>(validRows, stream, mr);
    int64_t offset = 0;
    for (size_t i = 0; i < batches.size(); ++i) {
      auto packed = packBatch(
          cudf::lists_column_view(batches[i].vectors),
          classified[i],
          batches[i].rowStart,
          impl->dimension,
          false,
          stream,
          mr);
      if (packed.rows == 0) {
        continue;
      }
      checkCuda(cudaMemcpyAsync(
          impl->bruteIds.data() + offset,
          packed.rowIds.data(),
          sizeof(int64_t) * static_cast<size_t>(packed.rows),
          cudaMemcpyDeviceToDevice,
          stream.get()));
      offset += packed.rows;
    }
    return std::unique_ptr<RuntimeVectorIndex>(
        new RuntimeVectorIndex(std::move(impl)));
  }

  const auto pqDim = choosePqDim(impl->dimension, pqBits, options.pqDim);
  const auto trainRows = std::max<int32_t>(options.trainRows, lists);
  rmm::device_uvector<float> sample(
      static_cast<size_t>(std::min<int64_t>(trainRows, validRows)) *
          impl->dimension,
      stream,
      mr);
  int64_t sampleRows = 0;
  bool trained = false;
  std::vector<DenseBatch> pending;
  auto trainFromSample = [&](int64_t rows) {
    cuvs::neighbors::ivf_pq::index_params params;
    params.metric = options.cosine || options.largerIsBetter
        ? cuvs::distance::DistanceType::InnerProduct
        : cuvs::distance::DistanceType::L2Expanded;
    params.n_lists = static_cast<uint32_t>(lists);
    params.pq_bits = static_cast<uint32_t>(pqBits);
    params.pq_dim = static_cast<uint32_t>(pqDim);
    params.add_data_on_build = false;
    params.kmeans_trainset_fraction = 1;
    params.kmeans_n_iters = 10;
    params.conservative_memory_allocation = true;
    auto dataset =
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
            sample.data(), rows, impl->dimension);
    try {
      impl->ivf =
          cuvs::neighbors::ivf_pq::build(impl->resources, params, dataset);
    } catch (const std::exception& error) {
      failRuntime(std::string("cuVS IVF-PQ build failed: ") + error.what());
    }
    trained = true;
  };
  auto extendPacked = [&](const DenseBatch& packed) {
    if (packed.rows == 0) {
      return;
    }
    auto vectors =
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
            packed.values.data(), packed.rows, impl->dimension);
    auto ids = raft::make_device_vector_view<const int64_t, int64_t>(
        packed.rowIds.data(), packed.rows);
    try {
      cuvs::neighbors::ivf_pq::extend(
          impl->resources, vectors, std::optional{ids}, &*impl->ivf);
    } catch (const std::exception& error) {
      failRuntime(std::string("cuVS IVF-PQ extend failed: ") + error.what());
    }
  };

  for (size_t i = 0; i < batches.size(); ++i) {
    auto packed = packBatch(
        cudf::lists_column_view(batches[i].vectors),
        classified[i],
        batches[i].rowStart,
        impl->dimension,
        options.cosine,
        stream,
        mr);
    if (!trained && packed.rows > 0 &&
        sampleRows < static_cast<int64_t>(sample.size() / impl->dimension)) {
      const auto room =
          static_cast<int64_t>(sample.size() / impl->dimension) - sampleRows;
      const auto take = std::min(room, packed.rows);
      checkCuda(cudaMemcpyAsync(
          sample.data() + sampleRows * impl->dimension,
          packed.values.data(),
          sizeof(float) * static_cast<size_t>(take) * impl->dimension,
          cudaMemcpyDeviceToDevice,
          stream.get()));
      sampleRows += take;
    }
    const bool last = i + 1 == batches.size();
    if (!trained) {
      pending.push_back(std::move(packed));
      if ((sampleRows >= trainRows || last) && sampleRows > 0) {
        trainFromSample(sampleRows);
        for (const auto& held : pending) {
          extendPacked(held);
        }
        pending.clear();
      }
    } else {
      extendPacked(packed);
    }
  }
  if (!trained) {
    failRuntime("Vector join failed to train an IVF-PQ index");
  }
  return std::unique_ptr<RuntimeVectorIndex>(
      new RuntimeVectorIndex(std::move(impl)));
}

} // namespace facebook::velox::cudf_velox
