/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/index/VectorIndexGpu.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissException.h>
#include <glog/logging.h>

#include <cuda_runtime.h>

#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble::index {
namespace {

constexpr size_t kCagraBuildIterations{12};

void checkCuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

// Caps a CAGRA degree so a small dataset can still build a graph.
size_t cappedGraphDegree(size_t requested, faiss::idx_t numVectors) {
  if (numVectors < 2 || requested == 0) {
    return 0;
  }
  const auto limit = static_cast<size_t>(numVectors - 1);
  auto degree = std::min(requested, limit);
  if (degree > 2 && degree % 2 != 0) {
    --degree;
  }
  return degree;
}

std::unique_ptr<faiss::Index> trainCagra(
    faiss::gpu::StandardGpuResources& resources,
    int deviceId,
    int dimensions,
    faiss::MetricType metric,
    size_t graphDegree,
    faiss::idx_t numVectors,
    const float* vectors) {
  const auto intermediate =
      std::min(graphDegree * 2, static_cast<size_t>(numVectors - 1));
  faiss::gpu::GpuIndexCagraConfig config;
  config.device = deviceId;
  config.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
  config.graph_degree = graphDegree;
  config.intermediate_graph_degree = std::max(graphDegree, intermediate);
  config.nn_descent_niter = kCagraBuildIterations;

  faiss::gpu::DeviceScope scope(deviceId);
  auto gpuIndex = std::make_unique<faiss::gpu::GpuIndexCagra>(
      &resources, dimensions, metric, config);
  gpuIndex->add(numVectors, vectors);

  auto cpuIndex = std::make_unique<faiss::IndexHNSWCagra>(
      dimensions, static_cast<int>(graphDegree / 2), metric);
  // Level 0 is the CAGRA graph. CPU search walks that level directly.
  cpuIndex->base_level_only = true;
  gpuIndex->copyTo(cpuIndex.get());
  return cpuIndex;
}

std::unique_ptr<faiss::gpu::GpuIndexCagra> cagraFromReconstructed(
    faiss::gpu::StandardGpuResources& resources,
    int deviceId,
    const faiss::Index& cpuIndex) {
  const auto numVectors = cpuIndex.ntotal;
  const auto dimensions = cpuIndex.d;
  if (numVectors < 2 || dimensions < 1) {
    return nullptr;
  }
  if (cpuIndex.metric_type != faiss::METRIC_L2 &&
      cpuIndex.metric_type != faiss::METRIC_INNER_PRODUCT) {
    return nullptr;
  }
  size_t graphDegree = 64;
  if (const auto* hnsw = dynamic_cast<const faiss::IndexHNSW*>(&cpuIndex)) {
    graphDegree = static_cast<size_t>(hnsw->hnsw.nb_neighbors(0));
  }
  graphDegree = cappedGraphDegree(graphDegree, numVectors);
  // copyTo derives HNSW M as degree/2 and divides by log(M).
  if (graphDegree < 4) {
    return nullptr;
  }
  if (static_cast<size_t>(numVectors) >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(dimensions)) {
    return nullptr;
  }
  std::vector<float> vectors(
      static_cast<size_t>(numVectors) * static_cast<size_t>(dimensions));
  cpuIndex.reconstruct_n(0, numVectors, vectors.data());

  const auto intermediate =
      std::min(graphDegree * 2, static_cast<size_t>(numVectors - 1));
  faiss::gpu::GpuIndexCagraConfig config;
  config.device = deviceId;
  config.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
  config.graph_degree = graphDegree;
  config.intermediate_graph_degree = std::max(graphDegree, intermediate);
  config.nn_descent_niter = kCagraBuildIterations;
  faiss::gpu::DeviceScope scope(deviceId);
  auto gpuIndex = std::make_unique<faiss::gpu::GpuIndexCagra>(
      &resources, dimensions, cpuIndex.metric_type, config);
  gpuIndex->add(numVectors, vectors.data());
  return gpuIndex;
}

} // namespace

struct GpuVectorIndex::IvfRaBitQIndex {
  IvfRaBitQIndex(
      cuvs::neighbors::ivf_rabitq::index<int64_t> built,
      uint32_t lists,
      uint32_t bits)
      : index{std::move(built)}, nLists{lists}, bitsPerDim{bits} {}

  cuvs::neighbors::ivf_rabitq::index<int64_t> index;
  uint32_t nLists{0};
  uint32_t bitsPerDim{1};
};

struct GpuVectorIndex::Device {
  explicit Device(int _id) : id{_id} {}

  const int id;

  faiss::gpu::StandardGpuResources resources;

  // Serializes use of 'resources', which FAISS does not make thread-safe.
  std::mutex mutex;
};

GpuVectorIndex::Device& GpuVectorIndex::device(int deviceId) {
  // Leaked so the state outlives static destructors that still hold indexes.
  static auto* devices = [] {
    auto* states = new std::vector<std::unique_ptr<Device>>();
    const auto count = numDevices();
    for (int id = 0; id < count; ++id) {
      states->push_back(std::make_unique<Device>(id));
    }
    return states;
  }();
  NIMBLE_CHECK_GE(deviceId, 0);
  NIMBLE_CHECK_LT(static_cast<size_t>(deviceId), devices->size());
  return *(*devices)[deviceId];
}

int GpuVectorIndex::numDevices() {
  try {
    return faiss::gpu::getNumDevices();
  } catch (const std::exception&) {
    return 0;
  }
}

bool GpuVectorIndex::supports(VectorIndexType indexType) {
  switch (indexType) {
    case VectorIndexType::kIvfFlat:
    case VectorIndexType::kIvfSq8:
    case VectorIndexType::kIvfPq:
    case VectorIndexType::kIvfRaBitQ:
    case VectorIndexType::kHnswSq8:
    case VectorIndexType::kCagra:
      return true;
    case VectorIndexType::kIvfRaBitQFastScan:
      return false;
  }
  NIMBLE_UNREACHABLE(
      "Unsupported vector index type: {}", static_cast<int>(indexType));
}

std::shared_ptr<const GpuVectorIndex> GpuVectorIndex::create(
    const faiss::Index& cpuIndex,
    VectorIndexType indexType) {
  NIMBLE_CHECK(supports(indexType), "Index type has no GPU implementation");
  auto& device = GpuVectorIndex::device(faiss::gpu::getCurrentDevice());
  try {
    const std::lock_guard<std::mutex> lock{device.mutex};
    faiss::gpu::DeviceScope scope(device.id);
    if (indexType == VectorIndexType::kIvfRaBitQ) {
      if (cpuIndex.metric_type != faiss::METRIC_L2) {
        LOG(WARNING) << "RaBitQ GPU search supports L2 only; staying on CPU";
        return nullptr;
      }
      const auto* ivf = dynamic_cast<const faiss::IndexIVFRaBitQ*>(&cpuIndex);
      if (ivf == nullptr || ivf->ntotal < 1 || ivf->nlist < 1) {
        return nullptr;
      }
      const auto numVectors = ivf->ntotal;
      const auto dimensions = ivf->d;
      std::vector<float> vectors(
          static_cast<size_t>(numVectors) * static_cast<size_t>(dimensions));
      ivf->reconstruct_n(0, numVectors, vectors.data());
      auto nLists = static_cast<uint32_t>(
          std::min<int64_t>(ivf->nlist, numVectors));
      auto bits = static_cast<uint32_t>(ivf->rabitq.nb_bits);
      if (bits < 1) {
        bits = 1;
      }
      if (bits > 9) {
        bits = 9;
      }
      auto& handle = device.resources.getRaftHandle(device.id);
      cuvs::neighbors::ivf_rabitq::index_params params;
      params.n_lists = nLists;
      params.bits_per_dim = bits;
      auto dataset = raft::make_host_matrix_view<const float, int64_t>(
          vectors.data(), numVectors, static_cast<int64_t>(dimensions));
      auto built =
          cuvs::neighbors::ivf_rabitq::build(handle, params, dataset);
      auto raBitQ =
          std::make_unique<IvfRaBitQIndex>(std::move(built), nLists, bits);
      return std::shared_ptr<const GpuVectorIndex>(
          new GpuVectorIndex(device, nullptr, std::move(raBitQ)));
    }
    if (indexType == VectorIndexType::kHnswSq8) {
      auto gpuIndex =
          cagraFromReconstructed(device.resources, device.id, cpuIndex);
      if (gpuIndex == nullptr) {
        return nullptr;
      }
      return std::shared_ptr<const GpuVectorIndex>(new GpuVectorIndex(
          device, std::move(gpuIndex), nullptr));
    }

    faiss::gpu::GpuClonerOptions options;
    options.use_cuvs = true;
    std::unique_ptr<faiss::Index> gpuIndex{faiss::gpu::index_cpu_to_gpu(
        &device.resources, device.id, &cpuIndex, &options)};
    return std::shared_ptr<const GpuVectorIndex>(
        new GpuVectorIndex(device, std::move(gpuIndex), nullptr));
  } catch (const std::exception& error) {
    LOG(WARNING) << "Vector index stays on CPU; GPU setup failed: "
                 << error.what();
    return nullptr;
  }
}

GpuVectorIndex::GpuVectorIndex(
    Device& device,
    std::unique_ptr<faiss::Index> index,
    std::unique_ptr<IvfRaBitQIndex> raBitQ)
    : device_{device},
      index_{std::move(index)},
      raBitQ_{std::move(raBitQ)} {
  NIMBLE_CHECK(
      index_ != nullptr || raBitQ_ != nullptr,
      "GPU vector index is empty");
}

GpuVectorIndex::~GpuVectorIndex() {
  const std::lock_guard<std::mutex> lock{device_.mutex};
  index_.reset();
  raBitQ_.reset();
}

int64_t GpuVectorIndex::maxSelection() {
  return faiss::gpu::getMaxKSelection(/*use_cuvs=*/true);
}

bool GpuVectorIndex::search(
    faiss::idx_t numQueries,
    const float* queryVectors,
    faiss::idx_t maxNumNeighbors,
    float* scores,
    faiss::idx_t* labels,
    const faiss::SearchParametersIVF& params) const {
  if (raBitQ_ != nullptr) {
    // cuVS IVF-RaBitQ search has no IDSelector. A filter stays on the CPU.
    if (params.sel != nullptr) {
      return false;
    }
    static_assert(sizeof(faiss::idx_t) == sizeof(int64_t));
    const std::lock_guard<std::mutex> lock{device_.mutex};
    faiss::gpu::DeviceScope scope(device_.id);
    try {
      auto& handle = device_.resources.getRaftHandle(device_.id);
      const auto stream = raft::resource::get_cuda_stream(handle).get();
      const auto dimensions = static_cast<int64_t>(raBitQ_->index.dim());
      auto queries = raft::make_device_matrix<float, int64_t>(
          handle, numQueries, dimensions);
      auto neighbors = raft::make_device_matrix<int64_t, int64_t>(
          handle, numQueries, maxNumNeighbors);
      auto distances = raft::make_device_matrix<float, int64_t>(
          handle, numQueries, maxNumNeighbors);
      checkCuda(cudaMemcpyAsync(
          queries.data_handle(),
          queryVectors,
          sizeof(float) * static_cast<size_t>(numQueries) *
              static_cast<size_t>(dimensions),
          cudaMemcpyHostToDevice,
          stream));
      cuvs::neighbors::ivf_rabitq::search_params searchParams;
      searchParams.n_probes = static_cast<uint32_t>(std::min<size_t>(
          params.nprobe, raBitQ_->nLists));
      searchParams.mode = raBitQ_->bitsPerDim <= 1
          ? cuvs::neighbors::ivf_rabitq::search_mode::LUT16
          : cuvs::neighbors::ivf_rabitq::search_mode::QUANT4;
      cuvs::neighbors::ivf_rabitq::search(
          handle,
          searchParams,
          raBitQ_->index,
          queries.view(),
          neighbors.view(),
          distances.view());
      checkCuda(cudaMemcpyAsync(
          labels,
          neighbors.data_handle(),
          sizeof(int64_t) * static_cast<size_t>(numQueries) *
              static_cast<size_t>(maxNumNeighbors),
          cudaMemcpyDeviceToHost,
          stream));
      checkCuda(cudaMemcpyAsync(
          scores,
          distances.data_handle(),
          sizeof(float) * static_cast<size_t>(numQueries) *
              static_cast<size_t>(maxNumNeighbors),
          cudaMemcpyDeviceToHost,
          stream));
      checkCuda(cudaStreamSynchronize(stream));
    } catch (const std::exception& error) {
      NIMBLE_FAIL("cuVS RaBitQ search failed: {}", error.what());
    }
    return true;
  }

  // Pass a fresh object so CPU-only fields such as quantizer_params do not
  // reach FAISS GPU, which rejects unknown parameter types.
  faiss::SearchParametersIVF gpuParams;
  gpuParams.nprobe = params.nprobe;
  gpuParams.sel = params.sel;
  const std::lock_guard<std::mutex> lock{device_.mutex};
  try {
    index_->search(
        numQueries, queryVectors, maxNumNeighbors, scores, labels, &gpuParams);
  } catch (const faiss::FaissException& error) {
    NIMBLE_FAIL("FAISS GPU search failed: {}", error.what());
  }
  return true;
}

void GpuVectorIndex::searchCagra(
    faiss::idx_t numQueries,
    const float* queryVectors,
    faiss::idx_t maxNumNeighbors,
    float* scores,
    faiss::idx_t* labels,
    size_t intermediateResults,
    faiss::IDSelector* selector) const {
  faiss::gpu::SearchParametersCagra params;
  params.itopk_size =
      std::max(intermediateResults, static_cast<size_t>(maxNumNeighbors));
  params.sel = selector;
  const std::lock_guard<std::mutex> lock{device_.mutex};
  try {
    index_->search(
        numQueries, queryVectors, maxNumNeighbors, scores, labels, &params);
  } catch (const faiss::FaissException& error) {
    NIMBLE_FAIL("CAGRA GPU search failed: {}", error.what());
  }
}

std::unique_ptr<faiss::Index> buildGpuCagraIndex(
    int dimensions,
    int numConnections,
    faiss::MetricType metric,
    faiss::idx_t numVectors,
    const float* vectors) {
  if (GpuVectorIndex::numDevices() < 1 || vectors == nullptr ||
      numConnections < 1) {
    return nullptr;
  }
  const auto graphDegree = cappedGraphDegree(
      static_cast<size_t>(numConnections) * 2, numVectors);
  if (graphDegree < 4) {
    return nullptr;
  }
  try {
    faiss::gpu::StandardGpuResources resources;
    return trainCagra(
        resources,
        faiss::gpu::getCurrentDevice(),
        dimensions,
        metric,
        graphDegree,
        numVectors,
        vectors);
  } catch (const std::exception& error) {
    LOG(WARNING) << "CAGRA GPU build failed; building the graph on CPU: "
                 << error.what();
    return nullptr;
  }
}

} // namespace facebook::nimble::index
