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

#include "velox/common/vectorindex/GpuLocalVectorIndex.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <mutex>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissException.h>
#include <glog/logging.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::vectorindex {
namespace {

constexpr size_t kCagraBuildIterations{12};
// CAGRA keeps at least 'k' candidates; a larger 'k' searches on the CPU.
constexpr int64_t kMaxCagraK{1024};

struct Device {
  explicit Device(int32_t id) : id{id} {}

  const int32_t id;

  faiss::gpu::StandardGpuResources resources;

  // Serializes use of 'resources', which FAISS does not make thread-safe.
  std::mutex mutex;
};

// A local index copied to the device by the FAISS GPU cloner.
class ClonedIndex : public LocalVectorIndexAccelerator::Index {
 public:
  ClonedIndex(
      std::shared_ptr<Device> device,
      std::unique_ptr<faiss::Index> index,
      int64_t maxK,
      size_t efSearch = 0)
      : device_{std::move(device)},
        index_{std::move(index)},
        maxK_{maxK},
        efSearch_{efSearch} {}

  ~ClonedIndex() override {
    const std::lock_guard<std::mutex> lock(device_->mutex);
    index_.reset();
  }

  bool search(
      const Embeddings& queries,
      int64_t k,
      std::span<int64_t> ids,
      std::span<float> scores) const override {
    if (k > maxK_) {
      return false;
    }
    // The GPU selection writes past a short index instead of the padding id
    // the CPU search writes. Search only as many neighbors as the index holds
    // and fill the rest here.
    const auto gpuK = std::min(k, std::max<int64_t>(index_->ntotal, 0));
    if (gpuK < 1) {
      return false;
    }
    // Set for a stored CAGRA index: hnsw.efSearch is its intermediate result
    // count. Other cloned indexes keep the nprobe copied with them.
    faiss::gpu::SearchParametersCagra cagraParams;
    const faiss::SearchParameters* params = nullptr;
    if (efSearch_ > 0) {
      cagraParams.itopk_size = std::max(efSearch_, static_cast<size_t>(gpuK));
      params = &cagraParams;
    }
    std::vector<int64_t> limitedIds;
    std::vector<float> limitedScores;
    auto* searchIds = ids.data();
    auto* searchScores = scores.data();
    if (gpuK < k) {
      limitedIds.resize(queries.size() * gpuK);
      limitedScores.resize(queries.size() * gpuK);
      searchIds = limitedIds.data();
      searchScores = limitedScores.data();
    }
    {
      const std::lock_guard<std::mutex> lock(device_->mutex);
      try {
        index_->search(
            queries.size(),
            queries.data(),
            gpuK,
            searchScores,
            searchIds,
            params);
      } catch (const faiss::FaissException& e) {
        VELOX_FAIL("FAISS GPU local vector index search failed: {}", e.what());
      }
    }
    // cuVS writes the maximum id where the CPU search writes kPaddingId.
    constexpr int64_t kMissing = std::numeric_limits<int64_t>::max();
    auto* rawIds = gpuK < k ? limitedIds.data() : ids.data();
    const auto idCount =
        queries.size() * static_cast<size_t>(gpuK < k ? gpuK : k);
    for (size_t i = 0; i < idCount; ++i) {
      if (rawIds[i] == kMissing) {
        rawIds[i] = -1;
      }
    }
    if (gpuK < k) {
      const auto queriesSize = static_cast<int64_t>(queries.size());
      for (int64_t query = 0; query < queriesSize; ++query) {
        for (int64_t neighbor = 0; neighbor < gpuK; ++neighbor) {
          ids[query * k + neighbor] = limitedIds[query * gpuK + neighbor];
          scores[query * k + neighbor] = limitedScores[query * gpuK + neighbor];
        }
        for (int64_t neighbor = gpuK; neighbor < k; ++neighbor) {
          ids[query * k + neighbor] = -1;
        }
      }
    }
    return true;
  }

 private:
  const std::shared_ptr<Device> device_;
  std::unique_ptr<faiss::Index> index_;
  const int64_t maxK_;
  const size_t efSearch_;
};

// A kHnswSq local index rebuilt as a CAGRA graph. CAGRA numbers embeddings
// in the order they were added; 'ids' maps them back to the local index ids.
class CagraIndex : public LocalVectorIndexAccelerator::Index {
 public:
  CagraIndex(
      std::shared_ptr<Device> device,
      std::unique_ptr<faiss::gpu::GpuIndexCagra> index,
      const std::vector<faiss::idx_t>& ids,
      size_t efSearch)
      : device_{std::move(device)},
        index_{std::move(index)},
        ids_{ids},
        efSearch_{efSearch} {}

  ~CagraIndex() override {
    const std::lock_guard<std::mutex> lock(device_->mutex);
    index_.reset();
  }

  bool search(
      const Embeddings& queries,
      int64_t k,
      std::span<int64_t> ids,
      std::span<float> scores) const override {
    if (k > kMaxCagraK) {
      return false;
    }
    const auto gpuK = std::min(k, static_cast<int64_t>(ids_.size()));
    if (gpuK < 1) {
      return false;
    }
    faiss::gpu::SearchParametersCagra params;
    params.itopk_size = std::max(efSearch_, static_cast<size_t>(gpuK));
    std::vector<int64_t> limitedIds;
    std::vector<float> limitedScores;
    auto* searchIds = ids.data();
    auto* searchScores = scores.data();
    if (gpuK < k) {
      limitedIds.resize(queries.size() * gpuK);
      limitedScores.resize(queries.size() * gpuK);
      searchIds = limitedIds.data();
      searchScores = limitedScores.data();
    }
    {
      const std::lock_guard<std::mutex> lock(device_->mutex);
      try {
        index_->search(
            queries.size(),
            queries.data(),
            gpuK,
            searchScores,
            searchIds,
            &params);
      } catch (const faiss::FaissException& e) {
        VELOX_FAIL("CAGRA local vector index search failed: {}", e.what());
      }
    }
    const auto queriesSize = static_cast<int64_t>(queries.size());
    for (int64_t query = 0; query < queriesSize; ++query) {
      for (int64_t neighbor = 0; neighbor < k; ++neighbor) {
        const auto slot = query * k + neighbor;
        if (neighbor >= gpuK) {
          ids[slot] = -1;
          continue;
        }
        const auto id = searchIds[query * gpuK + neighbor];
        if (id >= 0 && static_cast<size_t>(id) < ids_.size()) {
          ids[slot] = ids_[id];
          if (gpuK < k) {
            scores[slot] = searchScores[query * gpuK + neighbor];
          }
        } else {
          ids[slot] = -1;
        }
      }
    }
    return true;
  }

 private:
  const std::shared_ptr<Device> device_;
  std::unique_ptr<faiss::gpu::GpuIndexCagra> index_;
  const std::vector<faiss::idx_t>& ids_;
  const size_t efSearch_;
};

// Returns the even graph degree nearest 'requested' that 'numEmbeddings' can
// fill, or 0 if they are too few for a CAGRA graph.
size_t graphDegree(size_t requested, int64_t numEmbeddings) {
  auto degree = std::min<size_t>(requested, numEmbeddings - 1);
  if (degree % 2 != 0) {
    --degree;
  }
  return degree < 4 ? 0 : degree;
}

std::unique_ptr<LocalVectorIndexAccelerator::Index> loadCagra(
    const std::shared_ptr<Device>& device,
    const faiss::IndexIDMap& idMap) {
  const auto& graph = dynamic_cast<const faiss::IndexHNSW&>(*idMap.index);
  const auto degree = graphDegree(graph.hnsw.nb_neighbors(0), graph.ntotal);
  if (degree == 0) {
    return nullptr;
  }
  std::vector<float> embeddings(graph.ntotal * graph.d);
  graph.reconstruct_n(0, graph.ntotal, embeddings.data());

  faiss::gpu::GpuIndexCagraConfig config;
  config.device = device->id;
  config.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
  config.graph_degree = degree;
  config.intermediate_graph_degree =
      std::max<size_t>(degree, std::min<size_t>(2 * degree, graph.ntotal - 1));
  config.nn_descent_niter = kCagraBuildIterations;
  faiss::gpu::DeviceScope scope(device->id);
  auto cagra = std::make_unique<faiss::gpu::GpuIndexCagra>(
      &device->resources, graph.d, graph.metric_type, config);
  cagra->add(graph.ntotal, embeddings.data());
  return std::make_unique<CagraIndex>(
      device, std::move(cagra), idMap.id_map, graph.hnsw.efSearch);
}

// Trains 'cpu' on the device, adds the embeddings, and copies the trained
// quantizer and filled lists back into 'cpu'. Uses the FAISS GPU IVF index
// rather than cuVS's, so training runs the same k-means as the CPU build.
template <typename GpuIvf, typename CpuIvf>
void buildIvf(
    Device& device,
    CpuIvf& cpu,
    const Embeddings& embeddings,
    std::span<const int64_t> ids) {
  faiss::gpu::GpuClonerOptions options;
  options.use_cuvs = false;
  std::unique_ptr<faiss::Index> index(
      faiss::gpu::index_cpu_to_gpu(
          &device.resources, device.id, &cpu, &options));
  auto& gpu = dynamic_cast<GpuIvf&>(*index);
  gpu.cp = cpu.cp;
  gpu.train(embeddings.size(), embeddings.data());
  gpu.add_with_ids(embeddings.size(), embeddings.data(), ids.data());
  gpu.copyTo(&cpu);
}

// Builds a CAGRA graph on the device and stores it in 'index', an IndexIDMap
// over an IndexHNSWCagra, so the written bytes search on CPU and on GPU.
bool buildCagra(
    Device& device,
    faiss::Index& index,
    const Embeddings& embeddings,
    std::span<const int64_t> ids) {
  auto& idMap = dynamic_cast<faiss::IndexIDMap&>(index);
  auto& cpu = dynamic_cast<faiss::IndexHNSWCagra&>(*idMap.index);
  const auto degree = graphDegree(cpu.hnsw.nb_neighbors(0), embeddings.size());
  if (degree == 0) {
    return false;
  }
  faiss::gpu::GpuIndexCagraConfig config;
  config.device = device.id;
  config.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
  config.graph_degree = degree;
  config.intermediate_graph_degree = std::max<size_t>(
      degree, std::min<size_t>(2 * degree, embeddings.size() - 1));
  config.nn_descent_niter = kCagraBuildIterations;
  faiss::gpu::DeviceScope scope(device.id);
  faiss::gpu::GpuIndexCagra gpu(
      &device.resources, cpu.d, cpu.metric_type, config);
  gpu.add(embeddings.size(), embeddings.data());
  const auto efSearch = cpu.hnsw.efSearch;
  cpu.base_level_only = true;
  gpu.copyTo(&cpu);
  cpu.base_level_only = true;
  cpu.hnsw.efSearch = efSearch;
  idMap.id_map.assign(ids.begin(), ids.end());
  idMap.ntotal = cpu.ntotal;
  return idMap.ntotal == static_cast<faiss::idx_t>(ids.size());
}

// hnsw.efSearch of a CAGRA local index, which may be the index itself or the
// index under its id map.
size_t cagraEfSearch(const faiss::Index& index) {
  const auto* graph = dynamic_cast<const faiss::IndexHNSWCagra*>(&index);
  if (graph == nullptr) {
    graph = dynamic_cast<const faiss::IndexHNSWCagra*>(
        dynamic_cast<const faiss::IndexIDMap&>(index).index);
  }
  return graph->hnsw.efSearch;
}

class GpuAccelerator : public LocalVectorIndexAccelerator {
 public:
  explicit GpuAccelerator(const GpuLocalVectorIndexOptions& options)
      : options_{options}, device_{std::make_shared<Device>(options.device)} {}

  std::unique_ptr<Index> load(VectorIndexType type, const faiss::Index& index)
      const override {
    if (index.ntotal < options_.minSearchSize) {
      return nullptr;
    }
    const std::lock_guard<std::mutex> lock(device_->mutex);
    try {
      faiss::gpu::DeviceScope scope(device_->id);
      if (type == VectorIndexType::kHnswSq) {
        return loadCagra(
            device_, dynamic_cast<const faiss::IndexIDMap&>(index));
      }
      faiss::gpu::GpuClonerOptions options;
      options.use_cuvs = true;
      std::unique_ptr<faiss::Index> copy(
          faiss::gpu::index_cpu_to_gpu(
              &device_->resources, device_->id, &index, &options));
      const bool cagra = type == VectorIndexType::kCagra;
      return std::make_unique<ClonedIndex>(
          device_,
          std::move(copy),
          cagra ? kMaxCagraK : faiss::gpu::getMaxKSelection(/*use_cuvs=*/true),
          cagra ? cagraEfSearch(index) : 0);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Local vector index stays on the CPU: " << e.what();
      return nullptr;
    }
  }

  bool build(
      VectorIndexType type,
      faiss::Index& index,
      const Embeddings& embeddings,
      std::span<const int64_t> ids) const override {
    if (!options_.build ||
        static_cast<int64_t>(embeddings.size()) < options_.minBuildSize) {
      return false;
    }
    const std::lock_guard<std::mutex> lock(device_->mutex);
    try {
      faiss::gpu::DeviceScope scope(device_->id);
      switch (type) {
        case VectorIndexType::kIvfFlat:
          buildIvf<faiss::gpu::GpuIndexIVFFlat>(
              *device_,
              dynamic_cast<faiss::IndexIVFFlat&>(index),
              embeddings,
              ids);
          return true;
        case VectorIndexType::kIvfSq:
          buildIvf<faiss::gpu::GpuIndexIVFScalarQuantizer>(
              *device_,
              dynamic_cast<faiss::IndexIVFScalarQuantizer&>(index),
              embeddings,
              ids);
          return true;
        case VectorIndexType::kCagra:
          return buildCagra(*device_, index, embeddings, ids);
        case VectorIndexType::kFlat:
        case VectorIndexType::kHnswSq:
          return false;
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Local vector index builds on the CPU: " << e.what();
      VELOX_CHECK_EQ(index.ntotal, 0);
    }
    return false;
  }

 private:
  const GpuLocalVectorIndexOptions options_;
  const std::shared_ptr<Device> device_;
};

} // namespace

std::shared_ptr<const LocalVectorIndexAccelerator>
makeGpuLocalVectorIndexAccelerator(const GpuLocalVectorIndexOptions& options) {
  VELOX_USER_CHECK(
      options.device >= 0 && options.device < gpuLocalVectorIndexDevices(),
      "No CUDA device {} for local vector indexes",
      options.device);
  return std::make_shared<GpuAccelerator>(options);
}

int32_t gpuLocalVectorIndexDevices() {
  try {
    return faiss::gpu::getNumDevices();
  } catch (const std::exception&) {
    return 0;
  }
}

} // namespace facebook::velox::common::vectorindex
