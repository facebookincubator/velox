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
#include <mutex>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissException.h>
#include <folly/ScopeGuard.h>
#include <glog/logging.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::vectorindex {
namespace {

struct Device {
  explicit Device(int32_t id) : id{id} {}

  const int32_t id;

  faiss::gpu::StandardGpuResources resources;

  // Serializes use of 'resources', which FAISS does not make thread-safe.
  std::mutex mutex;
};

// Owns one device allocation. Freed if a later copy in the constructor fails,
// which would otherwise skip the index destructor.
struct DeviceIds {
  DeviceIds() = default;
  DeviceIds(const DeviceIds&) = delete;
  DeviceIds& operator=(const DeviceIds&) = delete;

  ~DeviceIds() {
    reset();
  }

  void reset() {
    cudaFree(pointer);
    pointer = nullptr;
  }

  void copyFrom(const void* host, size_t bytes) {
    if (bytes == 0) {
      return;
    }
    auto error = cudaMalloc(reinterpret_cast<void**>(&pointer), bytes);
    VELOX_CHECK(
        error == cudaSuccess,
        "Cannot allocate the device id map of a local vector index: {}",
        cudaGetErrorString(error));
    error = cudaMemcpy(pointer, host, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      cudaFree(pointer);
      pointer = nullptr;
      VELOX_FAIL(
          "Cannot copy the device id map of a local vector index: {}",
          cudaGetErrorString(error));
    }
  }

  int64_t* pointer{nullptr};
};

// Runs 'search' with 'device''s resources ordered on 'stream'.
template <typename Search>
void searchOnStream(
    Device& device,
    void* stream,
    std::string_view what,
    Search&& search) {
  const std::lock_guard<std::mutex> lock(device.mutex);
  faiss::gpu::DeviceScope scope(device.id);
  device.resources.setDefaultStream(
      device.id, static_cast<cudaStream_t>(stream));
  SCOPE_EXIT {
    device.resources.revertDefaultStream(device.id);
  };
  try {
    search();
  } catch (const faiss::FaissException& e) {
    VELOX_FAIL("{} failed: {}", what, e.what());
  }
}

// A local index copied to the device by the FAISS GPU cloner.
class ClonedIndex : public LocalVectorIndexAccelerator::Index {
 public:
  ClonedIndex(
      std::shared_ptr<Device> device,
      std::unique_ptr<faiss::Index> index,
      int64_t maxK)
      : device_{std::move(device)}, index_{std::move(index)}, maxK_{maxK} {
    // The FAISS GPU cloner keeps an id map around the device index and
    // translates its ids on the CPU. A device search uses the device index
    // directly, so the translation moves to the device.
    const auto* idMap = dynamic_cast<const faiss::IndexIDMap*>(index_.get());
    if (idMap == nullptr ||
        dynamic_cast<faiss::gpu::GpuIndex*>(idMap->index) == nullptr) {
      return;
    }
    static_assert(sizeof(faiss::idx_t) == sizeof(int64_t));
    deviceIds_.copyFrom(
        idMap->id_map.data(), idMap->id_map.size() * sizeof(faiss::idx_t));
  }

  ClonedIndex(const ClonedIndex&) = delete;
  ClonedIndex& operator=(const ClonedIndex&) = delete;
  ClonedIndex(ClonedIndex&&) = delete;
  ClonedIndex& operator=(ClonedIndex&&) = delete;

  ~ClonedIndex() override {
    const std::lock_guard<std::mutex> lock(device_->mutex);
    faiss::gpu::DeviceScope scope(device_->id);
    index_.reset();
    deviceIds_.reset();
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
            queries.size(), queries.data(), gpuK, searchScores, searchIds);
      } catch (const faiss::FaissException& e) {
        VELOX_FAIL("FAISS GPU local vector index search failed: {}", e.what());
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

  bool searchDevice(
      const float* queries,
      int64_t numQueries,
      int64_t k,
      int64_t* ids,
      float* scores,
      void* stream) const override {
    if (k < 1 || k > maxK_ || k > index_->ntotal) {
      return false;
    }
    auto* gpu = dynamic_cast<faiss::gpu::GpuIndex*>(index_.get());
    if (gpu == nullptr) {
      if (const auto* idMap =
              dynamic_cast<const faiss::IndexIDMap*>(index_.get())) {
        gpu = dynamic_cast<faiss::gpu::GpuIndex*>(idMap->index);
      }
    }
    if (gpu == nullptr) {
      return false;
    }
    searchOnStream(
        *device_, stream, "FAISS GPU local vector index search", [&] {
          gpu->search(numQueries, queries, k, scores, ids);
        });
    return true;
  }

  const int64_t* deviceIdMap() const override {
    return deviceIds_.pointer;
  }

 private:
  const std::shared_ptr<Device> device_;
  std::unique_ptr<faiss::Index> index_;
  const int64_t maxK_;
  DeviceIds deviceIds_;
};

// An exact flat copy of a kHnswSq index too small for an HNSW graph. The
// device index numbers embeddings in the order they were added; 'ids' maps
// them back to the local index ids.
class MappedIndex : public LocalVectorIndexAccelerator::Index {
 public:
  // Called with 'device''s mutex held and its device current.
  MappedIndex(
      std::shared_ptr<Device> device,
      std::unique_ptr<faiss::Index> index,
      const std::vector<faiss::idx_t>& ids,
      int64_t maxK)
      : device_{std::move(device)},
        index_{std::move(index)},
        ids_{ids},
        maxK_{maxK} {
    deviceIds_.copyFrom(ids_.data(), ids_.size() * sizeof(int64_t));
  }

  MappedIndex(const MappedIndex&) = delete;
  MappedIndex& operator=(const MappedIndex&) = delete;
  MappedIndex(MappedIndex&&) = delete;
  MappedIndex& operator=(MappedIndex&&) = delete;

  ~MappedIndex() override {
    const std::lock_guard<std::mutex> lock(device_->mutex);
    faiss::gpu::DeviceScope scope(device_->id);
    index_.reset();
    deviceIds_.reset();
  }

  bool search(
      const Embeddings& queries,
      int64_t k,
      std::span<int64_t> ids,
      std::span<float> scores) const override {
    if (k > maxK_) {
      return false;
    }
    const auto gpuK = std::min(k, static_cast<int64_t>(ids_.size()));
    if (gpuK < 1) {
      return false;
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
            queries.size(), queries.data(), gpuK, searchScores, searchIds);
      } catch (const faiss::FaissException& e) {
        VELOX_FAIL("FAISS GPU local vector index search failed: {}", e.what());
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

  bool searchDevice(
      const float* queries,
      int64_t numQueries,
      int64_t k,
      int64_t* ids,
      float* scores,
      void* stream) const override {
    if (k < 1 || k > maxK_ || k > static_cast<int64_t>(ids_.size())) {
      return false;
    }
    searchOnStream(*device_, stream, "Local vector index device search", [&] {
      index_->search(numQueries, queries, k, scores, ids);
    });
    return true;
  }

  const int64_t* deviceIdMap() const override {
    return deviceIds_.pointer;
  }

 private:
  const std::shared_ptr<Device> device_;
  std::unique_ptr<faiss::Index> index_;
  // Aliases the CPU index id map. LocalVectorIndex destroys this accelerator
  // before that index.
  const std::vector<faiss::idx_t>& ids_;
  const int64_t maxK_;
  DeviceIds deviceIds_;
};

// FAISS has no GPU HNSW index. A graph too small to search as HNSW, when the
// caller requires a device index, is searched exactly with GpuIndexFlat.
// Returns null for a real HNSW graph and when a device index is not required.
std::unique_ptr<LocalVectorIndexAccelerator::Index> loadSmallHnsw(
    const std::shared_ptr<Device>& device,
    const faiss::IndexIDMap& idMap,
    bool requireDevice) {
  if (!requireDevice || idMap.ntotal < 1) {
    return nullptr;
  }
  const auto& graph = dynamic_cast<const faiss::IndexHNSW&>(*idMap.index);
  auto degree = std::min<size_t>(graph.hnsw.nb_neighbors(0), graph.ntotal - 1);
  if (degree % 2 != 0) {
    --degree;
  }
  if (degree >= 4) {
    return nullptr;
  }
  std::vector<float> embeddings(graph.ntotal * graph.d);
  graph.reconstruct_n(0, graph.ntotal, embeddings.data());
  faiss::gpu::DeviceScope scope(device->id);
  faiss::gpu::GpuIndexFlatConfig flatConfig;
  flatConfig.device = device->id;
  auto flat = std::make_unique<faiss::gpu::GpuIndexFlat>(
      &device->resources, graph.d, graph.metric_type, flatConfig);
  flat->add(graph.ntotal, embeddings.data());
  return std::make_unique<MappedIndex>(
      device,
      std::move(flat),
      idMap.id_map,
      faiss::gpu::getMaxKSelection(/*use_cuvs=*/false));
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

class GpuAccelerator : public LocalVectorIndexAccelerator {
 public:
  explicit GpuAccelerator(const GpuLocalVectorIndexOptions& options)
      : options_{options}, device_{std::make_shared<Device>(options.device)} {}

  std::unique_ptr<Index> load(VectorIndexKind type, const faiss::Index& index)
      const override {
    if (!options_.requireDevice && index.ntotal < options_.minSearchSize) {
      return nullptr;
    }
    const std::lock_guard<std::mutex> lock(device_->mutex);
    try {
      faiss::gpu::DeviceScope scope(device_->id);
      if (type == VectorIndexKind::kHnswSq) {
        auto flat = loadSmallHnsw(
            device_,
            dynamic_cast<const faiss::IndexIDMap&>(index),
            options_.requireDevice);
        VELOX_CHECK(
            flat != nullptr || !options_.requireDevice,
            "HNSW SQ has no FAISS GPU index");
        return flat;
      }
      faiss::gpu::GpuClonerOptions options;
      options.use_cuvs = false;
      std::unique_ptr<faiss::Index> copy(
          faiss::gpu::index_cpu_to_gpu(
              &device_->resources, device_->id, &index, &options));
      return std::make_unique<ClonedIndex>(
          device_,
          std::move(copy),
          faiss::gpu::getMaxKSelection(/*use_cuvs=*/false));
    } catch (const std::exception& e) {
      VELOX_CHECK(
          !options_.requireDevice,
          "Cannot load a local vector index onto the device: {}",
          e.what());
      LOG(WARNING) << "Local vector index stays on the CPU: " << e.what();
      return nullptr;
    }
  }

  bool build(
      VectorIndexKind type,
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
        case VectorIndexKind::kIvfFlat:
          buildIvf<faiss::gpu::GpuIndexIVFFlat>(
              *device_,
              dynamic_cast<faiss::IndexIVFFlat&>(index),
              embeddings,
              ids);
          return true;
        case VectorIndexKind::kIvfSq:
          buildIvf<faiss::gpu::GpuIndexIVFScalarQuantizer>(
              *device_,
              dynamic_cast<faiss::IndexIVFScalarQuantizer&>(index),
              embeddings,
              ids);
          return true;
        case VectorIndexKind::kFlat:
        case VectorIndexKind::kHnswSq:
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
