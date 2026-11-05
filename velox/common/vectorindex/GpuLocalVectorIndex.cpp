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
#include <vector>

#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
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
      int64_t maxK)
      : device_{std::move(device)}, index_{std::move(index)}, maxK_{maxK} {}

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

 private:
  const std::shared_ptr<Device> device_;
  std::unique_ptr<faiss::Index> index_;
  const int64_t maxK_;
};

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
        // FAISS has no GPU HNSW index. Search stays on the CPU.
        return nullptr;
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
