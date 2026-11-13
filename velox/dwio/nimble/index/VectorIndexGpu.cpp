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

#include <exception>
#include <mutex>
#include <utility>
#include <vector>

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <glog/logging.h>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble::index {

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
      return true;
    case VectorIndexType::kIvfRaBitQ:
    case VectorIndexType::kIvfRaBitQFastScan:
    case VectorIndexType::kHnswSq8:
      return false;
  }
  NIMBLE_UNREACHABLE(
      "Unsupported vector index type: {}", static_cast<int>(indexType));
}

// Builds an empty GPU index and then copies into it. The unique_ptr owns the
// index before copyFrom runs, so a failure while copying destroys the
// incomplete device index.
std::unique_ptr<faiss::Index> cloneToDevice(
    faiss::gpu::StandardGpuResources& resources,
    int deviceId,
    const faiss::Index& cpuIndex) {
  if (const auto* flat = dynamic_cast<const faiss::IndexIVFFlat*>(&cpuIndex)) {
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = deviceId;
    config.use_cuvs = false;
    auto gpuIndex = std::make_unique<faiss::gpu::GpuIndexIVFFlat>(
        &resources, flat->d, flat->nlist, flat->metric_type, config);
    gpuIndex->copyFrom(flat);
    return gpuIndex;
  }
  if (const auto* scalarQuantizer =
          dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(&cpuIndex)) {
    faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
    config.device = deviceId;
    config.use_cuvs = false;
    auto gpuIndex = std::make_unique<faiss::gpu::GpuIndexIVFScalarQuantizer>(
        &resources,
        scalarQuantizer->d,
        scalarQuantizer->nlist,
        scalarQuantizer->sq.qtype,
        scalarQuantizer->metric_type,
        scalarQuantizer->by_residual,
        config);
    gpuIndex->copyFrom(scalarQuantizer);
    return gpuIndex;
  }
  if (const auto* productQuantizer =
          dynamic_cast<const faiss::IndexIVFPQ*>(&cpuIndex)) {
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = deviceId;
    config.use_cuvs = false;
    auto gpuIndex = std::make_unique<faiss::gpu::GpuIndexIVFPQ>(
        &resources,
        productQuantizer->d,
        productQuantizer->nlist,
        productQuantizer->pq.M,
        productQuantizer->pq.nbits,
        productQuantizer->metric_type,
        config);
    gpuIndex->copyFrom(productQuantizer);
    return gpuIndex;
  }
  NIMBLE_FAIL("Index type has no GPU implementation");
}

std::shared_ptr<const GpuVectorIndex> GpuVectorIndex::create(
    const faiss::Index& cpuIndex,
    VectorIndexType indexType) {
  NIMBLE_CHECK(supports(indexType), "Index type has no GPU implementation");
  auto& device = GpuVectorIndex::device(faiss::gpu::getCurrentDevice());
  try {
    const std::lock_guard<std::mutex> lock{device.mutex};
    faiss::gpu::DeviceScope scope(device.id);
    auto gpuIndex = cloneToDevice(device.resources, device.id, cpuIndex);
    return std::shared_ptr<const GpuVectorIndex>(
        new GpuVectorIndex(device, std::move(gpuIndex)));
  } catch (const std::exception& error) {
    LOG(WARNING) << "Vector index stays on CPU; GPU setup failed: "
                 << error.what();
    return nullptr;
  }
}

GpuVectorIndex::GpuVectorIndex(
    Device& device,
    std::unique_ptr<faiss::Index> index)
    : device_{device}, index_{std::move(index)} {
  NIMBLE_CHECK_NOT_NULL(index_.get(), "GPU vector index is empty");
}

GpuVectorIndex::~GpuVectorIndex() {
  const std::lock_guard<std::mutex> lock{device_.mutex};
  index_.reset();
}

int64_t GpuVectorIndex::maxSelection() {
  return faiss::gpu::getMaxKSelection(/*use_cuvs=*/false);
}

bool GpuVectorIndex::search(
    faiss::idx_t numQueries,
    const float* queryVectors,
    faiss::idx_t maxNumNeighbors,
    float* scores,
    faiss::idx_t* labels,
    const faiss::SearchParametersIVF& params) const {
  // The classic FAISS GPU kernels ignore a row selector and would return
  // rows the caller excluded. Leave this batch on the CPU.
  if (params.sel != nullptr) {
    return false;
  }
  // Pass a fresh object so CPU-only fields such as quantizer_params do not
  // reach FAISS GPU, which rejects unknown parameter types.
  faiss::SearchParametersIVF gpuParams;
  gpuParams.nprobe = params.nprobe;
  const std::lock_guard<std::mutex> lock{device_.mutex};
  try {
    index_->search(
        numQueries, queryVectors, maxNumNeighbors, scores, labels, &gpuParams);
  } catch (const std::exception& error) {
    LOG(WARNING) << "Vector search stays on CPU; GPU search failed: "
                 << error.what();
    return false;
  }
  return true;
}

} // namespace facebook::nimble::index
