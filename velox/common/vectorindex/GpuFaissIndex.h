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

#include <bit>
#include <memory>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>

namespace facebook::velox::common::vectorindex {

// Configures the search algorithm and checks the visited-node budget with
// FAISS's default CAGRA search width, iterations, and hash fill rate.
inline bool configureCagraSearch(
    size_t graphDegree,
    size_t numVectors,
    faiss::gpu::SearchParametersCagra& params) {
  // Multi-CTA search reserves 32 of 256 buffer entries for candidates.
  constexpr size_t kMaxMultiCtaGraphDegree{224};
  if (graphDegree <= kMaxMultiCtaGraphDegree) {
    params.algo = faiss::gpu::search_algo::AUTO;
    return true;
  }
  params.algo = faiss::gpu::search_algo::MULTI_KERNEL;

  // cuVS search_plan.cuh allows a 2^13-entry small hash for unfiltered
  // searches and a 2^20-entry regular hash, each at half fill by default.
  constexpr size_t kSmallHashVisitedNodes{1 << 12};
  constexpr size_t kMaxVisitedNodes{1 << 19};
  if (params.itopk_size > kMaxVisitedNodes) {
    return false;
  }
  constexpr size_t kWarpSize{32};
  const auto candidates =
      (params.itopk_size + kWarpSize - 1) / kWarpSize * kWarpSize;
  if (params.sel == nullptr &&
      candidates + graphDegree <= kSmallHashVisitedNodes) {
    return true;
  }

  // Bound cuVS's default iteration count by the padded candidate count
  // plus its initial traversal, which expands by at least two per iteration.
  const auto iterations = candidates + std::bit_width(numVectors - 1);
  return iterations <= (kMaxVisitedNodes - candidates) / graphDegree;
}

inline std::unique_ptr<faiss::Index> cloneCuvsIndex(
    faiss::gpu::GpuResourcesProvider& resources,
    int device,
    const faiss::Index& cpuIndex) {
  // Keep ownership during copies that the pinned Faiss cloner does through
  // raw pointers. Leave Faiss's dispatch for other index types and ID maps.
  struct Cloner final : faiss::gpu::ToGpuCloner {
    using ToGpuCloner::ToGpuCloner;

    faiss::Index* clone_Index(const faiss::Index* index) override {
      if (const auto* flat = dynamic_cast<const faiss::IndexIVFFlat*>(index)) {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = device;
        config.use_cuvs = true;
        return std::make_unique<faiss::gpu::GpuIndexIVFFlat>(
                   provider, flat, config)
            .release();
      }
      if (const auto* sq =
              dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(index)) {
        faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
        config.device = device;
        config.use_cuvs = true;
        return std::make_unique<faiss::gpu::GpuIndexIVFScalarQuantizer>(
                   provider, sq, config)
            .release();
      }
      if (const auto* cagra =
              dynamic_cast<const faiss::IndexHNSWCagra*>(index)) {
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = device;
        auto gpu = std::make_unique<faiss::gpu::GpuIndexCagra>(
            provider, cagra->d, cagra->metric_type, config);
        gpu->copyFrom_ex(cagra, cagra->get_numeric_type());
        return gpu.release();
      }
      return ToGpuCloner::clone_Index(index);
    }
  };

  faiss::gpu::GpuClonerOptions options;
  options.use_cuvs = true;
  Cloner cloner(&resources, device, options);
  return std::unique_ptr<faiss::Index>(cloner.clone_Index(&cpuIndex));
}

} // namespace facebook::velox::common::vectorindex
