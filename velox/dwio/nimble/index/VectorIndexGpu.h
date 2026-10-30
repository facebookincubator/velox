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
#pragma once

#include <cstdint>
#include <memory>

#include <faiss/Index.h>
#include <faiss/IndexIVF.h>

#include "velox/dwio/nimble/index/VectorIndexConfig.h"

namespace facebook::nimble::index {

/// Holds a cuVS-backed GPU copy of a Nimble vector index.
///
/// IVF-Flat, IVF-SQ8 and IVF-PQ are FAISS cuVS clones. CAGRA is a FAISS
/// GpuIndexCagra clone of the stored graph. HNSW SQ8 has no GPU clone, so the
/// GPU index is a CAGRA graph built from the reconstructed vectors. RaBitQ
/// has no FAISS GPU index, so the GPU index is a cuVS IVF-RaBitQ index built
/// from the decoded vectors and only L2 search without a row filter.
///
/// FAISS GPU resources are not thread-safe, so searches on the same device
/// are serialized.
class GpuVectorIndex {
 public:
  /// Returns the number of visible CUDA devices.
  static int numDevices();

  /// Returns whether this build can search the index type on the GPU.
  static bool supports(VectorIndexType indexType);

  /// Copies an index of a supported type to the current CUDA device. Returns
  /// null when cuVS rejects the index configuration.
  static std::shared_ptr<const GpuVectorIndex> create(
      const faiss::Index& cpuIndex,
      VectorIndexType indexType);

  ~GpuVectorIndex();

  /// Returns the largest Top-K and probe count the GPU IVF kernels accept.
  static int64_t maxSelection();

  /// Searches an IVF or RaBitQ index. Host pointers are copied to and from
  /// the device. Only nprobe and sel are read from 'params'. Returns false
  /// when the index cannot apply 'params' and the caller should use the CPU.
  bool search(
      faiss::idx_t numQueries,
      const float* queryVectors,
      faiss::idx_t maxNumNeighbors,
      float* scores,
      faiss::idx_t* labels,
      const faiss::SearchParametersIVF& params) const;

  /// Searches a CAGRA graph. 'intermediateResults' is the CAGRA itopk size
  /// and must be at least 'maxNumNeighbors'. 'selector' may be null.
  void searchCagra(
      faiss::idx_t numQueries,
      const float* queryVectors,
      faiss::idx_t maxNumNeighbors,
      float* scores,
      faiss::idx_t* labels,
      size_t intermediateResults,
      faiss::IDSelector* selector) const;

 private:
  struct Device;
  struct IvfRaBitQIndex;

  // Returns the process-wide state for one CUDA device.
  static Device& device(int deviceId);

  GpuVectorIndex(
      Device& device,
      std::unique_ptr<faiss::Index> index,
      std::unique_ptr<IvfRaBitQIndex> raBitQ);

  // Owns the FAISS resources and the lock shared by indexes on one device.
  Device& device_;

  // Released under the device lock because FAISS frees through 'device_'.
  std::unique_ptr<faiss::Index> index_;

  // cuVS IVF-RaBitQ index. Search mutates internal scratch state.
  mutable std::unique_ptr<IvfRaBitQIndex> raBitQ_;
};

/// Builds a populated IndexHNSWCagra on the GPU. Returns null when no device
/// is visible or cuVS rejects the build, so the caller can build on the CPU.
std::unique_ptr<faiss::Index> buildGpuCagraIndex(
    int dimensions,
    int numConnections,
    faiss::MetricType metric,
    faiss::idx_t numVectors,
    const float* vectors);

} // namespace facebook::nimble::index
