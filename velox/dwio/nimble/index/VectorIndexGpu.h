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

/// Holds a FAISS GPU copy of a Nimble IVF index.
///
/// IVF-Flat, IVF-SQ8 and IVF-PQ are cloned with FAISS's own GPU kernels.
/// HNSW and RaBitQ stay on the CPU. FAISS GPU resources are not thread-safe,
/// so searches on the same device are serialized.
class GpuVectorIndex {
 public:
  /// Returns the number of visible CUDA devices.
  static int numDevices();

  /// Returns whether this build can search the index type on the GPU.
  static bool supports(VectorIndexType indexType);

  /// Copies an index of a supported type to the current CUDA device. Returns
  /// null when FAISS rejects the index configuration.
  static std::shared_ptr<const GpuVectorIndex> create(
      const faiss::Index& cpuIndex,
      VectorIndexType indexType);

  ~GpuVectorIndex();

  /// Returns the largest Top-K and probe count the GPU IVF kernels accept.
  static int64_t maxSelection();

  /// Searches an IVF index. Host pointers are copied to and from the device.
  /// Only nprobe and sel are read from 'params'. Returns true when the search
  /// ran on the device.
  bool search(
      faiss::idx_t numQueries,
      const float* queryVectors,
      faiss::idx_t maxNumNeighbors,
      float* scores,
      faiss::idx_t* labels,
      const faiss::SearchParametersIVF& params) const;

 private:
  struct Device;

  // Returns the process-wide state for one CUDA device.
  static Device& device(int deviceId);

  GpuVectorIndex(Device& device, std::unique_ptr<faiss::Index> index);

  // Owns the FAISS resources and the lock shared by indexes on one device.
  Device& device_;

  // Released under the device lock because FAISS frees through 'device_'.
  std::unique_ptr<faiss::Index> index_;
};

} // namespace facebook::nimble::index
