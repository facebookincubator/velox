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

#include <cstdint>
#include <memory>

#include "velox/common/vectorindex/LocalVectorIndexAccelerator.h"

namespace facebook::velox::common::vectorindex {

struct GpuLocalVectorIndexOptions {
  /// The CUDA device that holds the device copies.
  int32_t device{0};

  /// Local indexes with fewer embeddings are searched on the CPU. Each search
  /// copies its queries to the device and its results back, which outweighs
  /// the search itself for a small index.
  int64_t minSearchSize{0};

  /// Trains and fills kIvfFlat and kIvfSq local indexes on the device. The
  /// bytes match the CPU build's layout, though not its bytes.
  bool build{false};

  /// Local indexes with fewer embeddings are built on the CPU.
  int64_t minBuildSize{0};

  /// Loads every local index onto the device, ignoring 'minSearchSize'. A
  /// kHnswSq index too small for an HNSW graph becomes an exact flat index.
  /// A larger kHnswSq index has no FAISS GPU implementation and the load
  /// throws rather than leaving the index on the CPU.
  bool requireDevice{false};
};

/// Returns an accelerator that runs local indexes on a GPU through FAISS's
/// own GPU kernels. kFlat, kIvfFlat and kIvfSq are copied with
/// index_cpu_to_gpu. kHnswSq stays on the CPU, except a graph too small to
/// search, which is an exact flat index when 'requireDevice' is set. Searches
/// on one device run one at a time. Throws if 'options.device' is not a CUDA
/// device.
std::shared_ptr<const LocalVectorIndexAccelerator>
makeGpuLocalVectorIndexAccelerator(
    const GpuLocalVectorIndexOptions& options = {});

/// The number of CUDA devices, 0 if there is no usable CUDA driver.
int32_t gpuLocalVectorIndexDevices();

} // namespace facebook::velox::common::vectorindex
