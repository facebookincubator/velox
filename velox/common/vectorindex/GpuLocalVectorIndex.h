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
};

/// Returns an accelerator that runs local indexes on a GPU through FAISS GPU
/// and cuVS. kFlat, kIvfFlat, kIvfSq and kCagra indexes are copied to the
/// device as they are: the first three by index_cpu_to_gpu, and kCagra from
/// its stored IndexHNSWCagra. A kHnswSq index is the one type rebuilt, as a
/// CAGRA graph over its reconstructed embeddings, so its scores come from
/// full-precision embeddings rather than the 8-bit codes. 'hnsw.efSearch' is
/// CAGRA's intermediate result count. Searches on one device run one at a
/// time. Throws if 'options.device' is not a CUDA device.
std::shared_ptr<const LocalVectorIndexAccelerator>
makeGpuLocalVectorIndexAccelerator(
    const GpuLocalVectorIndexOptions& options = {});

/// The number of CUDA devices, 0 if there is no usable CUDA driver.
int32_t gpuLocalVectorIndexDevices();

} // namespace facebook::velox::common::vectorindex
