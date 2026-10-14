/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/common/memory/CustomMemoryResource.h"

#include <memory>
#include <string_view>

namespace facebook::velox::core {
class QueryCtx;
}

namespace facebook::velox::cudf_velox {

/// Custom memory resource tag used for cuDF device memory accounting.
inline constexpr std::string_view kCudfMemoryResourceTag{"gpu"};

/// Creates the process GPU resource used to arbitrate capacity among query GPU
/// roots. The SharedArbitrator first transfers unused capacity and then asks
/// reclaimable query roots to spill used device memory.
/// Capacity is a logical GPU budget in bytes, independent of RMM pool sizing.
/// The caller must retain an unregistered resource until all its pools are
/// gone.
std::shared_ptr<memory::CustomMemoryResource> createCudfCustomMemoryResource(
    int64_t capacity);

/// Installs one process-wide GPU resource in the global custom-resource
/// registry and retains it for the process lifetime, including after
/// unregisterCudf(). An application-provided resource under the "gpu" tag is
/// reused. Subsequent registration cannot replace it or change its capacity.
/// Register during application startup, before attaching query pools.
std::shared_ptr<memory::CustomMemoryResource> registerCudfMemoryResource(
    int64_t capacity);

/// Returns the resource selected by registerCudfMemoryResource, or nullptr.
std::shared_ptr<memory::CustomMemoryResource> cudfCustomMemoryResource();

/// Adds a per-query GPU root backed by the process-lived cuDF backend.
/// Call once per query after MemoryManager initialization and before creating
/// Tasks. Does nothing before backend registration;
/// no query-scoped ownership of the backend is needed.
void attachCudfMemoryPool(core::QueryCtx& queryCtx);

} // namespace facebook::velox::cudf_velox
