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

namespace facebook::velox::cudf_velox {

/// Custom memory resource tag used for cuDF device memory accounting.
inline constexpr std::string_view kCudfMemoryResourceTag{"gpu"};

/// Creates the process GPU resource used to arbitrate capacity among query GPU
/// roots. The SharedArbitrator first transfers unused capacity and then asks
/// reclaimable query roots to spill used device memory.
std::shared_ptr<memory::CustomMemoryResource> createCudfCustomMemoryResource(
    int64_t capacity);

/// Installs one process-wide GPU resource in the global custom-resource
/// registry. An application-provided resource under the "gpu" tag is reused.
std::shared_ptr<memory::CustomMemoryResource> registerCudfMemoryResource(
    int64_t capacity);

/// Removes a GPU resource installed by registerCudfMemoryResource. Query-scoped
/// registries and RMM wrappers retain it while their pools or vectors live.
void unregisterCudfMemoryResource();

/// Returns the resource selected by registerCudfMemoryResource, or nullptr.
std::shared_ptr<memory::CustomMemoryResource> cudfCustomMemoryResource();

} // namespace facebook::velox::cudf_velox
