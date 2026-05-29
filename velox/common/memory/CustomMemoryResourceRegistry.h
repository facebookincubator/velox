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

#include <memory>
#include <string>
#include <string_view>

#include "velox/common/ScopedRegistry.h"
#include "velox/common/memory/CustomMemoryResource.h"

namespace facebook::velox::memory {

/// Key under which a per-QueryCtx scoped CustomMemoryResourceRegistry is
/// stored on QueryCtx via QueryCtx::setRegistry / QueryCtx::registry. Tasks
/// use this key to look up resources when building the custom memory pool
/// hierarchy.
inline constexpr std::string_view kCustomMemoryResourceRegistryKey{
    "customMemoryResource"};

/// Entry point for the CustomMemoryResource registry. Provides the
/// process-global root and a factory for scoped registries. Callers
/// register, look up, and clear entries directly on the Registry instance
/// (Registry::insert, Registry::find, Registry::clear). Registry methods
/// are thread-safe.
class CustomMemoryResourceRegistry {
 public:
  using Registry = ScopedRegistry<std::string, CustomMemoryResource>;

  /// Process-global root registry. The reference is stable for the
  /// lifetime of the process.
  static Registry& global();

  /// Creates a scoped registry. Defaults to inheriting from the global
  /// registry; pass nullptr for isolation mode (no fallback).
  static std::shared_ptr<Registry> createRegistry(Registry* parent = &global());
};

} // namespace facebook::velox::memory
