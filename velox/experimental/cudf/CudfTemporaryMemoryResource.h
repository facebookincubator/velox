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

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <optional>

// Defined inline so that every cuDF library can call get_temp_mr(), not only
// the ones that link the library installing the per-call resources. Only RMM
// headers may be included here: CudfNoDefaults.h includes this header, and
// cuDF headers must come before it.

namespace facebook::velox::cudf_velox {

/// Temporary resource of the cuDF call running on this thread, set by
/// ScopedCudfMemoryResources. Use get_temp_mr() to read it.
inline thread_local std::optional<rmm::device_async_resource_ref>
    scopedTemporaryMemoryResource;

/// Returns the temporary resource selected for the current cuDF call, falling
/// back to RMM's current device resource outside an operator call.
inline rmm::device_async_resource_ref get_temp_mr() {
  return scopedTemporaryMemoryResource.has_value()
      ? *scopedTemporaryMemoryResource
      : rmm::mr::get_current_device_resource_ref();
}

} // namespace facebook::velox::cudf_velox
