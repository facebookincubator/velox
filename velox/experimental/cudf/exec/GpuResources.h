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

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox {

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;

/// Returns the temporary resource selected for the current cuDF call, falling
/// back to RMM's current device resource outside an operator call.
///
/// Also declared in CudfNoDefaults.h. That header cannot be included here: it
/// poisons cudf's implicit stream and resource getters, which the translation
/// units including this header call explicitly.
rmm::device_async_resource_ref get_temp_mr();

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/// Creates the process-wide cuDF resource used for APIs that still allocate
/// temporary storage through cudf::get_current_device_resource_ref(). While a
/// ScopedCudfMemoryResources is active, allocations and deallocations are
/// dispatched to its temporary resource. Outside a scope they use 'fallback'.
///
/// This is a transitional bridge until every cuDF API accepts temp_mr. Only
/// operational allocations whose allocation call occurs inside the cuDF API
/// scope may use this resource. The selected resource is remembered for
/// deallocation on any thread, but remains non-owning and must outlive the
/// allocation. Returned columns and tables must use the explicit owning output
/// resource instead.
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createThreadLocalTemporaryMemoryResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> fallback);

/// Installs per-call temporary and output resources for helpers that use
/// get_temp_mr() and get_output_mr(). The selected resource objects themselves
/// carry accounting identity, so allocations remain correctly attributed when
/// cuDF uses a different thread or stream after receiving the resource ref.
class ScopedCudfMemoryResources {
 public:
  ScopedCudfMemoryResources(
      rmm::device_async_resource_ref tempMr,
      rmm::device_async_resource_ref outputMr);
  ~ScopedCudfMemoryResources();

  ScopedCudfMemoryResources(const ScopedCudfMemoryResources&) = delete;
  ScopedCudfMemoryResources& operator=(const ScopedCudfMemoryResources&) =
      delete;
  ScopedCudfMemoryResources(ScopedCudfMemoryResources&&) = delete;
  ScopedCudfMemoryResources& operator=(ScopedCudfMemoryResources&&) = delete;

 private:
  std::optional<rmm::device_async_resource_ref> previousTempMr_;
  std::optional<rmm::device_async_resource_ref> previousOutputMr_;
};

/// Creates a memory resource based on the given mode.
///
/// @param mode rmm::mr::pool_memory_resource mode.
/// @param percent The initial percent of GPU memory to allocate for memory
/// resource.
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent);

/// Releases retired UCX exchange resources with no live packed buffers.
/// Returns true when no active or in-use exchange resources remain.
///
/// Defined in CudfMemoryResource.cpp and declared in both headers: callers of
/// this one cannot include CudfMemoryResource.h, which drags in cudf headers
/// that redefine the getters CudfNoDefaults.h poisons.
bool tryResetCudfExchangeMemoryResource();

/// Tears down all process-owned UCX exchange resources. There must be no
/// active query users or live packed buffers.
void resetCudfExchangeMemoryResource();

/// Returns the global CUDA stream pool used by cudf.
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

/// Records the CUDA device owning the primary context, so worker threads can
/// bind it via `ensureCudaContextForThread()`.
///
/// @param device The CUDA device ordinal that was current during registration.
void setCudfContextDevice(int device);

/// Binds the primary CUDA context on the calling thread, once (thread-local).
/// Needed as `cuda::stream_ref` uses the driver API, which does not lazily
/// create a context like the runtime API does.
void ensureCudaContextForThread();

/// Ensures that the calling thread has a CUDA context and returns the
/// `cudf::get_default_stream()`. Use it for host-thread stream work outside
/// cuDF operators.
[[nodiscard]] cuda::stream_ref getDefaultStreamForCurrentThread();

} // namespace facebook::velox::cudf_velox
