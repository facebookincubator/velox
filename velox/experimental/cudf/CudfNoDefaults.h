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

// Include this header AFTER all cudf headers in .cpp files under
// velox/experimental/cudf/ to detect accidental use of cudf default arguments
// for stream and mr parameters at compile time.
//
// cudf public APIs declare defaults like:
//   rmm::cuda_stream_view stream = cudf::get_default_stream(),
//   rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()
//
// When a call site omits stream or mr, the compiler generates a call to these
// default getter functions. The __attribute__((error)) redeclarations below
// cause a hard compile error for any such generated call. The noinline
// attribute prevents the inline body of get_current_device_resource_ref() from
// being substituted, ensuring the call (and thus the error) survives.
//
// For intentional explicit usage where you want the current device memory
// resource, use cudf_velox::get_temp_mr(). It calls the underlying RMM
// function directly, bypassing the error-attributed cudf wrapper.

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {

#if defined(__GNUC__) && !defined(__clang__)

__attribute__((
    error("cudf default stream argument used. Pass stream explicitly."),
    noinline)) rmm::cuda_stream_view const get_default_stream();

__attribute__((
    error("cudf default memory resource argument used. Pass mr explicitly."),
    noinline)) rmm::device_async_resource_ref
get_current_device_resource_ref();

#endif

} // namespace cudf

namespace facebook::velox::cudf_velox {

/// Returns the current device memory resource as an async resource reference.
/// Equivalent to cudf::get_current_device_resource_ref(), but bypasses the
/// __attribute__((error)) redeclaration above by calling the underlying RMM
/// function directly. Use this at call sites where you intentionally want
/// the current default device memory resource.
inline rmm::device_async_resource_ref get_temp_mr() {
  return rmm::mr::get_current_device_resource();
}

} // namespace facebook::velox::cudf_velox
