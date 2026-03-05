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

// Tag-dispatched overload of cudf::get_default_stream() that bypasses the
// __attribute__((error)) poison in CudfNoDefaults.h. Call sites that
// intentionally need the default CUDA stream should use:
//
//   cudf::get_default_stream(cudf::allow_default_stream)
//
// The definition lives in GpuResources.cpp (which does NOT include
// CudfNoDefaults.h, so it can call the real cudf::get_default_stream()).

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

struct allow_default_stream_t {};
constexpr allow_default_stream_t allow_default_stream{};

rmm::cuda_stream_view const get_default_stream(allow_default_stream_t);

} // namespace cudf
