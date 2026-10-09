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

// GPU shadow for velox/common/base/Exceptions.h.
//
// Device code cannot throw, so the check macros evaluate their arguments and
// discard them. A call() body that validates its input with VELOX_USER_CHECK,
// such as BitCountFunction or DivideFunction, therefore returns an unchecked
// result on the GPU. TODO(gpu-sfi-checks): report a failed check per row.
#pragma once

namespace facebook::velox::gpu_shadow_detail {

// Uses its arguments, so a local that exists only to feed a check is not
// reported as unused. Not sizeof(...): nvcc then warns (#550-D) that a host
// helper whose body is only a check sets a parameter it never uses.
template <typename... Ts>
constexpr void useArgs(const Ts&...) {}

} // namespace facebook::velox::gpu_shadow_detail

#define VELOX_GPU_SHADOW_NOOP_CHECK(...) \
  ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__)

#ifndef VELOX_CHECK
#define VELOX_CHECK(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_EQ
#define VELOX_CHECK_EQ(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_NE
#define VELOX_CHECK_NE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_LT
#define VELOX_CHECK_LT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_LE
#define VELOX_CHECK_LE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_GT
#define VELOX_CHECK_GT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_GE
#define VELOX_CHECK_GE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_NOT_NULL
#define VELOX_CHECK_NOT_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_NULL
#define VELOX_CHECK_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_FAIL
#define VELOX_FAIL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif

#ifndef VELOX_UNREACHABLE
#define VELOX_UNREACHABLE(...) __builtin_unreachable()
#endif

#ifndef VELOX_USER_CHECK
#define VELOX_USER_CHECK(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_EQ
#define VELOX_USER_CHECK_EQ(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_NE
#define VELOX_USER_CHECK_NE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_LT
#define VELOX_USER_CHECK_LT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_LE
#define VELOX_USER_CHECK_LE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_GT
#define VELOX_USER_CHECK_GT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_GE
#define VELOX_USER_CHECK_GE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_NOT_NULL
#define VELOX_USER_CHECK_NOT_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_FAIL
#define VELOX_USER_FAIL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif

#ifndef VELOX_DCHECK
#define VELOX_DCHECK(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_EQ
#define VELOX_DCHECK_EQ(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_NE
#define VELOX_DCHECK_NE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_LT
#define VELOX_DCHECK_LT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_LE
#define VELOX_DCHECK_LE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_GT
#define VELOX_DCHECK_GT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_GE
#define VELOX_DCHECK_GE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_NOT_NULL
#define VELOX_DCHECK_NOT_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif

#ifndef VELOX_NYI
#define VELOX_NYI(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_UNSUPPORTED
#define VELOX_UNSUPPORTED(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_ARITHMETIC_ERROR
#define VELOX_ARITHMETIC_ERROR(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
