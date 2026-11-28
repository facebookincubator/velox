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
// discard them, and a body keeps running after a check fails. The result can be
// worse than unchecked: abs() of the smallest TINYINT returns the input where
// the CPU raises an error, and bit_count() with a width its check rejects makes
// countBits read past the one word it is given. The macros apply in nvcc's host
// pass too, so host code in a shadow-compiled .cu loses its checks as well.
// Until a failed check is reported per row, no function whose call(), or a
// helper it calls, reaches a check may be registered for the GPU.
// TODO(gpu-sfi-checks): report a failed check per row.
//
// The macros are defined unconditionally, so if the real header is also
// included after this one, the compiler reports each redefinition instead of
// one set silently winning.
#pragma once

#include <cstdlib>

// The real header defines the same macros to throw. Reaching it first means the
// target does not put gpu_shadows/ ahead of the Velox source root.
#ifdef VELOX_CHECK
#error "The real Exceptions.h was included before its GPU shadow"
#endif

namespace facebook::velox::gpu_shadow_detail {

// Uses its arguments, so a local that exists only to feed a check is not
// reported as unused. Not sizeof(...): nvcc then warns (#550-D) that a host
// helper whose body is only a check sets a parameter it never uses.
template <typename... Ts>
constexpr void useArgs(const Ts&...) {}

} // namespace facebook::velox::gpu_shadow_detail

#define VELOX_GPU_SHADOW_NOOP_CHECK(...) \
  ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__)

#define VELOX_CHECK(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_CHECK_EQ(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_CHECK_NE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_CHECK_LT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_CHECK_LE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_CHECK_GT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_CHECK_GE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_CHECK_NOT_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_CHECK_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_FAIL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)

// Fails loudly rather than marking the branch undefined, which would let the
// optimizer delete the code around it. __trap() is not declared noreturn, so
// __builtin_unreachable() follows it to end a function that returns a value.
// nvcc honors it only as a statement: in a comma expression it still warns
// about the missing return (#940-D). The trap still runs first.
#ifdef __CUDA_ARCH__
#define VELOX_UNREACHABLE(...) \
  do {                         \
    __trap();                  \
    __builtin_unreachable();   \
  } while (0)
#else
#define VELOX_UNREACHABLE(...) std::abort()
#endif

#define VELOX_USER_CHECK(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_EQ(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_NE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_LT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_LE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_GT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_GE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_NOT_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_USER_FAIL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)

#define VELOX_DCHECK(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_EQ(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_NE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_LT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_LE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_GT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_GE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_NOT_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)

#define VELOX_NYI(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_UNSUPPORTED(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#define VELOX_ARITHMETIC_ERROR(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
