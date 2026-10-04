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
// A kernel cannot throw, so a failed check records that its row was declined
// and the body continues; see GpuErrorSink.cuh. The condition is evaluated on
// every row. The message is dropped: the host re-evaluates a declined row
// through Velox, which produces the real message and error code. What travels
// with the row is its class, user or runtime, which decides whether a TRY may
// swallow it.
//
// Every macro the real header defines is defined here, so that a body reaches
// no wrongly defined one;
// velox/experimental/cudf/scripts/check-gpu-shadow-macros.py keeps the two
// lists equal. The DCHECK families follow NDEBUG as the real header does: real
// checks in a debug build, and in a release build nothing, evaluating neither
// the condition nor the message arguments. VELOX_DEBUG_ONLY marks a local that
// only a debug check reads.
//
// The macros are defined unconditionally, so if the real header is also
// included after this one, the compiler reports each redefinition instead of
// one set silently winning.
#pragma once

// The real header defines the same macros to throw. Reaching it first means the
// target does not put gpu_shadows/ ahead of the Velox source root.
#ifdef VELOX_CHECK
#error "The real Exceptions.h was included before its GPU shadow"
#endif

#include "velox/experimental/cudf/functions/GpuErrorSink.cuh"

namespace facebook::velox::gpu_shadow_detail {

// Uses the message arguments, which are not formatted on the device, so that a
// local computed only for a check is not reported as unused (nvcc warning
// #550-D for a FOLLY_ALWAYS_INLINE helper). Called only on the failure path,
// since the condition already uses the operands.
template <typename... Ts>
__host__ __device__ constexpr void useArgs(const Ts&...) {}

} // namespace facebook::velox::gpu_shadow_detail

// A condition that must hold, in the two error classes.
#define VELOX_GPU_SHADOW_CHECK_KIND(kind, cond, ...)              \
  do {                                                            \
    if (!(cond)) {                                                \
      ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__); \
      ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(kind);     \
    }                                                             \
  } while (0)

#define VELOX_GPU_SHADOW_CHECK(cond, ...)                                  \
  VELOX_GPU_SHADOW_CHECK_KIND(                                             \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kRuntimeError, \
      cond __VA_OPT__(, ) __VA_ARGS__)

#define VELOX_GPU_SHADOW_USER_CHECK(cond, ...)                          \
  VELOX_GPU_SHADOW_CHECK_KIND(                                          \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUserError, \
      cond __VA_OPT__(, ) __VA_ARGS__)

// The binary-comparison forms. The operands are used by the condition, so only
// the message arguments go to useArgs.
#define VELOX_GPU_SHADOW_CHECK_OP(a, b, op, ...) \
  VELOX_GPU_SHADOW_CHECK((a)op(b) __VA_OPT__(, ) __VA_ARGS__)

#define VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, op, ...) \
  VELOX_GPU_SHADOW_USER_CHECK((a)op(b) __VA_OPT__(, ) __VA_ARGS__)

// An unconditional failure: VELOX_FAIL and the error-specific forms.
#define VELOX_GPU_SHADOW_FAIL_KIND(kind, ...)                   \
  do {                                                          \
    ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__); \
    ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(kind);     \
  } while (0)

#define VELOX_GPU_SHADOW_FAIL(...)                                        \
  VELOX_GPU_SHADOW_FAIL_KIND(                                             \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kRuntimeError \
          __VA_OPT__(, ) __VA_ARGS__)

#define VELOX_GPU_SHADOW_USER_FAIL(...)                                \
  VELOX_GPU_SHADOW_FAIL_KIND(                                          \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUserError \
          __VA_OPT__(, ) __VA_ARGS__)

// ---------------------------------------------------------------------------
// Runtime errors: VeloxRuntimeError on the real path, which a TRY must not
// swallow.
// ---------------------------------------------------------------------------

#define VELOX_CHECK(...) VELOX_GPU_SHADOW_CHECK(__VA_ARGS__)
#define VELOX_CHECK_EQ(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, == __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_CHECK_NE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, != __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_CHECK_LT(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, < __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_CHECK_LE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, <= __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_CHECK_GT(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, > __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_CHECK_GE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, >= __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_CHECK_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) == nullptr __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_CHECK_NOT_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) != nullptr __VA_OPT__(, ) __VA_ARGS__)

// Checks ok() only: a status on the device carries no message.
#define VELOX_CHECK_OK(expr) VELOX_GPU_SHADOW_CHECK((expr).ok(), #expr)

// A runtime error is already uncatchable by TRY, so these need no class of
// their own.
#define VELOX_CHECK_UNSUPPORTED_INPUT_UNCATCHABLE(...) \
  VELOX_GPU_SHADOW_CHECK(__VA_ARGS__)

#define VELOX_FAIL(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#define VELOX_FAIL_UNSUPPORTED_INPUT_UNCATCHABLE(...) \
  VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#define VELOX_NYI(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
// VELOX_NYI raises a runtime error, and so do these when the condition fails.
#define VELOX_NYI_CHECK(...) VELOX_GPU_SHADOW_CHECK(__VA_ARGS__)
#define VELOX_NYI_CHECK_EQ(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, == __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_NYI_CHECK_NE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, != __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_NYI_CHECK_LT(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, < __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_NYI_CHECK_LE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, <= __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_NYI_CHECK_GT(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, > __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_NYI_CHECK_GE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, >= __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_NYI_CHECK_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) == nullptr __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_NYI_CHECK_NOT_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) != nullptr __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_UNSUPPORTED(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#define VELOX_FILE_NOT_FOUND_ERROR(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#define VELOX_TRACE_LIMIT_EXCEEDED(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)

// Not __builtin_unreachable(): reaching this declines one row rather than
// making the whole launch undefined.
#define VELOX_UNREACHABLE(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)

// ---------------------------------------------------------------------------
// User errors: VeloxUserError on the real path, which a TRY above the
// expression is allowed to turn into a null.
// ---------------------------------------------------------------------------

#define VELOX_USER_CHECK(...) VELOX_GPU_SHADOW_USER_CHECK(__VA_ARGS__)
#define VELOX_USER_CHECK_EQ(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, == __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_CHECK_NE(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, != __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_CHECK_LT(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, < __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_CHECK_LE(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, <= __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_CHECK_GT(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, > __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_CHECK_GE(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, >= __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_CHECK_NULL(p, ...) \
  VELOX_GPU_SHADOW_USER_CHECK((p) == nullptr __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_CHECK_NOT_NULL(p, ...) \
  VELOX_GPU_SHADOW_USER_CHECK((p) != nullptr __VA_OPT__(, ) __VA_ARGS__)

#define VELOX_USER_FAIL(...) VELOX_GPU_SHADOW_USER_FAIL(__VA_ARGS__)

// VELOX_ARITHMETIC_ERROR is a VeloxUserError on the real path.
#define VELOX_ARITHMETIC_ERROR(...) VELOX_GPU_SHADOW_USER_FAIL(__VA_ARGS__)
#define VELOX_SCHEMA_MISMATCH_ERROR(...) VELOX_GPU_SHADOW_USER_FAIL(__VA_ARGS__)

// ---------------------------------------------------------------------------
// Debug-only, following NDEBUG as the real header does. In a release build the
// real forms are VELOX_CHECK(true), which evaluates neither the condition nor
// the message arguments; these expand to nothing for the same reason.
// ---------------------------------------------------------------------------

#ifndef NDEBUG

#define VELOX_DCHECK(...) VELOX_CHECK(__VA_ARGS__)
#define VELOX_DCHECK_EQ(...) VELOX_CHECK_EQ(__VA_ARGS__)
#define VELOX_DCHECK_NE(...) VELOX_CHECK_NE(__VA_ARGS__)
#define VELOX_DCHECK_LT(...) VELOX_CHECK_LT(__VA_ARGS__)
#define VELOX_DCHECK_LE(...) VELOX_CHECK_LE(__VA_ARGS__)
#define VELOX_DCHECK_GT(...) VELOX_CHECK_GT(__VA_ARGS__)
#define VELOX_DCHECK_GE(...) VELOX_CHECK_GE(__VA_ARGS__)
#define VELOX_DCHECK_NULL(...) VELOX_CHECK_NULL(__VA_ARGS__)
#define VELOX_DCHECK_NOT_NULL(...) VELOX_CHECK_NOT_NULL(__VA_ARGS__)

#define VELOX_USER_DCHECK(...) VELOX_USER_CHECK(__VA_ARGS__)
#define VELOX_USER_DCHECK_EQ(...) VELOX_USER_CHECK_EQ(__VA_ARGS__)
#define VELOX_USER_DCHECK_NE(...) VELOX_USER_CHECK_NE(__VA_ARGS__)
#define VELOX_USER_DCHECK_LT(...) VELOX_USER_CHECK_LT(__VA_ARGS__)
#define VELOX_USER_DCHECK_LE(...) VELOX_USER_CHECK_LE(__VA_ARGS__)
#define VELOX_USER_DCHECK_GT(...) VELOX_USER_CHECK_GT(__VA_ARGS__)
#define VELOX_USER_DCHECK_GE(...) VELOX_USER_CHECK_GE(__VA_ARGS__)
#define VELOX_USER_DCHECK_NULL(...) VELOX_USER_CHECK_NULL(__VA_ARGS__)
#define VELOX_USER_DCHECK_NOT_NULL(...) VELOX_USER_CHECK_NOT_NULL(__VA_ARGS__)

#define VELOX_DEBUG_ONLY

#else // NDEBUG

// Kept for bodies that reference their arguments through it; the debug macros
// below do not use it, as in the real release forms.
#define VELOX_GPU_SHADOW_NOOP_CHECK(...) \
  ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__)

#define VELOX_DCHECK(...)
#define VELOX_DCHECK_EQ(...)
#define VELOX_DCHECK_NE(...)
#define VELOX_DCHECK_LT(...)
#define VELOX_DCHECK_LE(...)
#define VELOX_DCHECK_GT(...)
#define VELOX_DCHECK_GE(...)
#define VELOX_DCHECK_NULL(...)
#define VELOX_DCHECK_NOT_NULL(...)

#define VELOX_USER_DCHECK(...)
#define VELOX_USER_DCHECK_EQ(...)
#define VELOX_USER_DCHECK_NE(...)
#define VELOX_USER_DCHECK_LT(...)
#define VELOX_USER_DCHECK_LE(...)
#define VELOX_USER_DCHECK_GT(...)
#define VELOX_USER_DCHECK_GE(...)
#define VELOX_USER_DCHECK_NULL(...)
#define VELOX_USER_DCHECK_NOT_NULL(...)

#define VELOX_DEBUG_ONLY [[maybe_unused]]

#endif // NDEBUG
