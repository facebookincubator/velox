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

// GPU shadow for velox/common/base/Exceptions.h
//
// =========================================================================
// KNOWN LIMITATION: VELOX_CHECK* macros are currently silent no-ops on GPU.
// =========================================================================
//
// Real Velox throws `VeloxException` on a failed check. C++ exceptions are
// not supported on CUDA device code (you cannot `throw` from inside a
// kernel and unwind across thread blocks), so a direct port of the
// throwing semantics is impossible.
//
// In this MVP foundation PR the macros expand to a no-op variadic
// helper `useArgs(args...)`. The helper:
//
//   * uses each argument (via pass-by-const-ref) so locals computed
//     solely to feed a check (e.g. `upperBound` in
//     `BitCountFunction::call`) don't trigger unused-variable warnings,
//     and parameters in `FOLLY_ALWAYS_INLINE` helpers like `checkRadix`
//     are not flagged as "set but never used" by nvcc warning #550-D;
//   * is `constexpr` and inline-empty, so under optimization the call
//     itself disappears -- only the argument evaluation (reading a
//     register or literal) remains, which is essentially free for the
//     simple expressions that Velox check sites use.
//
// We deliberately do NOT wrap this in `sizeof(...)` (an unevaluated
// context). That would suppress the runtime evaluation but causes nvcc
// to emit "parameter set but never used" warnings on host helpers
// whose entire body is a check.
//
// Any Velox SFI `call()` body that relies on VELOX_USER_CHECK for input
// validation (e.g. `BitCountFunction` requires `2 <= bits <= 64`,
// `DivideFunction` checks for divide-by-zero) will SILENTLY produce a
// wrong result on GPU when the precondition fails.
//
// PLANNED RESOLUTION (PR 3+ / GpuSimpleFunctionAdapter):
// The adapter that wraps `Fn::call()` into a CUDA kernel will introduce
// a per-row error-propagation mechanism. Concretely, one of:
//   - A per-row `gpu_error_t` argument threaded into the macro body,
//     so a failed check flips the output row's null bit and records an
//     error code that the host inspects post-kernel.
//   - A device-side error flag with a status word, returned to the
//     host through a stream-scoped status buffer.
//
// Until that lands, treat these macros as advisory documentation of
// preconditions that GPU callers must enforce upstream (input filtering,
// null mask combining, etc.) before invoking a `Fn<GpuExec>::call()`.
//
// Grep `TODO(gpu-sfi-checks)` to find every site that needs an update
// when the error-propagation design lands.
#pragma once

namespace facebook::velox::gpu_shadow_detail {

// Variadic no-op consumer used by the VELOX_CHECK* shadow macros. The
// function body is empty; under optimization the call is elided and
// only the (cheap) argument evaluation remains. Pass-by-const-ref
// ensures both unused-variable warnings AND "parameter set but never
// used" warnings on host helpers are suppressed.
template <typename... Ts>
constexpr void useArgs(const Ts&...) {}

} // namespace facebook::velox::gpu_shadow_detail

// Shared expansion for all no-op check macros. We evaluate each
// argument (essentially free for the simple expressions Velox uses)
// and discard the result. See file-level header for the trade-off
// rationale (`sizeof` unevaluated context would be cheaper but causes
// nvcc warning #550-D on host helpers whose body is purely checks).
#define VELOX_GPU_SHADOW_NOOP_CHECK(...) \
  ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__)

// TODO(gpu-sfi-checks): wire into per-row error-propagation in adapter.
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

// TODO(gpu-sfi-checks): wire into per-row error-propagation in adapter.
#ifndef VELOX_NYI
#define VELOX_NYI(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_UNSUPPORTED
#define VELOX_UNSUPPORTED(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_ARITHMETIC_ERROR
#define VELOX_ARITHMETIC_ERROR(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
