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

// Macros to disable deprecation warnings
#ifdef __clang__
#define VELOX_SUPPRESS_STRINGOP_OVERFLOW_WARNING
#define VELOX_UNSUPPRESS_STRINGOP_OVERFLOW_WARNING
#else
#define VELOX_SUPPRESS_STRINGOP_OVERFLOW_WARNING \
  _Pragma("GCC diagnostic push");                \
  _Pragma("GCC diagnostic ignored \"-Wstringop-overflow\"")
#define VELOX_UNSUPPRESS_STRINGOP_OVERFLOW_WARNING \
  _Pragma("GCC diagnostic pop");
#endif

// Disable deprecated-declarations for Clang and GCC
#ifdef __clang__
#define VELOX_SUPPRESS_DEPRECATED_WARNING \
  _Pragma("clang diagnostic push");       \
  _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")

#define VELOX_UNSUPPRESS_DEPRECATED_WARNING _Pragma("clang diagnostic pop")
#else
#define VELOX_SUPPRESS_DEPRECATED_WARNING \
  _Pragma("GCC diagnostic push");         \
  _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

#define VELOX_UNSUPPRESS_DEPRECATED_WARNING _Pragma("GCC diagnostic pop")
#endif

// Disable missing-field-initializers for Clang and GCC
#ifdef __clang__
#if defined(__has_warning) && __has_warning("-Wmissing-field-initializers")
#define VELOX_SUPPRESS_MISSING_DESIGNATED_FIELD_INITIALIZERS_WARNING \
  _Pragma("clang diagnostic push");                                  \
  _Pragma("clang diagnostic ignored \"-Wmissing-field-initializers\"")
#define VELOX_UNSUPPRESS_MISSING_DESIGNATED_FIELD_INITIALIZERS_WARNING \
  _Pragma("clang diagnostic pop");
#else
#define VELOX_SUPPRESS_MISSING_DESIGNATED_FIELD_INITIALIZERS_WARNING
#define VELOX_UNSUPPRESS_MISSING_DESIGNATED_FIELD_INITIALIZERS_WARNING
#endif
#else
#define VELOX_SUPPRESS_MISSING_DESIGNATED_FIELD_INITIALIZERS_WARNING
#define VELOX_UNSUPPRESS_MISSING_DESIGNATED_FIELD_INITIALIZERS_WARNING
#endif

#define VELOX_CONCAT(x, y) x##y
// Need this extra layer to expand __COUNTER__.
#define VELOX_VARNAME_IMPL(x, y) VELOX_CONCAT(x, y)
#define VELOX_VARNAME(x) VELOX_VARNAME_IMPL(x, __COUNTER__)

// Workaround for GCC bug, it was fixed only in GCC 13.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=93413
// TLDR: GCC 12 and earlier do not support constexpr static variables for
// non-template classes with virtual default destructors.
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 13
#define VELOX_CONSTEXPR_SINGLETON static const
#else
#define VELOX_CONSTEXPR_SINGLETON static constexpr
#endif

// Marks a function as one that GPU execution is expected to be able to call.
//
// Under nvcc this expands to `__host__ __device__`, so the function is
// compiled for the device as well as the host. Under any host-only compiler it
// expands to nothing, leaving CPU builds unchanged.
//
// Two things it deliberately is not:
//
//   - It is not an inlining hint. A non-template function defined in a header
//     still needs its own `inline`; write `VELOX_GPU_COMPATIBLE inline void
//     f()`. Templates and in-class member functions are already implicitly
//     inline and need nothing extra.
//   - It is not a guarantee. nvcc reports a call from an annotated function
//     into host-only code as a *warning*, then emits a kernel that silently
//     computes the wrong answer. Anything carrying this macro therefore has to
//     stay within what device code can reach: no exceptions, no allocation, no
//     runtime indexing of a namespace- or class-scope constexpr table, and no
//     compiler builtin lacking a device implementation. Build GPU targets with
//     `--diag-error=20011` so the compiler enforces that rather than a reader.
#ifdef __CUDACC__
#define VELOX_GPU_COMPATIBLE __host__ __device__
#else
#define VELOX_GPU_COMPATIBLE
#endif
