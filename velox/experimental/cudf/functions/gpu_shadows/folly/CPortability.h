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

// GPU shadow for folly/CPortability.h.
//
// Provides no-op stubs for Folly compiler portability macros so that
// Velox function headers can compile under nvcc without pulling in Folly.
//
// =========================================================================
// ODR SAFETY (currently by discipline; `inline namespace` deferred)
// =========================================================================
//
// Several shadows define entities with the SAME qualified name as real
// Velox (e.g. `velox::CompareFlags`, `velox::Status`,
// `bits::countBits`, `functions::plus<T>`) but with DIFFERENT bodies,
// members, or types. In a single linked binary that combines shadow-
// compiled TUs (.cu) and real-Velox TUs (.cpp), this is technically an
// ODR violation. For PR 1+2 the shadow-using target
// (`velox_cudf_gpu_shadow_compile_test`) is an OBJECT library that
// never links into a runnable binary, so the violation is inert.
//
// Once runtime integration begins (PR 5+ -- a CudfFunction wrapper
// living in a .cu library that links against real Velox), the
// recommended remedy is to wrap each ODR-risky shadow's content in
// `inline namespace gpu_shadow_v1 { ... }`. The `inline` part of the
// inline namespace makes the names transparent to qualified lookup --
// SFI code referencing the shadowed entity compiles unchanged -- but
// the mangled symbol carries `gpu_shadow_v1`, so accidental cross-TU
// type passing fails LOUDLY at link time rather than silently
// corrupting at runtime.
//
// This wrapping was prototyped and reverted in PR 1+2 because mixing
// inline-namespaced declarations with non-inline forward declarations
// (e.g. those in `GpuExec.h`) and nested namespaces declared from
// multiple shadows (e.g. `velox::util` shared by Metaprogramming.h
// and FloatingPointUtil.h) triggers compiler ambiguity errors that
// require a coordinated namespace design across both shadow and
// non-shadow headers. Deferring to PR 5+ keeps PR 1+2 focused.
//
// In the meantime, the discipline is: SHADOW TYPES MUST NOT CROSS THE
// .cu / .cpp TU BOUNDARY. Use POD types (cudf::column_view, primitive
// scalars) at any boundary where shadow-compiled code calls into or
// is called by host code using real Velox types.
#pragma once

// Faithful to real Folly. It deliberately does NOT imply __host__ __device__:
// blanket-promoting every annotated function to device code forced nvcc to
// compile bodies nothing calls, and turned this shadow into an ODR hazard for
// any real Folly header that reached it. Functions GPU code actually calls
// carry VELOX_GPU_COMPATIBLE at their definition instead.
#ifndef FOLLY_ALWAYS_INLINE
#define FOLLY_ALWAYS_INLINE inline
#endif

// Feature-detection macros, defined exactly as real Folly defines them.
//
// Unlike FOLLY_ALWAYS_INLINE, these carry no GPU meaning; they are here
// because a Velox header that reaches a *real* Folly header (Traits.h,
// CheckedMath.h, ...) leaves that header expanding `#if FOLLY_HAS_BUILTIN(x)`
// against this shadow. Omitting them does not disable a feature, it produces
// "missing binary operator before token" at the use site.
#ifndef FOLLY_HAS_BUILTIN
#if defined(__has_builtin)
#define FOLLY_HAS_BUILTIN(...) __has_builtin(__VA_ARGS__)
#else
#define FOLLY_HAS_BUILTIN(...) 0
#endif
#endif

#ifndef FOLLY_HAS_FEATURE
#if defined(__has_feature)
#define FOLLY_HAS_FEATURE(...) __has_feature(__VA_ARGS__)
#else
#define FOLLY_HAS_FEATURE(...) 0
#endif
#endif

#ifndef FOLLY_HAS_WARNING
#if defined(__has_warning)
#define FOLLY_HAS_WARNING(...) __has_warning(__VA_ARGS__)
#else
#define FOLLY_HAS_WARNING(...) 0
#endif
#endif

#ifndef FOLLY_NOINLINE
#define FOLLY_NOINLINE
#endif

#ifndef FOLLY_FALLTHROUGH
#define FOLLY_FALLTHROUGH [[fallthrough]]
#endif

#ifndef LIKELY
#define LIKELY(x) (__builtin_expect((x), 1))
#endif

#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif
