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

// GPU shadow for folly/CPortability.h
// Provides no-op stubs for Folly compiler portability macros so that
// Velox function headers can compile under nvcc without pulling in Folly.
#pragma once

#ifndef FOLLY_ALWAYS_INLINE
#ifdef __CUDACC__
#define FOLLY_ALWAYS_INLINE __host__ __device__ inline
#else
#define FOLLY_ALWAYS_INLINE inline
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
