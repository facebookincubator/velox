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

// GPU shadow for folly/Likely.h.
//
// The real header routes FOLLY_LIKELY through folly/lang/Builtin.h, which
// does not parse under nvcc. Nothing here has GPU-specific meaning: the
// definitions below expand to the same __builtin_expect the real header
// produces on GCC and Clang. The shadow exists purely to keep the include
// from dragging in the rest of Folly.
//
// See the ODR note in this directory's CPortability.h; the same discipline
// applies.
#pragma once

#ifndef FOLLY_BUILTIN_EXPECT
#define FOLLY_BUILTIN_EXPECT(exp, val) __builtin_expect((exp), (val))
#endif

#ifndef FOLLY_LIKELY
#define FOLLY_LIKELY(...) FOLLY_BUILTIN_EXPECT((__VA_ARGS__), 1)
#endif

#ifndef FOLLY_UNLIKELY
#define FOLLY_UNLIKELY(...) FOLLY_BUILTIN_EXPECT((__VA_ARGS__), 0)
#endif

#ifndef LIKELY
#define LIKELY(x) (__builtin_expect((x), 1))
#endif

#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif
