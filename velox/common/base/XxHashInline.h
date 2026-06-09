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

// xxhash with XXH_INLINE_ALL is not compatible with WASM native SIMD APIs
// (__wasm_simd128__) so we hide it to fall back to SSE2 compatibility bindings,
// which compile to the same WASM SIMD128 instructions.
#ifdef __EMSCRIPTEN__
#pragma push_macro("__wasm_simd128__")
#undef __wasm_simd128__
#endif
#define XXH_INLINE_ALL
#include <xxhash.h> // @manual=third-party//xxHash:xxhash
#ifdef __EMSCRIPTEN__
#pragma pop_macro("__wasm_simd128__")
#endif
