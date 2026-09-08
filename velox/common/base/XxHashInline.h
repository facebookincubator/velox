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

#define XXH_INLINE_ALL
// Force memcpy-based reads. The default packed-struct method
// (XXH_FORCE_MEMORY_ACCESS=1) auto-selected for GCC on ARM produces wrong
// hashes at -O2 due to a strict-aliasing miscompilation.
#if defined(__aarch64__) && !defined(XXH_FORCE_MEMORY_ACCESS)
#define XXH_FORCE_MEMORY_ACCESS 0
#endif
#include <xxhash.h> // @manual=third-party//xxHash:xxhash
