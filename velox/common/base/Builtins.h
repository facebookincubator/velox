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

// Provide GCC __builtin_popcount* on MSVC (not compiler builtins).
#if defined(_MSC_VER)
#include <folly/portability/Builtins.h>

// ARM64 MSVC may have either no __builtin_popcount declaration or a definition
// supplied by the Folly port. Use macros so callers compile in both cases
// without adding duplicate function bodies.
#if defined(_M_ARM64) || defined(_M_ARM)
#include <intrin.h>
#ifndef __builtin_popcount
#define __builtin_popcount(x) \
  static_cast<int>(_CountOneBits(static_cast<unsigned long>(x)))
#endif
#ifndef __builtin_popcountl
#define __builtin_popcountl(x) \
  static_cast<int>(_CountOneBits(static_cast<unsigned long>(x)))
#endif
#ifndef __builtin_popcountll
#define __builtin_popcountll(x) \
  static_cast<int>(_CountOneBits64(static_cast<unsigned long long>(x)))
#endif
#endif // ARM64
#endif // _MSC_VER
