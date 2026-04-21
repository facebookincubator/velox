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

// GPU shadow for velox/type/Type.h
// Provides minimal tag type declarations needed by Velox function headers.
// The full Velox Type.h pulls in Folly, fmt, and the entire type system.
#pragma once

#include <cstdint>

namespace facebook::velox {

struct Varchar {};
struct Varbinary {};
struct Date {};
struct IntervalDayTime {};
struct IntervalYearMonth {};
struct Time {};

template <typename P, typename S>
struct ShortDecimal {};

template <typename P, typename S>
struct LongDecimal {};

template <typename TKey, typename TVal>
struct Map {};

template <typename TElement>
struct Array {};

template <typename... T>
struct Row {};

using int128_t = __int128;

} // namespace facebook::velox
