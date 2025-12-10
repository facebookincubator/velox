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

#include <cstdint>
#include <limits>

namespace facebook::velox {

using Cost = uint64_t;

// This assumes we wont have signature longer than 1M argument.
inline constexpr Cost kMaxFunctionArgs = 1'000'000;

// This assumes we wont have function rank number greater than 4.
inline constexpr Cost kMaxFunctionRank = 4;

inline constexpr Cost kNullCoercionCost = 1;

inline constexpr Cost kInvalidCost = std::numeric_limits<Cost>::max();

} // namespace facebook::velox
