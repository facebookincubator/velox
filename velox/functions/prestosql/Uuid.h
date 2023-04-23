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

#include "folly/Random.h"
#include "velox/functions/Macros.h"
#include "velox/type/Uuid.h"

namespace facebook::velox::functions {

template <typename T>
struct UuidFunction {
  static constexpr bool is_deterministic = false;

  FOLLY_ALWAYS_INLINE bool call(Uuid& result) {
    uint128_t lsb = folly::Random::rand64();
    uint128_t msb = folly::Random::rand64();
    result = Uuid((msb << 64) | lsb);
    return true;
  }
};

} // namespace facebook::velox::functions
