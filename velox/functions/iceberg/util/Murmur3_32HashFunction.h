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

#include <folly/CPortability.h>
#include <cstdint>

namespace facebook::velox::functions::iceberg::util {
class Murmur3_32HashFunction {
 public:
  static int hashBigint(uint64_t input);

  static int32_t hashString(const char* const input, uint32_t len);

 private:
  FOLLY_ALWAYS_INLINE static uint32_t mixK1(uint32_t k1);

  FOLLY_ALWAYS_INLINE static uint32_t mixH1(uint32_t h1, uint32_t k1);

  FOLLY_ALWAYS_INLINE static uint32_t fmix(uint32_t h1, uint32_t length);

  static constexpr int kSeed = 0;
};
} // namespace facebook::velox::functions::iceberg::util
