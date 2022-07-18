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

#include <string>
#include "velox/common/base/Exceptions.h"
#include "velox/type/LongDecimal.h"
#include "velox/type/ShortDecimal.h"
#include "velox/type/Type.h"

namespace facebook::velox {

static const int128_t POWERS_OF_TEN[]{
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
    10000000000,
    100000000000,
    1000000000000,
    10000000000000,
    100000000000000,
    1000000000000000,
    10000000000000000,
    100000000000000000,
    1000000000000000000,
    1000000000000000000 * (int128_t)10,
    1000000000000000000 * (int128_t)100,
    1000000000000000000 * (int128_t)1000,
    1000000000000000000 * (int128_t)10000,
    1000000000000000000 * (int128_t)100000,
    1000000000000000000 * (int128_t)1000000,
    1000000000000000000 * (int128_t)10000000,
    1000000000000000000 * (int128_t)100000000,
    1000000000000000000 * (int128_t)1000000000,
    1000000000000000000 * (int128_t)10000000000,
    1000000000000000000 * (int128_t)100000000000,
    1000000000000000000 * (int128_t)1000000000000,
    1000000000000000000 * (int128_t)10000000000000,
    1000000000000000000 * (int128_t)100000000000000,
    1000000000000000000 * (int128_t)1000000000000000,
    1000000000000000000 * (int128_t)10000000000000000,
    1000000000000000000 * (int128_t)100000000000000000,
    1000000000000000000 * (int128_t)1000000000000000000,
    1000000000000000000 * (int128_t)1000000000000000000 * (int128_t)10,
    1000000000000000000 * (int128_t)1000000000000000000 * (int128_t)100};

std::string formatAsDecimal(uint8_t scale, int128_t unscaledValue);

template <typename T>
inline std::string decimalToString(const T& value, const TypePtr& type) {
  VELOX_UNSUPPORTED();
}

template <>
inline std::string decimalToString<LongDecimal>(
    const LongDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asLongDecimal();
  return formatAsDecimal(decimalType.scale(), value.unscaledValue());
}

template <>
inline std::string decimalToString<ShortDecimal>(
    const ShortDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asShortDecimal();
  return formatAsDecimal(decimalType.scale(), value.unscaledValue());
}

} // namespace facebook::velox
