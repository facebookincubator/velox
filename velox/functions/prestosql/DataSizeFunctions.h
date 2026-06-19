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

#include <map>
#include <string>
#include "velox/functions/Macros.h"
#include "velox/functions/lib/string/StringCore.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions {

#ifdef _MSC_VER
// MSVC doesn't support int128_t as enum underlying type.
// Use constexpr values instead.
namespace Unit {
constexpr int128_t BYTE = 1;
constexpr int128_t KILOBYTE = int128_t(1) << 10;
constexpr int128_t MEGABYTE = int128_t(1) << 20;
constexpr int128_t GIGABYTE = int128_t(1) << 30;
constexpr int128_t TERABYTE = int128_t(1) << 40;
constexpr int128_t PETABYTE = int128_t(1) << 50;
constexpr int128_t EXABYTE = int128_t(1) << 60;
constexpr int128_t ZETTABYTE = int128_t(1) << 70;
constexpr int128_t YOTTABYTE = int128_t(1) << 80;
} // namespace Unit
using UnitValue = int128_t;
#else
enum class Unit : int128_t {
  BYTE = 1,
  KILOBYTE = int128_t(1) << 10,
  MEGABYTE = int128_t(1) << 20,
  GIGABYTE = int128_t(1) << 30,
  TERABYTE = int128_t(1) << 40,
  PETABYTE = int128_t(1) << 50,
  EXABYTE = int128_t(1) << 60,
  ZETTABYTE = int128_t(1) << 70,
  YOTTABYTE = int128_t(1) << 80
};
using UnitValue = Unit;
#endif

inline UnitValue parseUnit(std::string_view dataSize, size_t valueLength) {
  static const std::map<std::string_view, UnitValue> unitMap = {
      {"B", Unit::BYTE},
      {"kB", Unit::KILOBYTE},
      {"MB", Unit::MEGABYTE},
      {"GB", Unit::GIGABYTE},
      {"TB", Unit::TERABYTE},
      {"PB", Unit::PETABYTE},
      {"EB", Unit::EXABYTE},
      {"ZB", Unit::ZETTABYTE},
      {"YB", Unit::YOTTABYTE},
  };
  try {
    std::string_view unitString = dataSize.substr(valueLength);
    auto it = unitMap.find(unitString);
    VELOX_USER_CHECK(it != unitMap.end(), "Invalid data size: '{}'", dataSize);
    return it->second;
  } catch (const std::exception&) {
    VELOX_USER_FAIL("Invalid data size: '{}'", dataSize);
  }
}
inline double parseValue(std::string_view dataSize, size_t valueLength) {
  try {
    std::string_view value = dataSize.substr(0, valueLength);
    return folly::to<double>(value);
  } catch (const std::exception&) {
    VELOX_USER_FAIL("Invalid data size: '{}'", dataSize);
  }
}

inline int128_t getDecimal(std::string_view dataSize) {
  size_t valueLength = 0;
  while (valueLength < dataSize.length() &&
         (isdigit(dataSize[valueLength]) || dataSize[valueLength] == '.')) {
    valueLength++;
  }
  VELOX_USER_CHECK_GT(valueLength, 0, "Invalid data size: '{}'", dataSize);
  double value = parseValue(dataSize, valueLength);
  UnitValue unit = parseUnit(dataSize, valueLength);

  int128_t factor = static_cast<int128_t>(unit);
#ifdef _MSC_VER
  // On MSVC, Int128 has no implicit double conversion, so `double * Int128`
  // would truncate the double to int64_t first, losing the fractional part.
  // Multiply in double precision, then convert the result to Int128.
  double product = value * static_cast<double>(factor);
  int128_t scaledValue;
  if (product >= -9.2e18 && product < 9.2e18) {
    scaledValue = static_cast<int128_t>(static_cast<int64_t>(product));
  } else {
    // Large value: construct Int128 from high and low 64-bit parts.
    constexpr double k2pow64 = 18446744073709551616.0; // 2^64
    bool negative = product < 0;
    double absProduct = negative ? -product : product;
    int64_t high = static_cast<int64_t>(absProduct / k2pow64);
    uint64_t low = static_cast<uint64_t>(
        absProduct - static_cast<double>(high) * k2pow64);
    scaledValue = int128_t(high, low);
    if (negative) {
      scaledValue = -scaledValue;
    }
  }
#else
  int128_t scaledValue = static_cast<int128_t>(value * factor);
#endif

  // Ensure the result is within a valid range.
  try {
    DecimalUtil::valueInRange(scaledValue);
    std::vector<char> encodedValue(
        DecimalUtil::getByteArrayLength(scaledValue));
    DecimalUtil::toByteArray(scaledValue, encodedValue.data());
  } catch (const std::exception&) {
    VELOX_USER_FAIL("Value out of range: '{}' ('{}B')", dataSize, scaledValue);
  }
  return scaledValue;
}

template <typename TExec>
struct ParsePrestoDataSizeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);
  static constexpr bool is_default_ascii_behavior = true;
  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<LongDecimal<P1, S1>>& result,
      const arg_type<Varchar>& input) {
    // TODO: Remove explicit std::string_view cast.
    result = getDecimal(std::string_view(input));
  }
  FOLLY_ALWAYS_INLINE void call(
      out_type<LongDecimal<P1, S1>>& result,
      const arg_type<Varchar>& input) {
    // If ASCII input process else fail.
    if (stringCore::isAscii(input.data(), input.size())) {
      // TODO: Remove explicit std::string_view cast.
      result = getDecimal(std::string_view(input));
    } else {
      VELOX_USER_FAIL("Invalid data size: '{}'", input);
    }
  }
};

} // namespace facebook::velox::functions
