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

/// DecimalUtil is a static class that holds helper functions for Decimal Type.
class DecimalUtil {
 public:
  static const int128_t kPowersOfTen[LongDecimalType::kMaxPrecision];

  /// Helper function to convert a decimal value to string.
  template <typename T>
  static std::string decimalToString(const T& value, const TypePtr& type) {
    VELOX_UNSUPPORTED();
  }

  template <>
  static std::string decimalToString<LongDecimal>(
      const LongDecimal& value,
      const TypePtr& type);

  template <>
  static std::string decimalToString<ShortDecimal>(
      const ShortDecimal& value,
      const TypePtr& type);
};
} // namespace facebook::velox
