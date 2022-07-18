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

#include "DecimalUtils.h"

namespace facebook::velox {

template <>
std::string decimalToString<LongDecimal>(
    const LongDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asLongDecimal();
  return formatDecimal(decimalType.scale(), value.unscaledValue());
}

template <>
std::string decimalToString<ShortDecimal>(
    const ShortDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asShortDecimal();
  return formatDecimal(decimalType.scale(), value.unscaledValue());
}
} // namespace facebook::velox