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
#include "velox/functions/sparksql/ToPrettyString.h"

#include <algorithm>
#include <cctype>

#include <folly/String.h>

#include "velox/common/EnumDefine.h"

namespace facebook::velox::functions::sparksql::detail {
namespace {
const auto& binaryOutputStyleNames() {
  static const folly::F14FastMap<BinaryOutputStyle, std::string_view> kNames = {
      {BinaryOutputStyle::kHexDiscrete, "HEX_DISCRETE"},
      {BinaryOutputStyle::kHex, "HEX"},
      {BinaryOutputStyle::kBase64, "BASE64"},
      {BinaryOutputStyle::kUtf8, "UTF-8"},
      {BinaryOutputStyle::kBasic, "BASIC"},
  };
  return kNames;
}
} // namespace

VELOX_DEFINE_ENUM_NAME(BinaryOutputStyle, binaryOutputStyleNames);

BinaryOutputStyle parseBinaryOutputStyle(std::string_view style) {
  std::string normalized{folly::trimWhitespace(style)};
  std::transform(
      normalized.begin(), normalized.end(), normalized.begin(), [](char c) {
        return std::toupper(static_cast<unsigned char>(c));
      });
  if (normalized.empty()) {
    return BinaryOutputStyle::kHexDiscrete;
  }
  const auto result = BinaryOutputStyleName::tryToBinaryOutputStyle(normalized);
  VELOX_USER_CHECK(
      result.has_value(),
      "Unsupported value for binary output style: '{}'. "
      "Expected one of: HEX_DISCRETE, HEX, BASE64, UTF-8, BASIC.",
      style);
  return result.value();
}
} // namespace facebook::velox::functions::sparksql::detail
