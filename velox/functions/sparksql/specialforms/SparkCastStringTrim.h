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

#include <cstddef>
#include <cstdint>

#include "velox/type/StringView.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {
namespace detail {

// Spark UTF8String.trimAll trims boundary bytes 0x00..0x20 and 0x7F.
inline bool isUtf8StringTrimAllByte(char value) {
  const auto byte = static_cast<uint8_t>(value);
  return byte <= 0x20 || byte == 0x7F;
}

// Java String.trim trims boundary characters <= U+0020.
inline bool isJavaStringTrimByte(char value) {
  return static_cast<uint8_t>(value) <= 0x20;
}

template <typename TShouldTrim>
inline StringView trimBytes(const StringView& view, TShouldTrim shouldTrim) {
  if (view.empty()) {
    return StringView("");
  }

  const auto* data = view.data();
  size_t start = 0;
  size_t end = view.size();

  while (start < end && shouldTrim(data[start])) {
    ++start;
  }

  if (start >= end) {
    return StringView("");
  }

  while (end > start && shouldTrim(data[end - 1])) {
    --end;
  }

  return StringView(data + start, end - start);
}

inline StringView trimUtf8StringAll(const StringView& view) {
  return trimBytes(view, isUtf8StringTrimAllByte);
}

inline StringView trimJavaString(const StringView& view) {
  return trimBytes(view, isJavaStringTrimByte);
}

} // namespace detail

// Keep these target-type-specific rules in sync with Spark. Spark currently
// uses UTF8String-style trimming for most casts but Java String.trim for
// floating point and decimal casts. See
// https://issues.apache.org/jira/browse/SPARK-59182.
inline StringView trimStringForCast(
    const StringView& view,
    const Type& toType) {
  if (toType.isDecimal() || toType.kind() == TypeKind::REAL ||
      toType.kind() == TypeKind::DOUBLE) {
    return detail::trimJavaString(view);
  }

  return detail::trimUtf8StringAll(view);
}

} // namespace facebook::velox::functions::sparksql
