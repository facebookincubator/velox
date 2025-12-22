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

#include "velox/functions/Macros.h"
#include "velox/functions/lib/string/StringCore.h"

namespace facebook::velox::functions {

/// strpos and strrpos functions for varbinary
/// strpos(varbinary, varbinary) → bigint
///     Returns the starting byte position of the first instance of the pattern
///     in the binary data. Positions start with 1. If not found, 0 is returned.
/// strpos(varbinary, varbinary, instance) → bigint
///     Returns the byte position of the N-th instance of the pattern.
///     instance must be a positive number. Positions start with 1. If not
///     found, 0 is returned.
/// strrpos(varbinary, varbinary) → bigint
///     Returns the starting byte position of the first instance of the pattern
///     in the binary data counting from the end. Positions start with 1. If not
///     found, 0 is returned.
/// strrpos(varbinary, varbinary, instance) → bigint
///     Returns the byte position of the N-th instance of the pattern
///     counting from the end. Instance must be a positive number. Positions
///     start with 1. If not found, 0 is returned.
template <typename T, bool lpos>
struct StrPosVarbinaryFunctionBase {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<int64_t>& result,
      const arg_type<Varbinary>& haystack,
      const arg_type<Varbinary>& needle,
      const arg_type<int64_t>& instance = 1) {
    VELOX_USER_CHECK_GT(instance, 0, "'instance' must be a positive number");
    if (needle.size() == 0) {
      result = 1;
      return;
    }

    int64_t byteIndex = -1;
    if constexpr (lpos) {
      byteIndex = stringCore::findNthInstanceByteIndexFromStart(
          std::string_view(haystack.data(), haystack.size()),
          std::string_view(needle.data(), needle.size()),
          instance);
    } else {
      byteIndex = stringCore::findNthInstanceByteIndexFromEnd(
          std::string_view(haystack.data(), haystack.size()),
          std::string_view(needle.data(), needle.size()),
          instance);
    }

    // Return 1-based byte position, or 0 if not found.
    result = byteIndex == -1 ? 0 : byteIndex + 1;
  }
};

template <typename T>
struct StrLPosVarbinaryFunction : public StrPosVarbinaryFunctionBase<T, true> {
};

template <typename T>
struct StrRPosVarbinaryFunction : public StrPosVarbinaryFunctionBase<T, false> {
};

/// contains for varbinary - returns true if the pattern exists in the binary
/// data contains(varbinary, varbinary) → boolean
template <typename T>
struct ContainsVarbinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& result,
      const arg_type<Varbinary>& haystack,
      const arg_type<Varbinary>& needle) {
    if (needle.size() == 0) {
      result = true;
      return;
    }
    auto pos = std::string_view(haystack.data(), haystack.size())
                   .find(std::string_view(needle.data(), needle.size()));
    result = pos != std::string_view::npos;
  }
};

} // namespace facebook::velox::functions
