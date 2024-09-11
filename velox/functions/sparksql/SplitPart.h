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

#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {

/// split_part(string, delimiter, index) -> varchar
///
///     Splits string on delimiter and returns the part at index.
///     Field indexes start with 1.
///     If the index is larger than the number of fields, then empty string is
///     returned. If the index is 0, throws. If the index is negative, returns
///     the part from last to first.
template <typename T>
struct SplitPartFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& delimiter,
      const int32_t& index) {
    return doCall(result, input, delimiter, index);
  }

 private:
  void doCall(
      out_type<Varchar>& output,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& delimiter,
      const int32_t& index) {
    VELOX_USER_CHECK(
        index != 0,
        "The index 0 is invalid. An index shall be either < 0 or > 0");
    if (input.empty()) {
      output.setEmpty();
      return;
    }
    if (delimiter.empty()) {
      if (index == 1 || index == -1) {
        output.setNoCopy(StringView(input.data(), input.size()));
        return;
      }
      output.setEmpty();
      return;
    }

    std::string_view delim =
        std::string_view(delimiter.data(), delimiter.size());
    std::string_view inputSv = std::string_view(input.data(), input.size());
    if (inputSv.find(delim) == std::string_view::npos &&
        (index == 1 || index == -1)) {
      output.setNoCopy(StringView(input.data(), input.size()));
      return;
    }
    if (index > 0) {
      return forwardCount(output, input, inputSv, delim, index);
    } else {
      return backwardCount(output, input, inputSv, delim, -index);
    }
  }

  void forwardCount(
      out_type<Varchar>& output,
      const arg_type<Varchar>& input,
      const std::string_view& inputSv,
      const std::string_view& delimiter,
      const int32_t& index) {
    int64_t iteration = 1;
    size_t curPos = 0;
    while (curPos <= inputSv.size()) {
      size_t start = curPos;
      curPos = inputSv.find(delimiter, curPos);
      if (iteration == index) {
        size_t end = curPos;
        if (end == std::string_view::npos) {
          end = inputSv.size();
        }
        output.setNoCopy(StringView(input.data() + start, end - start));
        return;
      }

      if (curPos == std::string_view::npos) {
        output.setEmpty();
        return;
      }
      curPos += delimiter.size();
      iteration++;
    }
    output.setEmpty();
  }

  void backwardCount(
      out_type<Varchar>& output,
      const arg_type<Varchar>& input,
      const std::string_view& inputSv,
      const std::string_view& delimiter,
      const int32_t& index) {
    int64_t iteration = 1;
    int64_t curPos = inputSv.size() - 1;
    while (curPos >= 0) {
      size_t end = curPos + 1;
      curPos = inputSv.rfind(delimiter, curPos);
      if (iteration == index) {
        size_t start = curPos;
        if (start == std::string_view::npos) {
          start = 0;
        } else {
          start += 1;
        }
        output.setNoCopy(StringView(input.data() + start, end - start));
        return;
      }

      if (curPos == std::string_view::npos) {
        output.setEmpty();
        return;
      }
      curPos -= delimiter.size();
      iteration++;
    }
    output.setEmpty();
  }
};
} // namespace facebook::velox::functions::sparksql
