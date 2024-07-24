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

#include "folly/container/F14Set.h"
#include "velox/functions/Udf.h"

namespace facebook::velox::functions::sparksql {

template <typename T>
struct StringToMapFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;
  void call(
      out_type<Map<Varchar, Varchar>>& out,
      const arg_type<Varchar>& input) {
    callImpl(out, input, entryDelimiter_, keyValueDelimiter_);
  }
  void call(
      out_type<Map<Varchar, Varchar>>& out,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& entryDelimiter) {
    VELOX_USER_CHECK_GE(
        entryDelimiter.size(), 1, "entryDelimiter's size should >= 1.");
    callImpl(out, input, entryDelimiter, keyValueDelimiter_);
  }
  void call(
      out_type<Map<Varchar, Varchar>>& out,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& entryDelimiter,
      const arg_type<Varchar>& keyValueDelimiter) {
    VELOX_USER_CHECK_GE(
        entryDelimiter.size(), 1, "entryDelimiter's size should >= 1.");
    VELOX_USER_CHECK_GE(
        keyValueDelimiter.size(), 1, "keyValueDelimiter's size should >= 1.");
    callImpl(out, input, entryDelimiter, keyValueDelimiter);
  }

 private:
  /**
   * @brief Retrieve the index of the pattern string
   *
   * This function retrieves all subscripts in the target string that match the
   * start and end of the pattern
   *
   * @param targetStr The target string to search for
   * @param pattern Search pattern.
   */
  std::vector<std::pair<int, int>> findPatternIndex(
      const StringView& targetStr,
      const StringView& pattern) {
    std::vector<std::pair<int, int>> result;
    auto patternStr = pattern.str();
    patternStr.insert(patternStr.begin(), '(');
    patternStr.push_back(')');
    re2::RE2 re(patternStr);
    auto target = targetStr.str();
    re2::StringPiece sp(target);
    re2::StringPiece match;
    while (re2::RE2::FindAndConsume(&sp, re, &match)) {
      size_t startIndex = match.data() - target.data();
      size_t endIndex = startIndex + match.size();
      result.push_back({startIndex, endIndex});
    }
    return result;
  }
  /**
   * @brief Separate the target string based on the delimiter
   *
   * This function will separate the target string based on a delimiter and
   * return an array of separated strings
   *
   * @param targetStr The target string to be separated
   * @param delimeter Delimiter
   * @param onlyOne When encountering the first delimiter, simply separate the
   * string into two longer parts
   */
  std::vector<StringView> split(
      const StringView& targetStr,
      const StringView& delimeter,
      bool onlyOne = false) {
    std::vector<StringView> result;
    auto allIndex = findPatternIndex(targetStr, delimeter);
    int begin = 0;
    for (int i = 0; i < allIndex.size(); ++i) {
      result.push_back(
          StringView(targetStr.data() + begin, allIndex[i].first - begin));
      begin = allIndex[i].second;
      if (onlyOne) {
        break;
      }
    }
    if (begin <= targetStr.size()) {
      result.push_back(
          StringView(targetStr.data() + begin, targetStr.size() - begin));
    }
    return result;
  }

  void callImpl(
      out_type<Map<Varchar, Varchar>>& out,
      const StringView& input,
      const StringView& entryDelimiter,
      const StringView& keyValueDelimiter) {
    folly::F14FastSet<std::string_view> keys;
    auto entryItems = split(input, entryDelimiter);
    for (const auto& entryItem : entryItems) {
      auto kvItems = split(entryItem, keyValueDelimiter);
      VELOX_USER_CHECK(
          keys.insert(kvItems[0]).second,
          "Duplicate keys are not allowed: '{}'.",
          kvItems[0]);
      if (kvItems.size() > 1) {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter.setNoCopy(kvItems[0]);
        valueWriter.setNoCopy(kvItems[1]);
      } else {
        out.add_null().setNoCopy(kvItems[0]);
      }
    }
  }

  static StringView entryDelimiter_;
  static StringView keyValueDelimiter_;
};
template <typename T>
StringView StringToMapFunction<T>::entryDelimiter_(",");

template <typename T>
StringView StringToMapFunction<T>::keyValueDelimiter_(":");

} // namespace facebook::velox::functions::sparksql
