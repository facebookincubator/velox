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
  static StringView entryDelimiter_;
  static StringView keyValueDelimiter_;

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;
  // One parameter
  void call(
      out_type<Map<Varchar, Varchar>>& out,
      const arg_type<Varchar>& input) {
    callImpl(out, input, entryDelimiter_, keyValueDelimiter_);
  }
  // Two parameters
  void call(
      out_type<Map<Varchar, Varchar>>& out,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& entryDelimiter) {
    VELOX_USER_CHECK(
        entryDelimiter.size() >= 1, "entryDelimiter's size should >= 1.");
    callImpl(out, input, entryDelimiter, keyValueDelimiter_);
  }
  // Three parameters
  void call(
      out_type<Map<Varchar, Varchar>>& out,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& entryDelimiter,
      const arg_type<Varchar>& keyValueDelimiter) {
    VELOX_USER_CHECK(
        entryDelimiter.size() >= 1, "entryDelimiter's size should >= 1.");
    VELOX_USER_CHECK(
        keyValueDelimiter.size() >= 1, "keyValueDelimiter's size should >= 1.");
    callImpl(out, input, entryDelimiter, keyValueDelimiter);
  }

 private:
  std::vector<std::pair<int, int>> strStrRex(
      const StringView& allStr,
      const StringView& s) {
    std::vector<std::pair<int, int>> result;
    boost::regex re(s.str());
    boost::smatch match;
    auto inputs = allStr.str();
    std::string::const_iterator search_start = inputs.begin();
    while (boost::regex_search(search_start, inputs.cend(), match, re)) {
      result.push_back(
          {match[0].first - inputs.begin(), match[0].second - inputs.begin()});
      search_start = match[0].second;
    }
    return result;
  }
  std::vector<StringView> splitMult(
      const StringView& target,
      const StringView& delimeter,
      bool onlyOne = false) {
    std::vector<StringView> result;
    auto allIndex = strStrRex(target, delimeter);
    int begin = 0;
    for (int i = 0; i < allIndex.size(); ++i) {
      result.push_back(
          StringView(target.data() + begin, allIndex[i].first - begin));
      begin = allIndex[i].second;
      if (onlyOne) {
        break;
      }
    }
    if (begin <= target.size()) {
      result.push_back(
          StringView(target.data() + begin, target.size() - begin));
    }
    return result;
  }

  void callImpl(
      out_type<Map<Varchar, Varchar>>& out,
      const StringView& input,
      const StringView& entryDelimiter,
      const StringView& keyValueDelimiter) {
    folly::F14FastSet<std::string_view> keys;
    auto entryItems = splitMult(input, entryDelimiter);
    for (const auto& entryItem : entryItems) {
      auto kvItems = splitMult(entryItem, keyValueDelimiter, true);
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
};
template <typename T>
StringView StringToMapFunction<T>::entryDelimiter_(",");

template <typename T>
StringView StringToMapFunction<T>::keyValueDelimiter_(":");

} // namespace facebook::velox::functions::sparksql
