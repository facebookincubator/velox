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
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& entryDelimiter,
      const arg_type<Varchar>& keyValueDelimiter) {
    VELOX_USER_CHECK_EQ(
        entryDelimiter.size(), 1, "entryDelimiter's size should be 1.");
    VELOX_USER_CHECK_EQ(
        keyValueDelimiter.size(), 1, "keyValueDelimiter's size should be 1.");

    callImpl(
        out,
        toStringView(input),
        toStringView(entryDelimiter),
        toStringView(keyValueDelimiter));
  }

 private:
  static std::string_view toStringView(const arg_type<Varchar>& input) {
    return std::string_view(input.data(), input.size());
  }

  void callImpl(
      out_type<Map<Varchar, Varchar>>& out,
      std::string_view input,
      std::string_view entryDelimiter,
      std::string_view keyValueDelimiter) const {
    size_t pos = 0;
    folly::F14FastSet<std::string_view> keys;

    auto nextEntryPos = input.find(entryDelimiter, pos);
    while (nextEntryPos != std::string::npos) {
      processEntry(
          out,
          std::string_view(input.data() + pos, nextEntryPos - pos),
          keyValueDelimiter,
          keys);

      pos = nextEntryPos + 1;
      nextEntryPos = input.find(entryDelimiter, pos);
    }

    processEntry(
        out,
        std::string_view(input.data() + pos, input.size() - pos),
        keyValueDelimiter,
        keys);
  }

  void processEntry(
      out_type<Map<Varchar, Varchar>>& out,
      std::string_view entry,
      std::string_view keyValueDelimiter,
      folly::F14FastSet<std::string_view>& keys) const {
    const auto delimiterPos = entry.find(keyValueDelimiter, 0);
    // Allows keyValue delimiter not found.
    if (delimiterPos == std::string::npos) {
      out.add_null().setNoCopy(StringView(entry));
      return;
    }
    const auto key = std::string_view(entry.data(), delimiterPos);
    VELOX_USER_CHECK(
        keys.insert(key).second, "Duplicate keys are not allowed: '{}'.", key);
    const auto value = StringView(
        entry.data() + delimiterPos + 1, entry.size() - delimiterPos - 1);

    auto [keyWriter, valueWriter] = out.add_item();
    keyWriter.setNoCopy(StringView(key));
    valueWriter.setNoCopy(value);
  }
};

} // namespace facebook::velox::functions::sparksql
