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

#include <optional>
#include <string_view>
#include <vector>

#include "folly/container/F14Map.h"
#include "velox/core/QueryConfig.h"
#include "velox/functions/Udf.h"

namespace facebook::velox::functions::sparksql {

template <typename T>
struct StringToMapFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* /*entryDelimiter*/,
      const arg_type<Varchar>* /*keyValueDelimiter*/) {
    throwExceptionOnDuplicateMapKeys_ =
        config.throwExceptionOnDuplicateMapKeys();
  }

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
  // A parsed entry. 'value' is empty when the entry has no key/value
  // delimiter, in which case the whole entry is the key and the value is NULL.
  struct Entry {
    std::string_view key;
    std::optional<std::string_view> value;
  };

  static std::string_view toStringView(const arg_type<Varchar>& input) {
    return std::string_view(input.data(), input.size());
  }

  void callImpl(
      out_type<Map<Varchar, Varchar>>& out,
      std::string_view input,
      std::string_view entryDelimiter,
      std::string_view keyValueDelimiter) {
    // Entries are buffered before writing so that a repeated key can either
    // raise an error or overwrite the value of its first occurrence, which is
    // how Spark's ArrayBasedMapBuilder deduplicates under LAST_WIN: the key
    // keeps the position of its first occurrence and takes the last value.
    entries_.clear();
    keyToIndex_.clear();

    size_t pos = 0;
    auto nextEntryPos = input.find(entryDelimiter, pos);
    while (nextEntryPos != std::string::npos) {
      processEntry(
          std::string_view(input.data() + pos, nextEntryPos - pos),
          keyValueDelimiter);

      pos = nextEntryPos + 1;
      nextEntryPos = input.find(entryDelimiter, pos);
    }

    processEntry(
        std::string_view(input.data() + pos, input.size() - pos),
        keyValueDelimiter);

    out.reserve(entries_.size());
    for (const auto& entry : entries_) {
      if (entry.value.has_value()) {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter.setNoCopy(StringView(entry.key));
        valueWriter.setNoCopy(StringView(entry.value.value()));
      } else {
        out.add_null().setNoCopy(StringView(entry.key));
      }
    }
  }

  void processEntry(
      std::string_view entry,
      std::string_view keyValueDelimiter) {
    const auto delimiterPos = entry.find(keyValueDelimiter, 0);
    // Allows keyValue delimiter not found: the whole entry is the key and the
    // value is NULL.
    std::string_view key = entry;
    std::optional<std::string_view> value;
    if (delimiterPos != std::string::npos) {
      key = std::string_view(entry.data(), delimiterPos);
      value = std::string_view(
          entry.data() + delimiterPos + 1, entry.size() - delimiterPos - 1);
    }

    auto [it, inserted] = keyToIndex_.try_emplace(key, entries_.size());
    if (inserted) {
      entries_.push_back({key, value});
      return;
    }
    VELOX_USER_CHECK(
        !throwExceptionOnDuplicateMapKeys_,
        "Duplicate keys are not allowed: '{}'.",
        key);
    entries_[it->second].value = value;
  }

  bool throwExceptionOnDuplicateMapKeys_{false};

  // Scratch buffers reused across rows to avoid per-row allocation. They hold
  // views into the current row's input and are cleared at the start of each
  // call.
  std::vector<Entry> entries_;
  folly::F14FastMap<std::string_view, size_t> keyToIndex_;
};

} // namespace facebook::velox::functions::sparksql
