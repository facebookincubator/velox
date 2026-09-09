/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::nimble::benchmarks {

/// Owns immutable string storage and stable views used by native benchmarks.
struct StringBenchmarkCorpus {
  StringBenchmarkCorpus() = default;
  StringBenchmarkCorpus(const StringBenchmarkCorpus&) = delete;
  StringBenchmarkCorpus& operator=(const StringBenchmarkCorpus&) = delete;
  StringBenchmarkCorpus(StringBenchmarkCorpus&&) noexcept = default;
  StringBenchmarkCorpus& operator=(StringBenchmarkCorpus&&) noexcept = default;

  /// Owns the bytes referenced by values. Do not mutate after finalization.
  std::vector<std::string> storage;
  /// Provides sorted views into storage.
  std::vector<std::string_view> values;
  /// Sums logical string bytes without encoding metadata.
  uint64_t rawBytes{0};
  /// Records the largest logical string in bytes.
  size_t maxValueBytes{0};
};

/// Sorts owned values and builds stable views after storage is finalized.
inline StringBenchmarkCorpus finalizeStringBenchmarkCorpus(
    std::vector<std::string> storage) {
  std::sort(storage.begin(), storage.end());

  StringBenchmarkCorpus corpus;
  corpus.storage = std::move(storage);
  corpus.values.reserve(corpus.storage.size());
  for (const auto& value : corpus.storage) {
    corpus.values.push_back(value);
    corpus.rawBytes += value.size();
    corpus.maxValueBytes = std::max(corpus.maxValueBytes, value.size());
  }
  return corpus;
}

} // namespace facebook::nimble::benchmarks
