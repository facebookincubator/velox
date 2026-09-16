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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "velox/dwio/nimble/encodings/benchmarks/StringBenchmarkCorpus.h"

namespace facebook::nimble::benchmarks {

/// Default seed shared by the native benchmark and executable runner.
inline constexpr uint64_t kPrefixBenchmarkDefaultSeed = 0xC0FFEE;

/// Default row count for the representative sorted string corpus.
inline constexpr uint32_t kPrefixBenchmarkDefaultRowCount = 4096;

namespace detail {

inline std::string prefixBenchmarkEdgeValue(uint32_t row, uint64_t seed) {
  switch (row) {
    case 0:
    case 1:
      return {};
    case 2:
      return std::string{"\0embedded-prefix", 16};
    case 3:
    case 4:
      return "a";
    case 5:
      return "b/short";
    case 6:
      return "c/" + std::string(8 * 1024, 'c');
    default:
      return fmt::format("edge/{:02}/seed={:016x}", row, seed);
  }
}

inline std::string prefixBenchmarkPathValue(uint32_t row, uint64_t seed) {
  constexpr uint32_t kRestartInterval = 16;
  const uint32_t relativeRow = row - kRestartInterval;
  const uint32_t group = relativeRow / kRestartInterval;
  const uint32_t groupRow = relativeRow % kRestartInterval;
  std::string value = fmt::format(
      "warehouse/tenant={:06}/table=customer_events/schema=v7/"
      "date=2026-08-{:02}/hour={:02}/partition={:08}/",
      (group + seed) % 100'000,
      1 + group % 28,
      group % 24,
      group);

  static constexpr std::array<std::string_view, 15> kSuffixes{
      "",
      "column=account_id",
      "column=account_id",
      "column=account_name",
      "column=country_code",
      "column=event_name",
      "column=event_timestamp",
      "column=ingestion_timestamp",
      "column=metadata_json",
      "column=partition_date",
      "column=session_id",
      "column=source_service",
      "column=tenant_id",
      "column=user_agent",
      "column=user_id",
  };
  if (groupRow < kSuffixes.size()) {
    value.append(kSuffixes[groupRow]);
    if (groupRow == 3) {
      value.append("\0binary", 7);
    }
    return value;
  }

  value.append("payload=z/");
  value.append(1024, static_cast<char>('a' + ((group ^ seed) % 26)));
  return value;
}

} // namespace detail

/// Builds the deterministic SortedPathPrefixMixedLengths string corpus.
inline StringBenchmarkCorpus makePrefixBenchmarkCorpus(
    uint32_t rowCount = kPrefixBenchmarkDefaultRowCount,
    uint64_t seed = kPrefixBenchmarkDefaultSeed) {
  std::vector<std::string> storage;
  storage.reserve(rowCount);
  for (uint32_t row = 0; row < rowCount; ++row) {
    storage.push_back(
        row < 16 ? detail::prefixBenchmarkEdgeValue(row, seed)
                 : detail::prefixBenchmarkPathValue(row, seed));
  }
  return finalizeStringBenchmarkCorpus(std::move(storage));
}

} // namespace facebook::nimble::benchmarks
