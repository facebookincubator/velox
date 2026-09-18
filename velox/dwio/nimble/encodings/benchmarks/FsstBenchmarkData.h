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
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "velox/dwio/nimble/encodings/benchmarks/StringBenchmarkCorpus.h"

namespace facebook::nimble::benchmarks {

/// Default seed shared by the native benchmark and executable runner.
inline constexpr uint64_t kFsstBenchmarkDefaultSeed = 0xC0FFEE;

/// Default row count for the representative structured string corpus.
inline constexpr uint32_t kFsstBenchmarkDefaultRowCount = 4096;

namespace detail {

inline std::string fsstBenchmarkEdgeValue(uint32_t row, uint64_t seed) {
  switch (row) {
    case 0:
    case 1:
      return {};
    case 2:
    case 3:
      return "a";
    case 4: {
      std::string value{"edge/embedded"};
      value.append("\0nul", 4);
      return value;
    }
    case 5: {
      std::string value{"edge/rare-byte/"};
      value.push_back(static_cast<char>(0xff));
      return value;
    }
    case 6: {
      std::string value{"edge/rare-tail/"};
      value.push_back(static_cast<char>(0xff));
      return value;
    }
    case 7: {
      std::string value{"edge/long/"};
      value.append(8 * 1024, 'x');
      return value;
    }
    case 8: {
      std::string value{"edge/chunked/"};
      value.append(1025, 'q');
      value.append("\0tail", 5);
      return value;
    }
    default:
      return fmt::format("edge/{:02}/seed={:016x}", row, seed);
  }
}

inline std::string fsstBenchmarkUrlValue(uint32_t row, uint64_t seed) {
  return fmt::format(
      "https://api.example.test/v3/tenant/{:06}/users/{:08}/events/"
      "{:08}?locale=en-US&source=mobile&session={:016x}&page={}",
      (row + seed) % 100'000,
      row * 17,
      row * 31,
      seed ^ (static_cast<uint64_t>(row) * 0x9E3779B97F4A7C15ULL),
      row % 97);
}

inline std::string fsstBenchmarkLogValue(uint32_t row, uint64_t seed) {
  static constexpr std::array<std::string_view, 5> kLevels{
      "DEBUG", "INFO", "NOTICE", "WARN", "ERROR"};
  static constexpr std::array<std::string_view, 4> kComponents{
      "cache", "planner", "storage", "transport"};
  return fmt::format(
      R"({{"timestamp":"2026-08-09T21:{:02}:{:02}Z","level":"{}","component":"{}","tenant":{},"request":{},"message":"completed structured benchmark request","seed":"{:016x}"}})",
      row % 60,
      (row * 13) % 60,
      kLevels[row % kLevels.size()],
      kComponents[(row / 3) % kComponents.size()],
      (row + seed) % 10'000,
      row,
      seed);
}

inline std::string fsstBenchmarkUuidValue(uint32_t row, uint64_t seed) {
  const uint64_t mixed =
      seed ^ (static_cast<uint64_t>(row) * 0xD6E8FEB86659FD93ULL);
  return fmt::format(
      "event/{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}/trace/{:016x}",
      static_cast<uint32_t>(mixed),
      static_cast<uint32_t>(mixed >> 32) & 0xffff,
      row & 0xfff,
      0x8000 | (row & 0x3fff),
      mixed & 0xffffffffffffULL,
      mixed ^ 0xA5A5A5A5A5A5A5A5ULL);
}

inline std::string fsstBenchmarkLongValue(uint32_t row, uint64_t seed) {
  std::string value = fmt::format(
      "payload/tenant={:06}/partition={:08}/codec=fsst/data=",
      (row + seed) % 100'000,
      row / 8);
  value.append(
      1024 + (row % 4) * 128, static_cast<char>('a' + ((row ^ seed) % 26)));
  value.append("\0binary-tail/", 13);
  value.push_back(static_cast<char>(0xff));
  return value;
}

inline std::string fsstBenchmarkStructuredValue(uint32_t row, uint64_t seed) {
  switch (row % 8) {
    case 0:
    case 1:
    case 2:
    case 3:
      return fsstBenchmarkUrlValue(row, seed);
    case 4:
    case 5:
      return fsstBenchmarkLogValue(row, seed);
    case 6:
      return fsstBenchmarkUuidValue(row, seed);
    case 7:
      return fsstBenchmarkLongValue(row, seed);
  }
  return {};
}

} // namespace detail

/// Builds the deterministic SortedStructuredTextMixedLengths FSST corpus.
inline StringBenchmarkCorpus makeFsstBenchmarkCorpus(
    uint32_t rowCount = kFsstBenchmarkDefaultRowCount,
    uint64_t seed = kFsstBenchmarkDefaultSeed) {
  std::vector<std::string> storage;
  storage.reserve(rowCount);
  for (uint32_t row = 0; row < rowCount; ++row) {
    storage.push_back(
        row < 16 ? detail::fsstBenchmarkEdgeValue(row, seed)
                 : detail::fsstBenchmarkStructuredValue(row, seed));
  }
  return finalizeStringBenchmarkCorpus(std::move(storage));
}

} // namespace facebook::nimble::benchmarks
