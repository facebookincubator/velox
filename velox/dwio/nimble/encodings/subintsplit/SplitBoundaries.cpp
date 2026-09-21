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
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"

#include <charconv>

namespace facebook::nimble::subintsplit {
namespace {

// Parses a complete decimal integer; rejects trailing junk such as "7x".
std::optional<int> parseInt(std::string_view text) {
  int value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

} // namespace

std::string serializeSplitBoundaries(std::span<const SectionPlan> sections) {
  std::string result;
  for (const auto& section : sections) {
    if (!result.empty()) {
      result.push_back(';');
    }
    result += std::to_string(section.bitStart);
    result.push_back('-');
    result += std::to_string(section.bitEnd);
  }
  return result;
}

std::optional<std::vector<SectionPlan>> parseSplitBoundaries(
    std::string_view value,
    int numBits) {
  if (value.empty() || numBits <= 0) {
    return std::nullopt;
  }

  std::vector<SectionPlan> sections;
  sections.reserve(8);

  int expectedStart = 0;
  size_t cursor = 0;
  while (cursor < value.size()) {
    const auto dash = value.find('-', cursor);
    if (dash == std::string_view::npos) {
      return std::nullopt;
    }
    const auto semicolon = value.find(';', dash + 1);
    const size_t endLength = semicolon == std::string_view::npos
        ? value.size() - (dash + 1)
        : semicolon - (dash + 1);

    const auto bitStart = parseInt(value.substr(cursor, dash - cursor));
    const auto bitEnd = parseInt(value.substr(dash + 1, endLength));
    if (!bitStart.has_value() || !bitEnd.has_value()) {
      return std::nullopt;
    }

    // Sections must tile the width in order, leaving no gap and no overlap.
    if (*bitStart != expectedStart || *bitStart < 0 || *bitEnd < *bitStart ||
        *bitEnd >= numBits) {
      return std::nullopt;
    }

    sections.push_back({.bitStart = *bitStart, .bitEnd = *bitEnd});
    expectedStart = *bitEnd + 1;
    if (semicolon == std::string_view::npos) {
      break;
    }
    cursor = semicolon + 1;
  }

  if (sections.empty() || expectedStart != numBits) {
    return std::nullopt;
  }
  return sections;
}

std::unordered_map<std::string, std::string> makePreserveSplitConfig(
    std::span<const SectionPlan> sections) {
  return {
      {std::string(kSplitModeConfigKey), std::string(kSplitModePreserve)},
      {std::string(kSplitBoundariesConfigKey),
       serializeSplitBoundaries(sections)},
  };
}

} // namespace facebook::nimble::subintsplit
