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

#include <charconv>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"

namespace facebook::nimble::detail::subintsplit {

inline constexpr std::string_view kSplitModeConfigKey = "subintsplit.mode";
inline constexpr std::string_view kSplitBoundariesConfigKey =
    "subintsplit.boundaries";
inline constexpr std::string_view kSplitModeRecompute = "recompute";
inline constexpr std::string_view kSplitModePreserve = "preserve";

inline std::string serializeSplitBoundaries(
    std::span<const SegmentPlan> segments) {
  std::string result;
  for (size_t i = 0; i < segments.size(); ++i) {
    if (i > 0) {
      result.push_back(';');
    }
    result += std::to_string(segments[i].bitStart);
    result.push_back('-');
    result += std::to_string(segments[i].bitEnd);
  }
  return result;
}

inline std::optional<std::vector<SegmentPlan>> parseSplitBoundaries(
    std::string_view value,
    int kBits) {
  if (value.empty() || kBits <= 0) {
    return std::nullopt;
  }

  std::vector<SegmentPlan> segments;
  segments.reserve(8);

  int expectedStart = 0;
  size_t cursor = 0;
  while (cursor < value.size()) {
    const auto dashPos = value.find('-', cursor);
    if (dashPos == std::string_view::npos) {
      return std::nullopt;
    }

    const auto semiPos = value.find(';', dashPos + 1);
    const std::string_view startText = value.substr(cursor, dashPos - cursor);
    const std::string_view endText = value.substr(
        dashPos + 1,
        semiPos == std::string_view::npos ? value.size() - (dashPos + 1)
                                          : semiPos - (dashPos + 1));

    int bitStart = 0;
    int bitEnd = 0;
    const auto startResult = std::from_chars(
        startText.data(), startText.data() + startText.size(), bitStart);
    const auto endResult = std::from_chars(
        endText.data(), endText.data() + endText.size(), bitEnd);
    if (startResult.ec != std::errc{} ||
        startResult.ptr != startText.data() + startText.size() ||
        endResult.ec != std::errc{} ||
        endResult.ptr != endText.data() + endText.size()) {
      return std::nullopt;
    }

    if (bitStart != expectedStart || bitStart < 0 || bitEnd < bitStart ||
        bitEnd >= kBits) {
      return std::nullopt;
    }

    segments.push_back({.bitStart = bitStart, .bitEnd = bitEnd});
    expectedStart = bitEnd + 1;
    if (semiPos == std::string_view::npos) {
      break;
    }
    cursor = semiPos + 1;
  }

  if (segments.empty() || expectedStart != kBits) {
    return std::nullopt;
  }

  return segments;
}

inline std::unordered_map<std::string, std::string> makePreserveSplitConfig(
    std::span<const SegmentPlan> segments) {
  return {
      {std::string(kSplitModeConfigKey), std::string(kSplitModePreserve)},
      {std::string(kSplitBoundariesConfigKey),
       serializeSplitBoundaries(segments)},
  };
}

} // namespace facebook::nimble::detail::subintsplit
