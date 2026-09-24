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

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "velox/dwio/nimble/encodings/subintsplit/BitSection.h"

// Encoding-selection config that pins a SubIntSplit encoding to a previously
// chosen split, so a captured encoding layout replays byte-identically instead
// of re-running the planner over fresh data.
//
// Boundaries serialize as "start-end;start-end;…", LSB-first, covering every
// bit of the physical type with no gaps.

namespace facebook::nimble::subintsplit {

inline constexpr std::string_view kSplitModeConfigKey = "subintsplit.mode";
inline constexpr std::string_view kSplitBoundariesConfigKey =
    "subintsplit.boundaries";
inline constexpr std::string_view kSplitModeRecompute = "recompute";
inline constexpr std::string_view kSplitModePreserve = "preserve";

std::string serializeSplitBoundaries(std::span<const SectionPlan> sections);

/// Parses boundaries covering exactly `numBits` bits. Returns nullopt when the
/// text is malformed, or when the ranges leave a gap, overlap, or fail to cover
/// the full width.
std::optional<std::vector<SectionPlan>> parseSplitBoundaries(
    std::string_view value,
    int numBits);

/// Config that makes SubIntSplit reuse `sections` verbatim.
std::unordered_map<std::string, std::string> makePreserveSplitConfig(
    std::span<const SectionPlan> sections);

} // namespace facebook::nimble::subintsplit
