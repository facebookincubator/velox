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
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"

#include <algorithm>

namespace facebook::nimble::subintsplit {

FrequencyCounter::Result FrequencyCounter::countDirect(
    const std::vector<uint64_t>& values) {
  if (directCounts_.empty()) {
    directCounts_.assign(size_t{1} << kDirectIndexBits, 0u);
  }
  touchedSlots_.clear();
  touchedSlots_.reserve(values.size());

  uint32_t dominantCount = 0;
  for (const uint64_t value : values) {
    const uint32_t slot = static_cast<uint32_t>(value);
    const uint32_t count = ++directCounts_[slot];
    if (count == 1) {
      touchedSlots_.push_back(slot);
    }
    dominantCount = std::max(dominantCount, count);
  }

  const size_t uniqueCount = touchedSlots_.size();
  for (const uint32_t slot : touchedSlots_) {
    directCounts_[slot] = 0;
  }

  return {
      .uniqueCount = uniqueCount,
      .dominantCount = dominantCount,
      .capped = uniqueCount > kUniqueCountCap};
}

FrequencyCounter::Result FrequencyCounter::countHashed(
    const std::vector<uint64_t>& values) {
  frequencies_.clear();
  frequencies_.reserve(std::min(values.size(), kUniqueCountCap));

  uint32_t dominantCount = 0;
  bool capped = false;
  for (const uint64_t value : values) {
    if (capped) {
      break;
    }
    auto [entry, inserted] = frequencies_.try_emplace(value, 0u);
    const uint32_t count = ++entry->second;
    dominantCount = std::max(dominantCount, count);
    if (inserted && frequencies_.size() > kUniqueCountCap) {
      capped = true;
    }
  }

  return {
      .uniqueCount = frequencies_.size(),
      .dominantCount = dominantCount,
      .capped = capped};
}

namespace {

// Fused because min/max and run counting are both a single comparison per
// value; splitting them would double the loop for no clarity gain.
void collectValueStats(
    const std::vector<uint64_t>& values,
    bool wantMinMax,
    bool wantRuns,
    SectionMetrics& out) {
  if (!wantMinMax && !wantRuns) {
    return;
  }

  uint64_t min = values[0];
  uint64_t max = values[0];
  size_t runCount = 1;
  uint64_t previous = values[0];

  for (size_t i = 1; i < values.size(); ++i) {
    const uint64_t value = values[i];
    min = std::min(min, value);
    max = std::max(max, value);
    runCount += (value != previous);
    previous = value;
  }

  if (wantMinMax) {
    out.min = min;
    out.max = max;
    out.range = max - min;
  }
  if (wantRuns) {
    out.runCount = runCount;
    out.avgRunLength =
        static_cast<double>(values.size()) / static_cast<double>(runCount);
  }
}

} // namespace

SectionMetrics MetricCollector::compute(
    const std::vector<uint64_t>& values,
    MetricFlags flags,
    int bitWidth) {
  SectionMetrics out;
  if (values.empty()) {
    return out;
  }

  collectValueStats(
      values,
      hasFlag(flags, MetricFlag::MinMax),
      hasFlag(flags, MetricFlag::RunStats),
      out);

  const bool wantUnique = hasFlag(flags, MetricFlag::UniqueCount);
  const bool wantDominant = hasFlag(flags, MetricFlag::DominantValue);
  if (!wantUnique && !wantDominant) {
    return out;
  }

  const auto counted = FrequencyCounter::fitsDirectIndex(bitWidth)
      ? frequencies_.countDirect(values)
      : frequencies_.countHashed(values);

  if (wantUnique) {
    out.uniqueCount = counted.capped ? (FrequencyCounter::kUniqueCountCap + 1)
                                     : counted.uniqueCount;
    out.uniqueCountCapped = counted.capped;
  }
  if (wantDominant) {
    out.dominantCount = counted.dominantCount;
    out.dominantCountCapped = counted.capped;
  }

  return out;
}

} // namespace facebook::nimble::subintsplit
