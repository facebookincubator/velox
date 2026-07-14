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

#include "velox/common/memory/MemoryTier.h"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <folly/container/F14Set.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::memory {

MemoryTier::MemoryTier(std::vector<std::vector<std::string>> tiers)
    : tiers_{std::move(tiers)} {
  VELOX_CHECK(!tiers_.empty(), "MemoryTier requires at least one tier");
  folly::F14FastSet<std::string_view> seen;
  for (const auto& level : tiers_) {
    VELOX_CHECK(!level.empty(), "MemoryTier level must have at least one tag");
    for (const auto& tag : level) {
      VELOX_CHECK(!tag.empty(), "MemoryTier tag must not be empty");
      VELOX_CHECK(
          seen.insert(tag).second, "Duplicate tag in MemoryTier: {}", tag);
    }
  }
}

const std::vector<std::string>& MemoryTier::tierAt(size_t level) const {
  VELOX_CHECK_LT(level, tiers_.size(), "MemoryTier level out of range");
  return tiers_[level];
}

std::optional<size_t> MemoryTier::levelOf(std::string_view tag) const {
  for (size_t level = 0; level < tiers_.size(); ++level) {
    for (const auto& candidate : tiers_[level]) {
      if (candidate == tag) {
        return level;
      }
    }
  }
  return std::nullopt;
}

std::string MemoryTier::toString() const {
  std::vector<std::string> levels;
  levels.reserve(tiers_.size());
  for (const auto& level : tiers_) {
    levels.push_back(fmt::format("[{}]", fmt::join(level, ", ")));
  }
  return fmt::format("MemoryTier[{}]", fmt::join(levels, ", "));
}

} // namespace facebook::velox::memory
