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
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::memory {

/// Key under which a MemoryTier is stored on QueryCtx or OperatorCtx via
/// setRegistry / registry. Operators use this key to look up the effective
/// memory-tier hierarchy when choosing a relocation target.
inline constexpr std::string_view kMemoryTierKey{"memoryTier"};

/// Immutable description of a memory-tier hierarchy. Each level lists the
/// resource tags considered equivalent at that tier, ordered from fastest
/// (level 0) to slowest, and within a level from most to least preferred.
/// Tags name custom memory resources (see CustomMemoryResource) or the default
/// pool. For a GPU setup, level 0 might be {"rmm"}, level 1 {"pinnedHost"}, and
/// level 2 {"cxl", "dram"}.
///
/// Tags are not validated against any resource registry; a tag with no
/// registered resource never resolves to a pool during relocation lookup.
class MemoryTier {
 public:
  /// Creates a hierarchy where tiers[level] lists the tags at that level in
  /// preference order. Throws if 'tiers' is empty, any level is empty, any tag
  /// is empty, or a tag appears more than once across the hierarchy.
  explicit MemoryTier(std::vector<std::vector<std::string>> tiers);

  /// Returns the number of tier levels.
  size_t numTiers() const {
    return tiers_.size();
  }

  /// Returns the tags at 'level' in preference order. Throws if 'level' is not
  /// less than numTiers().
  const std::vector<std::string>& tierAt(size_t level) const;

  /// Returns the level containing 'tag', or std::nullopt if 'tag' is not part
  /// of this hierarchy.
  std::optional<size_t> levelOf(std::string_view tag) const;

  /// Returns a human-readable representation, e.g.
  /// "MemoryTier[[rmm], [pinnedHost], [cxl, dram]]".
  std::string toString() const;

 private:
  const std::vector<std::vector<std::string>> tiers_;
};

} // namespace facebook::velox::memory
