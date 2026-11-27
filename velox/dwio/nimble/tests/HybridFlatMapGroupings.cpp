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

#include "velox/dwio/nimble/tests/HybridFlatMapGroupings.h"

#include <cstdint>
#include <utility>

namespace facebook::nimble::test {

HybridFlatMap makeHybridFlatMap(
    ScalarKind keyKind,
    std::vector<std::vector<std::string>> explicitGroups,
    HybridFlatMap::KeyPresenceMode defaultGroupKeyPresenceMode) {
  std::vector<HybridFlatMap::Group> groups;
  groups.reserve(explicitGroups.size() + 1);
  for (uint32_t groupId = 0; groupId < explicitGroups.size(); ++groupId) {
    groups.push_back(
        HybridFlatMap::Group{
            .groupId = groupId,
            .groupKeys = std::move(explicitGroups[groupId]),
        });
  }
  groups.push_back(
      HybridFlatMap::Group{
          .groupId = HybridFlatMap::kDefaultGroupId,
          .groupKeys = {},
      });
  return HybridFlatMap{keyKind, std::move(groups), defaultGroupKeyPresenceMode};
}

} // namespace facebook::nimble::test
