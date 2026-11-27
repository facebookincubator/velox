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

#include <string>
#include <vector>

#include "velox/dwio/nimble/common/HybridFlatMap.h"

namespace facebook::nimble::test {

/// Returns a grouping of `keyKind` keys whose explicit group `i` has group ID
/// `i` and holds `explicitGroups[i]`, followed by the keyless Default group,
/// which records its keys as `defaultGroupKeyPresenceMode` says.
HybridFlatMap makeHybridFlatMap(
    ScalarKind keyKind,
    std::vector<std::vector<std::string>> explicitGroups,
    HybridFlatMap::KeyPresenceMode defaultGroupKeyPresenceMode);

} // namespace facebook::nimble::test
