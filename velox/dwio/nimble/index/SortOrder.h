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

#include <vector>

#include "folly/json/dynamic.h"
#include "velox/core/PlanNode.h"

namespace facebook::nimble {

/// Specifies the sort order for an index column.
/// Nulls are always sorted last.
struct SortOrder {
  bool ascending{true};

  bool operator==(const SortOrder& other) const = default;

  /// Serializes the sort order to a folly::dynamic object.
  folly::dynamic serialize() const;

  /// Deserializes a sort order from a folly::dynamic object.
  static SortOrder deserialize(const folly::dynamic& obj);

  /// Converts to velox::core::SortOrder.
  /// Nulls are always last.
  velox::core::SortOrder toVeloxSortOrder() const;
};

} // namespace facebook::nimble
