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
#include "velox/parse/SqlExpressionsParser.h"

namespace facebook::velox::parse {

namespace {
const auto& windowTypeNames() {
  static const folly::F14FastMap<WindowType, std::string_view> kNames = {
      {WindowType::kRows, "ROWS"},
      {WindowType::kRange, "RANGE"},
  };
  return kNames;
}

const auto& boundTypeNames() {
  static const folly::F14FastMap<BoundType, std::string_view> kNames = {
      {BoundType::kCurrentRow, "CURRENT ROW"},
      {BoundType::kUnboundedPreceding, "UNBOUNDED PRECEDING"},
      {BoundType::kUnboundedFollowing, "UNBOUNDED FOLLOWING"},
      {BoundType::kPreceding, "PRECEDING"},
      {BoundType::kFollowing, "FOLLOWING"},
  };
  return kNames;
}
} // namespace

VELOX_DEFINE_ENUM_NAME(WindowType, windowTypeNames);
VELOX_DEFINE_ENUM_NAME(BoundType, boundTypeNames);

std::string OrderByClause::toString() const {
  return fmt::format(
      "{} {} NULLS {}",
      expr->toString(),
      (ascending ? "ASC" : "DESC"),
      (nullsFirst ? "FIRST" : "LAST"));
}

} // namespace facebook::velox::parse
