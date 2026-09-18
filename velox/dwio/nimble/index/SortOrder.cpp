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

#include "velox/dwio/nimble/index/SortOrder.h"

namespace facebook::nimble {

folly::dynamic SortOrder::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["ascending"] = ascending;
  return obj;
}

SortOrder SortOrder::deserialize(const folly::dynamic& obj) {
  return SortOrder{.ascending = obj["ascending"].asBool()};
}

velox::core::SortOrder SortOrder::toVeloxSortOrder() const {
  return velox::core::SortOrder{ascending, /*nullsFirst=*/false};
}

} // namespace facebook::nimble
