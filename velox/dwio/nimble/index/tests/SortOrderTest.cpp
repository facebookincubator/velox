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

#include <gtest/gtest.h>

#include "velox/dwio/nimble/index/SortOrder.h"

namespace facebook::nimble::index::test {

TEST(SortOrderTest, basicOperations) {
  SortOrder defaultOrder;
  EXPECT_TRUE(defaultOrder.ascending);

  SortOrder ascending{.ascending = true};
  SortOrder descending{.ascending = false};
  EXPECT_EQ(defaultOrder, ascending);
  EXPECT_NE(ascending, descending);

  EXPECT_EQ(ascending, SortOrder::deserialize(ascending.serialize()));
  EXPECT_EQ(descending, SortOrder::deserialize(descending.serialize()));
}

TEST(SortOrderTest, toVeloxSortOrder) {
  SortOrder ascending{.ascending = true};
  auto veloxAsc = ascending.toVeloxSortOrder();
  EXPECT_TRUE(veloxAsc.isAscending());
  EXPECT_FALSE(veloxAsc.isNullsFirst());

  SortOrder descending{.ascending = false};
  auto veloxDesc = descending.toVeloxSortOrder();
  EXPECT_FALSE(veloxDesc.isAscending());
  EXPECT_FALSE(veloxDesc.isNullsFirst());
}

} // namespace facebook::nimble::index::test
