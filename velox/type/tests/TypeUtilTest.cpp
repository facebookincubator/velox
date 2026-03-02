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

#include "velox/type/TypeUtil.h"

#include <gtest/gtest.h>

namespace facebook::velox::type {
namespace {

TEST(TypeUtilTest, concatRowTypes) {
  auto keyType = ROW({"k0", "k1"}, {BIGINT(), INTEGER()});
  auto valueType = ROW({"v0", "v1"}, {VARBINARY(), ARRAY(BIGINT())});
  auto type = concatRowTypes({keyType, valueType});
  auto expected =
      ROW({"k0", "k1", "v0", "v1"},
          {BIGINT(), INTEGER(), VARBINARY(), ARRAY(BIGINT())});
  EXPECT_EQ(type->toString(), expected->toString());
}

TEST(TypeUtilTest, tryGetHomogeneousRowChild) {
  {
    auto child = tryGetHomogeneousRowChild(ROW({"c1", "c2", "c3"}, BIGINT()));
    ASSERT_NE(child, nullptr);
    ASSERT_EQ(*child, *BIGINT());
  }

  ASSERT_EQ(tryGetHomogeneousRowChild(INTEGER()), nullptr);
  ASSERT_EQ(tryGetHomogeneousRowChild(ROW({})), nullptr);
  ASSERT_EQ(tryGetHomogeneousRowChild(ROW({BIGINT(), VARCHAR()})), nullptr);
  ASSERT_EQ(
      tryGetHomogeneousRowChild(ROW({"c1", "c2"}, {BIGINT(), VARCHAR()})),
      nullptr);
}

} // namespace
} // namespace facebook::velox::type
