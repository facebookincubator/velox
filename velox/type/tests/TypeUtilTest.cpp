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

TEST(TypeUtilTest, ConcatRowTypes) {
  auto keyType = velox::ROW({"k0", "k1"}, {velox::BIGINT(), velox::INTEGER()});
  auto valueType = velox::ROW(
      {"v0", "v1"}, {velox::VARBINARY(), velox::ARRAY(velox::BIGINT())});
  auto type = concatRowTypes({keyType, valueType});
  auto expected = velox::ROW(
      {"k0", "k1", "v0", "v1"},
      {velox::BIGINT(),
       velox::INTEGER(),
       velox::VARBINARY(),
       velox::ARRAY(velox::BIGINT())});
  EXPECT_EQ(type->toString(), expected->toString());
}

TEST(TypeUtilTest, tryGetHomogeneousRowChild_NonRow) {
  auto t = velox::INTEGER();
  auto child = tryGetHomogeneousRowChild(t);
  ASSERT_EQ(child, nullptr);
}

TEST(TypeUtilTest, tryGetHomogeneousRowChild_EmptyRow) {
  auto row = velox::ROW({}, {});
  auto child = tryGetHomogeneousRowChild(row);
  ASSERT_EQ(child, nullptr);
}

TEST(TypeUtilTest, tryGetHomogeneousRowChild_Homogeneous) {
  auto row = velox::ROW(
      {"c1", "c2", "c3"}, {velox::BIGINT(), velox::BIGINT(), velox::BIGINT()});
  auto child = tryGetHomogeneousRowChild(row);
  ASSERT_NE(child, nullptr);
  ASSERT_TRUE(child->equals(*velox::BIGINT()));
}

TEST(TypeUtilTest, tryGetHomogeneousRowChild_Heterogeneous) {
  auto row = velox::ROW({"c1", "c2"}, {velox::BIGINT(), velox::VARCHAR()});
  auto child = tryGetHomogeneousRowChild(row);
  ASSERT_EQ(child, nullptr);
}

} // namespace
} // namespace facebook::velox::type
