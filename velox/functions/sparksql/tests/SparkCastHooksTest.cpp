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

#include "velox/functions/sparksql/specialforms/SparkCastHooks.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::sparksql::test {
namespace {
TEST(SparkCastHooksTest, matchRowFieldsByName) {
  // Test that matchRowFieldsByName() returns the correct value based on
  // the constructor param and the value in the QueryConfig.

  auto testMatchRowFieldsByName = [](bool queryConfigValue,
                                     bool castHooksValue,
                                     bool expected) {
    core::QueryConfig queryConfig(
        {{core::QueryConfig::kCastMatchStructByName,
          folly::to<std::string>(queryConfigValue)}});

    EXPECT_EQ(
        expected,
        SparkCastHooks(queryConfig, false /* allowOverflow */, castHooksValue)
            .matchRowFieldsByName());
  };

  // If either is set to true, matchRowFieldsByName should return true.
  testMatchRowFieldsByName(false, false, false);
  testMatchRowFieldsByName(true, false, true);
  testMatchRowFieldsByName(false, true, true);
  testMatchRowFieldsByName(true, true, true);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
