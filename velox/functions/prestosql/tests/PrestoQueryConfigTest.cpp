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
#include "velox/functions/prestosql/PrestoQueryConfig.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::prestosql::test {
namespace {

// Verifies the write/read loop: values set via qualify(key) in the underlying
// QueryConfig map are returned by the corresponding typed accessor.
TEST(PrestoQueryConfigTest, roundTrip) {
  {
    core::QueryConfig queryConfig({
        {PrestoQueryConfig::qualify(PrestoQueryConfig::kArrayAggIgnoreNulls),
         "true"},
    });
    EXPECT_TRUE(PrestoQueryConfig{queryConfig}.arrayAggIgnoreNulls());
  }

  {
    core::QueryConfig queryConfig({
        {PrestoQueryConfig::qualify(PrestoQueryConfig::kArrayAggIgnoreNulls),
         "false"},
    });
    EXPECT_FALSE(PrestoQueryConfig{queryConfig}.arrayAggIgnoreNulls());
  }
}

} // namespace
} // namespace facebook::velox::functions::prestosql::test
