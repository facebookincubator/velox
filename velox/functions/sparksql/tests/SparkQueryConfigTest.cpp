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
#include "velox/functions/sparksql/SparkQueryConfig.h"

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/SparkConfigProvider.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

// Verifies the write/read loop: values set via qualify(key) in the underlying
// QueryConfig map are returned by the corresponding typed accessor.
TEST(SparkQueryConfigTest, roundTrip) {
  {
    core::QueryConfig queryConfig({
        {SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled), "true"},
    });
    EXPECT_TRUE(SparkQueryConfig{queryConfig}.ansiEnabled());
  }

  {
    core::QueryConfig queryConfig({
        {SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled), "false"},
    });
    EXPECT_FALSE(SparkQueryConfig{queryConfig}.ansiEnabled());
  }
}

// Verifies SparkConfigProvider::normalize rejects out-of-range partition_id.
TEST(SparkConfigProviderTest, normalizeRejectsNegativePartitionId) {
  SparkConfigProvider provider;
  VELOX_ASSERT_USER_THROW(
      provider.normalize(SparkQueryConfig::kPartitionId, "-1"),
      "Spark partition id must be non-negative");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
