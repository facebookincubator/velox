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

#include "velox/experimental/cudf/CudfConfig.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <folly/Conv.h>
#include <gtest/gtest.h>

#include <limits>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, BatchConcatThresholdDefaults) {
  CudfConfig config;
  EXPECT_EQ(config.batchSizeMinThreshold, 100'000);
  EXPECT_FALSE(config.batchSizeMinBytes);
}

TEST(ConfigTest, CudfConfig) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kCudfBatchSizeMinThreshold, "123456"},
      {CudfConfig::kCudfBatchSizeMinBytes, "2147483648"}};

  CudfConfig config;
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_EQ(config.batchSizeMinThreshold, 123'456);
  ASSERT_EQ(config.batchSizeMinBytes.value(), 2'147'483'648);
}

TEST(ConfigTest, RejectsZeroBatchSizeMinBytes) {
  CudfConfig config;
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfBatchSizeMinBytes, "0"}};

  VELOX_ASSERT_USER_THROW(
      config.initialize(std::move(options)),
      "cuDF BatchConcat minimum byte target must be positive");
}

TEST(ConfigTest, ParsesMaximumBatchSizeMinBytes) {
  CudfConfig config;
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfBatchSizeMinBytes, "18446744073709551615"}};

  config.initialize(std::move(options));

  EXPECT_EQ(
      config.batchSizeMinBytes.value(), std::numeric_limits<uint64_t>::max());
}

TEST(ConfigTest, RejectsBatchSizeMinBytesOverflow) {
  CudfConfig config;
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfBatchSizeMinBytes, "18446744073709551616"}};

  EXPECT_THROW(config.initialize(std::move(options)), folly::ConversionError);
}
} // namespace facebook::velox::cudf_velox::test
