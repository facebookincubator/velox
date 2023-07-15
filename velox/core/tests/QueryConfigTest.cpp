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
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::core::test {

TEST(TestQueryConfig, emptyConfig) {
  std::unordered_map<std::string, std::string> configData;
  auto queryCtx = std::make_shared<QueryCtx>(nullptr, std::move(configData));
  const QueryConfig& config = queryCtx->queryConfig();

  ASSERT_FALSE(config.codegenEnabled());
  ASSERT_EQ(config.codegenConfigurationFilePath(), "");
  ASSERT_FALSE(config.isCastToIntByTruncate());
}

TEST(TestQueryConfig, setConfig) {
  std::string path = "/tmp/CodeGenConfig";
  std::unordered_map<std::string, std::string> configData(
      {{QueryConfig::kCodegenEnabled, "true"},
       {QueryConfig::kCodegenConfigurationFilePath, path}});
  auto queryCtx = std::make_shared<QueryCtx>(nullptr, std::move(configData));
  const QueryConfig& config = queryCtx->queryConfig();

  ASSERT_TRUE(config.codegenEnabled());
  ASSERT_EQ(config.codegenConfigurationFilePath(), path);
  ASSERT_FALSE(config.isCastToIntByTruncate());
}

TEST(TestQueryConfig, memConfig) {
  const std::string tz = "timezone1";
  const std::unordered_map<std::string, std::string> configData(
      {{QueryConfig::kCodegenEnabled, "true"},
       {QueryConfig::kSessionTimezone, tz}});

  {
    MemConfig cfg{configData};
    MemConfig cfg2{};
    auto configDataCopy = configData;
    MemConfig cfg3{std::move(configDataCopy)};
    ASSERT_TRUE(cfg.Config::get<bool>(QueryConfig::kCodegenEnabled));
    ASSERT_TRUE(cfg3.Config::get<bool>(QueryConfig::kCodegenEnabled));
    ASSERT_EQ(
        tz,
        cfg.Config::get<std::string>(QueryConfig::kSessionTimezone).value());
    ASSERT_FALSE(cfg.Config::get<std::string>("missing-entry").has_value());
    ASSERT_EQ(configData, cfg.values());
    ASSERT_EQ(configData, cfg.valuesCopy());
  }

  {
    MemConfigMutable cfg{configData};
    MemConfigMutable cfg2{};
    auto configDataCopy = configData;
    MemConfigMutable cfg3{std::move(configDataCopy)};
    ASSERT_TRUE(cfg.Config::get<bool>(QueryConfig::kCodegenEnabled).value());
    ASSERT_TRUE(cfg3.Config::get<bool>(QueryConfig::kCodegenEnabled).value());
    ASSERT_EQ(
        tz,
        cfg.Config::get<std::string>(QueryConfig::kSessionTimezone).value());
    ASSERT_FALSE(cfg.Config::get<std::string>("missing-entry").has_value());
    ASSERT_NO_THROW(cfg.setValue(QueryConfig::kCodegenEnabled, "false"));
    ASSERT_FALSE(cfg.Config::get<bool>(QueryConfig::kCodegenEnabled).value());
    const std::string tz2 = "timezone2";
    ASSERT_NO_THROW(cfg.setValue(QueryConfig::kSessionTimezone, tz2));
    ASSERT_EQ(
        tz2,
        cfg.Config::get<std::string>(QueryConfig::kSessionTimezone).value());
    ASSERT_THROW(cfg.values(), VeloxException);
    ASSERT_EQ(configData, cfg3.valuesCopy());
  }
}

TEST(TestQueryConfig, taskWriterCountConfig) {
  struct {
    std::optional<int> numWriterCounter;
    std::optional<int> numPartitionedWriterCounter;
    int expectedWriterCounter;
    int expectedPartitionedWriterCounter;

    std::string debugString() const {
      return fmt::format(
          "numWriterCounter[{}] numPartitionedWriterCounter[{}] expectedWriterCounter[{}] expectedPartitionedWriterCounter[{}]",
          numWriterCounter.value_or(0),
          numPartitionedWriterCounter.value_or(0),
          expectedWriterCounter,
          expectedPartitionedWriterCounter);
    }
  } testSettings[] = {
      {std::nullopt, std::nullopt, 1, 1},
      {std::nullopt, 4, 1, 4},
      {std::nullopt, 6, 1, 6},
      {2, 4, 2, 4},
      {4, 2, 4, 2},
      {4, 6, 4, 6},
      {6, 5, 6, 5},
      {6, 4, 6, 4},
      {6, std::nullopt, 6, 6}};
  for (const auto& testConfig : testSettings) {
    SCOPED_TRACE(testConfig.debugString());
    std::unordered_map<std::string, std::string> configData;
    if (testConfig.numWriterCounter.has_value()) {
      configData.emplace(
          QueryConfig::kTaskWriterCount,
          std::to_string(testConfig.numWriterCounter.value()));
    }
    if (testConfig.numPartitionedWriterCounter.has_value()) {
      configData.emplace(
          QueryConfig::kTaskPartitionedWriterCount,
          std::to_string(testConfig.numPartitionedWriterCounter.value()));
    }
    auto queryCtx = std::make_shared<QueryCtx>(nullptr, std::move(configData));
    const QueryConfig& config = queryCtx->queryConfig();
    ASSERT_EQ(config.taskWriterCount(), testConfig.expectedWriterCounter);
    ASSERT_EQ(
        config.taskPartitionedWriterCount(),
        testConfig.expectedPartitionedWriterCounter);
  }
}

} // namespace facebook::velox::core::test
