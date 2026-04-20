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

#include "velox/connectors/hive/FileConfig.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/config/Config.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;

TEST(FileConfigTest, defaultConfig) {
  FileConfig config(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  const auto emptySession = std::make_unique<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());

  EXPECT_FALSE(config.isOrcUseColumnNames(emptySession.get()));
  EXPECT_FALSE(config.isParquetUseColumnNames(emptySession.get()));
  EXPECT_FALSE(config.isFileColumnNamesReadAsLowerCase(emptySession.get()));
  EXPECT_FALSE(config.ignoreMissingFiles(emptySession.get()));
  EXPECT_EQ(config.maxCoalescedBytes(emptySession.get()), 128 << 20);
  EXPECT_EQ(config.maxCoalescedDistanceBytes(emptySession.get()), 512 << 10);
  EXPECT_EQ(config.prefetchRowGroups(), 1);
  EXPECT_EQ(config.parallelUnitLoadCount(emptySession.get()), 0);
  EXPECT_EQ(config.loadQuantum(emptySession.get()), 8 << 20);
  EXPECT_EQ(config.filePreloadThreshold(), 8UL << 20);
  EXPECT_EQ(config.readTimestampUnit(emptySession.get()), 3);
  EXPECT_TRUE(
      config.readTimestampPartitionValueAsLocalTime(emptySession.get()));
  EXPECT_FALSE(config.readStatsBasedFilterReorderDisabled(emptySession.get()));
  EXPECT_FALSE(config.preserveFlatMapsInMemory(emptySession.get()));
  EXPECT_FALSE(config.indexEnabled(emptySession.get()));
  EXPECT_FALSE(config.readerCollectColumnCpuMetrics(emptySession.get()));
  EXPECT_FALSE(config.fileMetadataCacheEnabled(emptySession.get()));
  EXPECT_FALSE(config.pinFileMetadata(emptySession.get()));
  EXPECT_EQ(config.orcFooterSpeculativeIoSize(emptySession.get()), 256UL << 10);
  EXPECT_EQ(
      config.parquetFooterSpeculativeIoSize(emptySession.get()), 256UL << 10);
  EXPECT_EQ(
      config.nimbleFooterSpeculativeIoSize(emptySession.get()), 8UL << 20);
}

TEST(FileConfigTest, overrideConfig) {
  std::unordered_map<std::string, std::string> configFromFile = {
      {FileConfig::kOrcUseColumnNames, "true"},
      {FileConfig::kParquetUseColumnNames, "true"},
      {FileConfig::kFileColumnNamesReadAsLowerCase, "true"},
      {FileConfig::kMaxCoalescedBytes, "100"},
      {FileConfig::kMaxCoalescedDistance, "100kB"},
      {FileConfig::kPrefetchRowGroups, "4"},
      {FileConfig::kLoadQuantum, std::to_string(4 << 20)},
      {FileConfig::kFilePreloadThreshold, std::to_string(16UL << 20)},
      {FileConfig::kReadStatsBasedFilterReorderDisabled, "true"},
      {FileConfig::kPreserveFlatMapsInMemory, "true"},
      {FileConfig::kIndexEnabled, "true"},
      {FileConfig::kReaderCollectColumnCpuMetrics, "true"},
      {FileConfig::kFileMetadataCacheEnabled, "true"},
      {FileConfig::kPinFileMetadata, "true"},
      {FileConfig::kOrcFooterSpeculativeIoSize, std::to_string(512UL << 10)},
      {FileConfig::kParquetFooterSpeculativeIoSize, std::to_string(1UL << 20)},
      {FileConfig::kNimbleFooterSpeculativeIoSize, std::to_string(4UL << 20)},
  };
  FileConfig config(
      std::make_shared<config::ConfigBase>(std::move(configFromFile)));
  const auto emptySession = std::make_unique<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());

  EXPECT_TRUE(config.isOrcUseColumnNames(emptySession.get()));
  EXPECT_TRUE(config.isParquetUseColumnNames(emptySession.get()));
  EXPECT_TRUE(config.isFileColumnNamesReadAsLowerCase(emptySession.get()));
  EXPECT_EQ(config.maxCoalescedBytes(emptySession.get()), 100);
  EXPECT_EQ(config.maxCoalescedDistanceBytes(emptySession.get()), 100 << 10);
  EXPECT_EQ(config.prefetchRowGroups(), 4);
  EXPECT_EQ(config.loadQuantum(emptySession.get()), 4 << 20);
  EXPECT_EQ(config.filePreloadThreshold(), 16UL << 20);
  EXPECT_TRUE(config.readStatsBasedFilterReorderDisabled(emptySession.get()));
  EXPECT_TRUE(config.preserveFlatMapsInMemory(emptySession.get()));
  EXPECT_TRUE(config.indexEnabled(emptySession.get()));
  EXPECT_TRUE(config.readerCollectColumnCpuMetrics(emptySession.get()));
  EXPECT_TRUE(config.fileMetadataCacheEnabled(emptySession.get()));
  EXPECT_TRUE(config.pinFileMetadata(emptySession.get()));
  EXPECT_EQ(config.orcFooterSpeculativeIoSize(emptySession.get()), 512UL << 10);
  EXPECT_EQ(
      config.parquetFooterSpeculativeIoSize(emptySession.get()), 1UL << 20);
  EXPECT_EQ(
      config.nimbleFooterSpeculativeIoSize(emptySession.get()), 4UL << 20);
}

TEST(FileConfigTest, overrideSession) {
  FileConfig config(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  std::unordered_map<std::string, std::string> sessionOverride = {
      {FileConfig::kOrcUseColumnNamesSession, "true"},
      {FileConfig::kParquetUseColumnNamesSession, "true"},
      {FileConfig::kFileColumnNamesReadAsLowerCaseSession, "true"},
      {FileConfig::kIgnoreMissingFilesSession, "true"},
      {FileConfig::kMaxCoalescedDistanceSession, "3MB"},
      {FileConfig::kLoadQuantumSession, std::to_string(4 << 20)},
      {FileConfig::kReadStatsBasedFilterReorderDisabledSession, "true"},
      {FileConfig::kPreserveFlatMapsInMemorySession, "true"},
      {FileConfig::kIndexEnabledSession, "true"},
      {FileConfig::kReaderCollectColumnCpuMetricsSession, "true"},
      {FileConfig::kFileMetadataCacheEnabledSession, "true"},
      {FileConfig::kPinFileMetadataSession, "true"},
      {FileConfig::kOrcFooterSpeculativeIoSizeSession,
       std::to_string(128UL << 10)},
      {FileConfig::kParquetFooterSpeculativeIoSizeSession,
       std::to_string(512UL << 10)},
      {FileConfig::kNimbleFooterSpeculativeIoSizeSession,
       std::to_string(2UL << 20)},
  };
  const auto session =
      std::make_unique<config::ConfigBase>(std::move(sessionOverride));

  EXPECT_TRUE(config.isOrcUseColumnNames(session.get()));
  EXPECT_TRUE(config.isParquetUseColumnNames(session.get()));
  EXPECT_TRUE(config.isFileColumnNamesReadAsLowerCase(session.get()));
  EXPECT_TRUE(config.ignoreMissingFiles(session.get()));
  EXPECT_EQ(config.maxCoalescedDistanceBytes(session.get()), 3 << 20);
  EXPECT_EQ(config.loadQuantum(session.get()), 4 << 20);
  EXPECT_TRUE(config.readStatsBasedFilterReorderDisabled(session.get()));
  EXPECT_TRUE(config.preserveFlatMapsInMemory(session.get()));
  EXPECT_TRUE(config.indexEnabled(session.get()));
  EXPECT_TRUE(config.readerCollectColumnCpuMetrics(session.get()));
  EXPECT_TRUE(config.fileMetadataCacheEnabled(session.get()));
  EXPECT_TRUE(config.pinFileMetadata(session.get()));
  EXPECT_EQ(config.orcFooterSpeculativeIoSize(session.get()), 128UL << 10);
  EXPECT_EQ(config.parquetFooterSpeculativeIoSize(session.get()), 512UL << 10);
  EXPECT_EQ(config.nimbleFooterSpeculativeIoSize(session.get()), 2UL << 20);
}

TEST(FileConfigTest, nullConfig) {
  VELOX_ASSERT_THROW(
      FileConfig(nullptr), "Config is null for FileConfig initialization");
}

TEST(FileConfigTest, invalidTimestampUnit) {
  FileConfig config(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  std::unordered_map<std::string, std::string> sessionOverride = {
      {FileConfig::kReadTimestampUnitSession, "5"},
  };
  const auto session =
      std::make_unique<config::ConfigBase>(std::move(sessionOverride));
  VELOX_ASSERT_THROW(
      config.readTimestampUnit(session.get()), "Invalid timestamp unit.");
}
