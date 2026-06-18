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

#include "velox/connectors/hive/HiveConfig.h"
#include "gtest/gtest.h"
#include "velox/common/config/Config.h"
#include "velox/dwio/common/Options.h"

#include <algorithm>

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using facebook::velox::connector::hive::HiveConfig;

TEST(HiveConfigTest, defaultConfig) {
  HiveConfig hiveConfig(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  const auto emptySession = std::make_unique<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());
  ASSERT_EQ(
      hiveConfig.insertExistingPartitionsBehavior(emptySession.get()),
      facebook::velox::connector::hive::HiveConfig::
          InsertExistingPartitionsBehavior::kError);
  ASSERT_EQ(hiveConfig.maxPartitionsPerWriters(emptySession.get()), 128);
  ASSERT_EQ(hiveConfig.immutablePartitions(), false);
  ASSERT_EQ(hiveConfig.gcsEndpoint(), "");
  ASSERT_EQ(hiveConfig.gcsCredentialsPath(), "");
  ASSERT_FALSE(hiveConfig.isOrcUseColumnNames(emptySession.get()));
  ASSERT_FALSE(hiveConfig.isFileColumnNamesReadAsLowerCase(emptySession.get()));

  ASSERT_EQ(hiveConfig.maxCoalescedBytes(emptySession.get()), 128 << 20);
  ASSERT_EQ(
      hiveConfig.maxCoalescedDistanceBytes(emptySession.get()), 512 << 10);
  ASSERT_FALSE(
      hiveConfig.readStatsBasedFilterReorderDisabled(emptySession.get()));
  ASSERT_EQ(hiveConfig.numCacheFileHandles(), 20'000);
  ASSERT_TRUE(hiveConfig.isFileHandleCacheEnabled());
  ASSERT_EQ(hiveConfig.sortWriterMaxOutputRows(emptySession.get()), 1024);
  ASSERT_EQ(
      hiveConfig.sortWriterMaxOutputBytes(emptySession.get()), 10UL << 20);
  ASSERT_EQ(
      hiveConfig.sortWriterFinishTimeSliceLimitMs(emptySession.get()), 5'000);
  ASSERT_TRUE(hiveConfig.isPartitionPathAsLowerCase(emptySession.get()));
  ASSERT_TRUE(hiveConfig.allowNullPartitionKeys(emptySession.get()));
  ASSERT_EQ(hiveConfig.loadQuantum(emptySession.get()), 8 << 20);
  ASSERT_FALSE(hiveConfig.preserveFlatMapsInMemory(emptySession.get()));
  ASSERT_FALSE(hiveConfig.indexEnabled(emptySession.get()));
  ASSERT_FALSE(hiveConfig.cacheMetadata(emptySession.get()));
  ASSERT_FALSE(hiveConfig.cacheIndex(emptySession.get()));
  ASSERT_EQ(
      hiveConfig.orcFooterSpeculativeIoSize(emptySession.get()), 256UL << 10);
  ASSERT_EQ(
      hiveConfig.nimbleFooterSpeculativeIoSize(emptySession.get()), 8UL << 20);
  ASSERT_FALSE(hiveConfig.nimbleStringDecoderZeroCopy(emptySession.get()));
  ASSERT_FALSE(hiveConfig.nimblePreserveDictionaryEncoding(emptySession.get()));
}

TEST(HiveConfigTest, overrideConfig) {
  std::unordered_map<std::string, std::string> configFromFile = {
      {HiveConfig::kInsertExistingPartitionsBehavior, "OVERWRITE"},
      {HiveConfig::kMaxPartitionsPerWriters, "120"},
      {HiveConfig::kImmutablePartitions, "true"},
      {HiveConfig::kGcsEndpoint, "hey"},
      {HiveConfig::kGcsCredentialsPath, "hey"},
      {HiveConfig::kOrcUseColumnNames, "true"},
      {HiveConfig::kFileColumnNamesReadAsLowerCase, "true"},
      {HiveConfig::kAllowNullPartitionKeys, "false"},
      {HiveConfig::kMaxCoalescedBytes, "100"},
      {HiveConfig::kMaxCoalescedDistance, "100kB"},
      {HiveConfig::kNumCacheFileHandles, "100"},
      {HiveConfig::kFileHandleExpirationDurationMs, "200"},
      {HiveConfig::kEnableFileHandleCache, "false"},
      {HiveConfig::kSortWriterMaxOutputRows, "100"},
      {HiveConfig::kSortWriterMaxOutputBytes, "100MB"},
      {HiveConfig::kSortWriterFinishTimeSliceLimitMs, "400"},
      {HiveConfig::kReadStatsBasedFilterReorderDisabled, "true"},
      {HiveConfig::kLoadQuantum, std::to_string(4 << 20)},
      {HiveConfig::kMaxBucketCount, std::to_string(100'000)},
      {HiveConfig::kPreserveFlatMapsInMemory, "true"},
      {HiveConfig::kIndexEnabled, "true"},
      {HiveConfig::kCacheMetadata, "true"},
      {HiveConfig::kCacheIndex, "true"},
      {HiveConfig::kOrcFooterSpeculativeIoSize, std::to_string(512UL << 10)},
      {HiveConfig::kNimbleFooterSpeculativeIoSize, std::to_string(4UL << 20)},
      {HiveConfig::kNimbleStringDecoderZeroCopy, "true"},
      {HiveConfig::kNimblePreserveDictionaryEncoding, "true"},
  };
  HiveConfig hiveConfig(
      std::make_shared<config::ConfigBase>(std::move(configFromFile)));
  auto emptySession = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());
  ASSERT_EQ(
      hiveConfig.insertExistingPartitionsBehavior(emptySession.get()),
      facebook::velox::connector::hive::HiveConfig::
          InsertExistingPartitionsBehavior::kOverwrite);
  ASSERT_EQ(hiveConfig.maxPartitionsPerWriters(emptySession.get()), 120);
  ASSERT_TRUE(hiveConfig.immutablePartitions());
  ASSERT_EQ(hiveConfig.gcsEndpoint(), "hey");
  ASSERT_EQ(hiveConfig.gcsCredentialsPath(), "hey");
  ASSERT_TRUE(hiveConfig.isOrcUseColumnNames(emptySession.get()));
  ASSERT_TRUE(hiveConfig.isFileColumnNamesReadAsLowerCase(emptySession.get()));
  ASSERT_FALSE(hiveConfig.allowNullPartitionKeys(emptySession.get()));
  ASSERT_EQ(hiveConfig.maxCoalescedBytes(emptySession.get()), 100);
  ASSERT_EQ(
      hiveConfig.maxCoalescedDistanceBytes(emptySession.get()), 100 << 10);
  ASSERT_EQ(hiveConfig.numCacheFileHandles(), 100);
  ASSERT_EQ(hiveConfig.fileHandleExpirationDurationMs(), 200);
  ASSERT_FALSE(hiveConfig.isFileHandleCacheEnabled());
  ASSERT_EQ(hiveConfig.sortWriterMaxOutputRows(emptySession.get()), 100);
  ASSERT_EQ(
      hiveConfig.sortWriterMaxOutputBytes(emptySession.get()), 100UL << 20);
  ASSERT_EQ(
      hiveConfig.sortWriterFinishTimeSliceLimitMs(emptySession.get()), 400);
  ASSERT_TRUE(
      hiveConfig.readStatsBasedFilterReorderDisabled(emptySession.get()));
  ASSERT_EQ(hiveConfig.loadQuantum(emptySession.get()), 4 << 20);
  ASSERT_EQ(hiveConfig.maxBucketCount(emptySession.get()), 100'000);
  ASSERT_TRUE(hiveConfig.preserveFlatMapsInMemory(emptySession.get()));
  ASSERT_TRUE(hiveConfig.indexEnabled(emptySession.get()));
  ASSERT_TRUE(hiveConfig.cacheMetadata(emptySession.get()));
  ASSERT_TRUE(hiveConfig.cacheIndex(emptySession.get()));
  ASSERT_EQ(
      hiveConfig.orcFooterSpeculativeIoSize(emptySession.get()), 512UL << 10);
  ASSERT_EQ(
      hiveConfig.nimbleFooterSpeculativeIoSize(emptySession.get()), 4UL << 20);
  ASSERT_TRUE(hiveConfig.nimbleStringDecoderZeroCopy(emptySession.get()));
  ASSERT_TRUE(hiveConfig.nimblePreserveDictionaryEncoding(emptySession.get()));
}

TEST(HiveConfigTest, overrideSession) {
  HiveConfig hiveConfig(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  std::unordered_map<std::string, std::string> sessionOverride = {
      {HiveConfig::kInsertExistingPartitionsBehaviorSession, "OVERWRITE"},
      {HiveConfig::kOrcUseColumnNamesSession, "true"},
      {HiveConfig::kFileColumnNamesReadAsLowerCaseSession, "true"},
      {HiveConfig::kSortWriterMaxOutputRowsSession, "20"},
      {HiveConfig::kSortWriterMaxOutputBytesSession, "20MB"},
      {HiveConfig::kMaxCoalescedDistanceSession, "3MB"},
      {HiveConfig::kSortWriterFinishTimeSliceLimitMsSession, "300"},
      {HiveConfig::kPartitionPathAsLowerCaseSession, "false"},
      {HiveConfig::kAllowNullPartitionKeysSession, "false"},
      {HiveConfig::kIgnoreMissingFilesSession, "true"},
      {HiveConfig::kReadStatsBasedFilterReorderDisabledSession, "true"},
      {HiveConfig::kLoadQuantumSession, std::to_string(4 << 20)},
      {HiveConfig::kPreserveFlatMapsInMemorySession, "true"},
      {HiveConfig::kIndexEnabledSession, "true"},
      {HiveConfig::kCacheMetadataSession, "true"},
      {HiveConfig::kCacheIndexSession, "true"},
      {HiveConfig::kOrcFooterSpeculativeIoSizeSession,
       std::to_string(128UL << 10)},
      {HiveConfig::kNimbleFooterSpeculativeIoSizeSession,
       std::to_string(2UL << 20)},
      {HiveConfig::kNimbleStringDecoderZeroCopySession, "true"},
      {HiveConfig::kNimblePreserveDictionaryEncodingSession, "true"},
  };
  const auto session =
      std::make_unique<config::ConfigBase>(std::move(sessionOverride));
  ASSERT_EQ(
      hiveConfig.insertExistingPartitionsBehavior(session.get()),
      facebook::velox::connector::hive::HiveConfig::
          InsertExistingPartitionsBehavior::kOverwrite);
  ASSERT_EQ(hiveConfig.maxPartitionsPerWriters(session.get()), 128);
  ASSERT_FALSE(hiveConfig.immutablePartitions());
  ASSERT_EQ(hiveConfig.gcsEndpoint(), "");
  ASSERT_EQ(hiveConfig.gcsCredentialsPath(), "");
  ASSERT_TRUE(hiveConfig.isOrcUseColumnNames(session.get()));
  ASSERT_TRUE(hiveConfig.isFileColumnNamesReadAsLowerCase(session.get()));

  ASSERT_EQ(hiveConfig.maxCoalescedBytes(session.get()), 128 << 20);
  ASSERT_EQ(hiveConfig.maxCoalescedDistanceBytes(session.get()), 3 << 20);
  ASSERT_EQ(hiveConfig.numCacheFileHandles(), 20'000);
  ASSERT_TRUE(hiveConfig.isFileHandleCacheEnabled());
  ASSERT_EQ(hiveConfig.sortWriterMaxOutputRows(session.get()), 20);
  ASSERT_EQ(hiveConfig.sortWriterMaxOutputBytes(session.get()), 20UL << 20);
  ASSERT_EQ(hiveConfig.sortWriterFinishTimeSliceLimitMs(session.get()), 300);
  ASSERT_FALSE(hiveConfig.isPartitionPathAsLowerCase(session.get()));
  ASSERT_FALSE(hiveConfig.allowNullPartitionKeys(session.get()));
  ASSERT_TRUE(hiveConfig.ignoreMissingFiles(session.get()));
  ASSERT_TRUE(hiveConfig.readStatsBasedFilterReorderDisabled(session.get()));
  ASSERT_EQ(hiveConfig.loadQuantum(session.get()), 4 << 20);
  ASSERT_TRUE(hiveConfig.preserveFlatMapsInMemory(session.get()));
  ASSERT_TRUE(hiveConfig.indexEnabled(session.get()));
  ASSERT_TRUE(hiveConfig.cacheMetadata(session.get()));
  ASSERT_TRUE(hiveConfig.cacheIndex(session.get()));
  ASSERT_EQ(hiveConfig.orcFooterSpeculativeIoSize(session.get()), 128UL << 10);
  ASSERT_EQ(hiveConfig.nimbleFooterSpeculativeIoSize(session.get()), 2UL << 20);
  ASSERT_TRUE(hiveConfig.nimbleStringDecoderZeroCopy(session.get()));
  ASSERT_TRUE(hiveConfig.nimblePreserveDictionaryEncoding(session.get()));
}

TEST(HiveConfigTest, maxTargetFileSizeConfigAndSessionKeys) {
  auto emptySession = std::make_unique<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());

  HiveConfig config(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{
              {HiveConfig::kParquetMaxTargetFileSize, "16MB"},
              {HiveConfig::kOrcMaxTargetFileSize, "24MB"},
              {HiveConfig::kNimbleMaxTargetFileSize, "40MB"},
          }));
  EXPECT_EQ(
      config.maxTargetFileSizeBytes(
          dwio::common::FileFormat::PARQUET, emptySession.get()),
      16UL << 20);
  EXPECT_EQ(
      config.maxTargetFileSizeBytes(
          dwio::common::FileFormat::DWRF, emptySession.get()),
      24UL << 20);
  EXPECT_EQ(
      config.maxTargetFileSizeBytes(
          dwio::common::FileFormat::ORC, emptySession.get()),
      24UL << 20);
  EXPECT_EQ(
      config.maxTargetFileSizeBytes(
          dwio::common::FileFormat::NIMBLE, emptySession.get()),
      40UL << 20);

  auto session = std::make_unique<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {HiveConfig::kParquetMaxTargetFileSizeSession, "32MB"},
          {HiveConfig::kOrcMaxTargetFileSizeSession, "48MB"},
          {HiveConfig::kNimbleMaxTargetFileSizeSession, "56MB"},
      });
  EXPECT_EQ(
      config.maxTargetFileSizeBytes(
          dwio::common::FileFormat::PARQUET, session.get()),
      32UL << 20);
  EXPECT_EQ(
      config.maxTargetFileSizeBytes(
          dwio::common::FileFormat::DWRF, session.get()),
      48UL << 20);
  EXPECT_EQ(
      config.maxTargetFileSizeBytes(
          dwio::common::FileFormat::ORC, session.get()),
      48UL << 20);
  EXPECT_EQ(
      config.maxTargetFileSizeBytes(
          dwio::common::FileFormat::NIMBLE, session.get()),
      56UL << 20);
}

TEST(HiveConfigTest, registeredParquetPropertiesUseSessionPrefix) {
  const auto& properties = HiveConfig::registeredProperties();

  auto hasProperty = [&](const std::string& name) {
    return std::any_of(
        properties.begin(), properties.end(), [&](const auto& property) {
          return property.name == name;
        });
  };

  EXPECT_TRUE(hasProperty(HiveConfig::kParquetUseColumnNamesSession));
  EXPECT_FALSE(hasProperty(HiveConfig::kParquetUseColumnNames));
}
