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

#include "velox/connectors/hive/FileHandle.h"

#include "gtest/gtest.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/HiveConfig.h"

using namespace facebook::velox::connector::hive;
using facebook::velox::connector::hive::HiveConfig;

TEST(HiveConfigTest, defaultConfig) {
  HiveConfig* hiveConfig = new HiveConfig({});
  ASSERT_EQ(
      hiveConfig->insertExistingPartitionsBehavior(),
      InsertExistingPartitionsBehavior::kError);
  ASSERT_EQ(hiveConfig->maxPartitionsPerWriters(), 100);
  ASSERT_EQ(hiveConfig->immutablePartitions(), false);
  ASSERT_EQ(hiveConfig->s3UseVirtualAddressing(), false);
  ASSERT_EQ(hiveConfig->s3GetLogLevel(), "FATAL");
  ASSERT_EQ(hiveConfig->s3UseSSL(), true);
  ASSERT_EQ(hiveConfig->s3UseInstanceCredentials(), false);
  ASSERT_EQ(hiveConfig->s3Endpoint(), "hey");
  ASSERT_EQ(hiveConfig->s3AccessKey(), std::nullopt);
  ASSERT_EQ(hiveConfig->s3SecretKey(), std::nullopt);
  ASSERT_EQ(hiveConfig->s3IAMRole(), std::nullopt);
  ASSERT_EQ(hiveConfig->s3IAMRoleSessionName(), "velox-session");
  ASSERT_EQ(hiveConfig->gcsEndpoint(), "");
  ASSERT_EQ(hiveConfig->gcsScheme(), "https");
  ASSERT_EQ(hiveConfig->gcsCredentials(), "");
  ASSERT_EQ(hiveConfig->isOrcUseColumnNames(), false);
  ASSERT_EQ(hiveConfig->isFileColumnNamesReadAsLowerCase(), false);

  ASSERT_EQ(hiveConfig->maxCoalescedBytes(), 128 << 20);
  ASSERT_EQ(hiveConfig->maxCoalescedDistanceBytes(), 512 << 10);
  ASSERT_EQ(hiveConfig->numCacheFileHandles(), 20'000);
  ASSERT_EQ(hiveConfig->isFileHandleCacheEnabled(), true);
  ASSERT_EQ(hiveConfig->sortWriterMaxOutputRows(), 1024);
  ASSERT_EQ(hiveConfig->sortWriterMaxOutputBytes(), 10UL << 20);
  ASSERT_EQ(hiveConfig->getOrcWriterMaxStripeSize(), 64L * 1024L * 1024L);
  ASSERT_EQ(hiveConfig->getOrcWriterMaxDictionaryMemory(), 16L * 1024L * 1024L);
}

TEST(HiveConfigTest, overrideConfig) {
  std::unordered_map<std::string, std::string> configFromFile = {
      {HiveConfig::kInsertExistingPartitionsBehavior, "OVERWRITE"},
      {HiveConfig::kMaxPartitionsPerWriters, "120"},
      {HiveConfig::kImmutablePartitions, "true"},
      {HiveConfig::kS3PathStyleAccess, "true"},
      {HiveConfig::kS3LogLevel, "Warning"},
      {HiveConfig::kS3SSLEnabled, "false"},
      {HiveConfig::kS3UseInstanceCredentials, "true"},
      {HiveConfig::kS3Endpoint, "hey"},
      {HiveConfig::kS3AwsAccessKey, "hello"},
      {HiveConfig::kS3AwsSecretKey, "hello"},
      {HiveConfig::kS3IamRole, "hello"},
      {HiveConfig::kS3IamRoleSessionName, "velox"},
      {HiveConfig::kGCSEndpoint, "hey"},
      {HiveConfig::kGCSScheme, "http"},
      {HiveConfig::kGCSCredentials, "hey"},
      {HiveConfig::kOrcUseColumnNames, "true"},
      {HiveConfig::kFileColumnNamesReadAsLowerCase, "true"},
      {HiveConfig::kMaxCoalescedBytes, "100"},
      {HiveConfig::kMaxCoalescedDistanceBytes, "100"},
      {HiveConfig::kNumCacheFileHandles, "100"},
      {HiveConfig::kEnableFileHandleCache, "false"},
      {HiveConfig::kOrcWriterMaxStripeSize, "100"},
      {HiveConfig::kOrcWriterMaxDictionaryMemory, "100"},
      {HiveConfig::kSortWriterMaxOutputRows, "100"},
      {HiveConfig::kSortWriterMaxOutputBytes, "100"}};
  HiveConfig* hiveConfig = new HiveConfig(configFromFile);
  ASSERT_EQ(
      hiveConfig->insertExistingPartitionsBehavior(),
      InsertExistingPartitionsBehavior::kOverwrite);
  ASSERT_EQ(hiveConfig->maxPartitionsPerWriters(), 120);
  ASSERT_EQ(hiveConfig->immutablePartitions(), true);
  ASSERT_EQ(hiveConfig->s3UseVirtualAddressing(), true);
  ASSERT_EQ(hiveConfig->s3GetLogLevel(), "Warning");
  ASSERT_EQ(hiveConfig->s3UseSSL(), false);
  ASSERT_EQ(hiveConfig->s3UseInstanceCredentials(), true);
  ASSERT_EQ(hiveConfig->s3Endpoint(), "hey");
  ASSERT_EQ(hiveConfig->s3AccessKey(), std::optional("hello"));
  ASSERT_EQ(hiveConfig->s3SecretKey(), std::optional("hello"));
  ASSERT_EQ(hiveConfig->s3IAMRole(), std::optional("hello"));
  ASSERT_EQ(hiveConfig->s3IAMRoleSessionName(), "velox");
  ASSERT_EQ(hiveConfig->gcsEndpoint(), "hey");
  ASSERT_EQ(hiveConfig->gcsScheme(), "http");
  ASSERT_EQ(hiveConfig->gcsCredentials(), "hey");
  ASSERT_EQ(hiveConfig->isOrcUseColumnNames(), true);
  ASSERT_EQ(hiveConfig->isFileColumnNamesReadAsLowerCase(), true);
  ASSERT_EQ(hiveConfig->maxCoalescedBytes(), 100);
  ASSERT_EQ(hiveConfig->maxCoalescedDistanceBytes(), 100);
  ASSERT_EQ(hiveConfig->numCacheFileHandles(), 100);
  ASSERT_EQ(hiveConfig->isFileHandleCacheEnabled(), false);
  ASSERT_EQ(hiveConfig->sortWriterMaxOutputRows(), 100);
  ASSERT_EQ(hiveConfig->sortWriterMaxOutputBytes(), 100);
  ASSERT_EQ(hiveConfig->getOrcWriterMaxStripeSize(), 100);
  ASSERT_EQ(hiveConfig->getOrcWriterMaxDictionaryMemory(), 100);
}
