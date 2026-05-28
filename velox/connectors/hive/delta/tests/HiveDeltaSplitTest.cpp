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

#include "velox/connectors/hive/delta/HiveDeltaSplit.h"
#include <gtest/gtest.h>
#include "velox/dwio/common/FileSink.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::connector::hive::delta;

class HiveDeltaSplitTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(HiveDeltaSplitTest, basicConstruction) {
  auto split = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/delta/file.parquet",
      dwio::common::FileFormat::PARQUET);

  EXPECT_EQ("test-connector", split->connectorId);
  EXPECT_EQ("/path/to/delta/file.parquet", split->filePath);
  EXPECT_EQ(dwio::common::FileFormat::PARQUET, split->fileFormat);
  EXPECT_EQ(0, split->start);
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(), split->length);
  EXPECT_TRUE(split->partitionKeys.empty());
  EXPECT_FALSE(split->tableBucketNumber.has_value());
  EXPECT_TRUE(split->customSplitInfo.empty());
  EXPECT_TRUE(split->cacheable);
}

TEST_F(HiveDeltaSplitTest, constructionWithAllParameters) {
  FileProperties properties = {.fileSize = 1024000, .modificationTime = 1234567890};
  auto extraInfo = std::make_shared<std::string>("delta metadata");

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
  partitionKeys["year"] = "2024";
  partitionKeys["month"] = "02";
  partitionKeys["day"] = "18";

  std::unordered_map<std::string, std::string> customInfo;
  customInfo["table_format"] = "hive-delta";
  customInfo["delta_version"] = "1";

  std::unordered_map<std::string, std::string> infoColumns;
  infoColumns["$path"] = "/path/to/delta/file.parquet";
  infoColumns["$file_size"] = "1024000";
  infoColumns["$file_modified_time"] = "1234567890";

  auto split = std::make_shared<HiveDeltaSplit>(
      "delta-connector",
      "/delta/table/year=2024/month=02/day=18/file.parquet",
      dwio::common::FileFormat::PARQUET,
      100,
      50000,
      partitionKeys,
      std::optional<int32_t>(5),
      customInfo,
      extraInfo,
      true,
      infoColumns,
      properties);

  EXPECT_EQ("delta-connector", split->connectorId);
  EXPECT_EQ("/delta/table/year=2024/month=02/day=18/file.parquet", split->filePath);
  EXPECT_EQ(dwio::common::FileFormat::PARQUET, split->fileFormat);
  EXPECT_EQ(100, split->start);
  EXPECT_EQ(50000, split->length);

  // Verify partition keys
  EXPECT_EQ(3, split->partitionKeys.size());
  EXPECT_EQ("2024", split->partitionKeys.at("year").value());
  EXPECT_EQ("02", split->partitionKeys.at("month").value());
  EXPECT_EQ("18", split->partitionKeys.at("day").value());

  // Verify bucket number
  EXPECT_TRUE(split->tableBucketNumber.has_value());
  EXPECT_EQ(5, split->tableBucketNumber.value());

  // Verify custom split info
  EXPECT_EQ(2, split->customSplitInfo.size());
  EXPECT_EQ("hive-delta", split->customSplitInfo.at("table_format"));
  EXPECT_EQ("1", split->customSplitInfo.at("delta_version"));

  // Verify extra file info
  EXPECT_EQ("delta metadata", *split->extraFileInfo);

  // Verify info columns
  EXPECT_EQ(3, split->infoColumns.size());
  EXPECT_EQ("/path/to/delta/file.parquet", split->infoColumns.at("$path"));
  EXPECT_EQ("1024000", split->infoColumns.at("$file_size"));
  EXPECT_EQ("1234567890", split->infoColumns.at("$file_modified_time"));

  // Verify file properties
  EXPECT_TRUE(split->properties.has_value());
  EXPECT_EQ(1024000, split->properties.value().fileSize.value());
  EXPECT_EQ(1234567890, split->properties.value().modificationTime.value());

  // Verify cacheable
  EXPECT_TRUE(split->cacheable);
}

TEST_F(HiveDeltaSplitTest, partitionKeysWithNullValues) {
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
  partitionKeys["year"] = "2024";
  partitionKeys["month"] = std::nullopt; // NULL partition value
  partitionKeys["day"] = "18";

  auto split = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET,
      0UL,
      std::numeric_limits<uint64_t>::max(),
      partitionKeys,
      std::nullopt,
      std::unordered_map<std::string, std::string>{},
      std::shared_ptr<std::string>{},
      true,
      std::unordered_map<std::string, std::string>{},
      std::nullopt);

  EXPECT_EQ(3, split->partitionKeys.size());
  EXPECT_TRUE(split->partitionKeys.at("year").has_value());
  EXPECT_EQ("2024", split->partitionKeys.at("year").value());
  EXPECT_FALSE(split->partitionKeys.at("month").has_value());
  EXPECT_TRUE(split->partitionKeys.at("day").has_value());
  EXPECT_EQ("18", split->partitionKeys.at("day").value());
}

TEST_F(HiveDeltaSplitTest, deltaTableFormatMarker) {
  std::unordered_map<std::string, std::string> customInfo;
  customInfo["table_format"] = "hive-delta";

  auto split = std::make_shared<HiveDeltaSplit>(
      "delta-connector",
      "/delta/table/file.parquet",
      dwio::common::FileFormat::PARQUET,
      0UL,
      std::numeric_limits<uint64_t>::max(),
      std::unordered_map<std::string, std::optional<std::string>>{},
      std::nullopt,
      customInfo,
      std::shared_ptr<std::string>{},
      true,
      std::unordered_map<std::string, std::string>{},
      std::nullopt);

  EXPECT_EQ("hive-delta", split->customSplitInfo.at("table_format"));
}

TEST_F(HiveDeltaSplitTest, infoColumnsForMetadata) {
  std::unordered_map<std::string, std::string> infoColumns;
  infoColumns["$path"] = "/delta/table/part-00000.parquet";
  infoColumns["$file_size"] = "2048576";
  infoColumns["$file_modified_time"] = "1708257600";

  auto split = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/delta/table/part-00000.parquet",
      dwio::common::FileFormat::PARQUET,
      0UL,
      std::numeric_limits<uint64_t>::max(),
      std::unordered_map<std::string, std::optional<std::string>>{},
      std::nullopt,
      std::unordered_map<std::string, std::string>{},
      std::shared_ptr<std::string>{},
      true,
      infoColumns,
      std::nullopt);

  EXPECT_EQ(3, split->infoColumns.size());
  EXPECT_EQ("/delta/table/part-00000.parquet", split->infoColumns.at("$path"));
  EXPECT_EQ("2048576", split->infoColumns.at("$file_size"));
  EXPECT_EQ("1708257600", split->infoColumns.at("$file_modified_time"));
}

TEST_F(HiveDeltaSplitTest, filePropertiesOptional) {
  // Test without file properties
  auto splitWithoutProps = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET);

  EXPECT_FALSE(splitWithoutProps->properties.has_value());

  // Test with file properties
  FileProperties properties = {
      .fileSize = 512000,
      .modificationTime = 1700000000};

  auto splitWithProps = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET,
      0UL,
      std::numeric_limits<uint64_t>::max(),
      std::unordered_map<std::string, std::optional<std::string>>{},
      std::nullopt,
      std::unordered_map<std::string, std::string>{},
      std::shared_ptr<std::string>{},
      true,
      std::unordered_map<std::string, std::string>{},
      properties);

  EXPECT_TRUE(splitWithProps->properties.has_value());
  EXPECT_EQ(512000, splitWithProps->properties.value().fileSize.value());
  EXPECT_EQ(1700000000, splitWithProps->properties.value().modificationTime.value());
}

TEST_F(HiveDeltaSplitTest, cacheableFlag) {
  // Test cacheable = true
  auto cacheableSplit = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET,
      0UL,
      std::numeric_limits<uint64_t>::max(),
      std::unordered_map<std::string, std::optional<std::string>>{},
      std::nullopt,
      std::unordered_map<std::string, std::string>{},
      std::shared_ptr<std::string>{},
      true,
      std::unordered_map<std::string, std::string>{},
      std::nullopt);

  EXPECT_TRUE(cacheableSplit->cacheable);

  // Test cacheable = false
  auto nonCacheableSplit = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET,
      0UL,
      std::numeric_limits<uint64_t>::max(),
      std::unordered_map<std::string, std::optional<std::string>>{},
      std::nullopt,
      std::unordered_map<std::string, std::string>{},
      std::shared_ptr<std::string>{},
      false,
      std::unordered_map<std::string, std::string>{},
      std::nullopt);

  EXPECT_FALSE(nonCacheableSplit->cacheable);
}

TEST_F(HiveDeltaSplitTest, differentFileFormats) {
  // Test with Parquet
  auto parquetSplit = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET);
  EXPECT_EQ(dwio::common::FileFormat::PARQUET, parquetSplit->fileFormat);

  // Test with ORC
  auto orcSplit = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.orc",
      dwio::common::FileFormat::ORC);
  EXPECT_EQ(dwio::common::FileFormat::ORC, orcSplit->fileFormat);
}

TEST_F(HiveDeltaSplitTest, splitRanges) {
  // Test full file read
  auto fullFileSplit = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET);
  EXPECT_EQ(0, fullFileSplit->start);
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(), fullFileSplit->length);

  // Test partial file read
  auto partialSplit = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET,
      1000,
      50000);
  EXPECT_EQ(1000, partialSplit->start);
  EXPECT_EQ(50000, partialSplit->length);
}

TEST_F(HiveDeltaSplitTest, inheritanceFromHiveConnectorSplit) {
  auto deltaSplit = std::make_shared<HiveDeltaSplit>(
      "test-connector",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET);

  // Verify it's a HiveConnectorSplit
  HiveConnectorSplit* hiveSplit = deltaSplit.get();
  EXPECT_NE(nullptr, hiveSplit);
  EXPECT_EQ("test-connector", hiveSplit->connectorId);
  EXPECT_EQ("/path/to/file.parquet", hiveSplit->filePath);
}