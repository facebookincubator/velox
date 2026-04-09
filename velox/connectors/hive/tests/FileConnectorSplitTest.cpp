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

#include "velox/connectors/hive/FileConnectorSplit.h"
#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;

TEST(FileConnectorSplitTest, construction) {
  FileConnectorSplit split(
      "connectorId",
      "/path/to/file.parquet",
      dwio::common::FileFormat::PARQUET,
      100,
      5000,
      /*splitWeight=*/2,
      /*cacheable=*/false);

  EXPECT_EQ(split.connectorId, "connectorId");
  EXPECT_EQ(split.filePath, "/path/to/file.parquet");
  EXPECT_EQ(split.fileFormat, dwio::common::FileFormat::PARQUET);
  EXPECT_EQ(split.start, 100);
  EXPECT_EQ(split.length, 5000);
  EXPECT_EQ(split.splitWeight, 2);
  EXPECT_FALSE(split.cacheable);
  EXPECT_FALSE(split.properties.has_value());
}

TEST(FileConnectorSplitTest, defaults) {
  FileConnectorSplit split(
      "connectorId", "/file.orc", dwio::common::FileFormat::ORC);

  EXPECT_EQ(split.start, 0);
  EXPECT_EQ(split.length, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(split.splitWeight, 0);
  EXPECT_TRUE(split.cacheable);
  EXPECT_FALSE(split.properties.has_value());
}

TEST(FileConnectorSplitTest, size) {
  FileConnectorSplit split(
      "connectorId", "/file.dwrf", dwio::common::FileFormat::DWRF, 0, 12345);

  EXPECT_EQ(split.size(), 12345);
}

TEST(FileConnectorSplitTest, getFileName) {
  FileConnectorSplit split(
      "connectorId",
      "/path/to/data/part-00000.parquet",
      dwio::common::FileFormat::PARQUET);

  EXPECT_EQ(split.getFileName(), "part-00000.parquet");
}

TEST(FileConnectorSplitTest, getFileNameNoSlash) {
  FileConnectorSplit split(
      "connectorId", "file.orc", dwio::common::FileFormat::ORC);

  EXPECT_EQ(split.getFileName(), "file.orc");
}

TEST(FileConnectorSplitTest, fileProperties) {
  FileProperties props = {.fileSize = 1024, .modificationTime = 999};
  FileConnectorSplit split(
      "connectorId",
      "/file.parquet",
      dwio::common::FileFormat::PARQUET,
      0,
      std::numeric_limits<uint64_t>::max(),
      0,
      true,
      props);

  ASSERT_TRUE(split.properties.has_value());
  EXPECT_EQ(split.properties->fileSize.value(), 1024);
  EXPECT_EQ(split.properties->modificationTime.value(), 999);
}
