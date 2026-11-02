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
#include "velox/connectors/hive/FileConnector.h"

#include <folly/executors/InlineExecutor.h>
#include <gtest/gtest.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempFilePath.h"

namespace facebook::velox::connector::hive {
namespace {

class TestFileConnector : public FileConnector {
 public:
  TestFileConnector(
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* executor)
      : FileConnector(
            "test-file",
            std::make_shared<FileConfig>(std::move(config), "test-file."),
            executor) {}

  FileHandleCachedPtr open(const std::string& path) {
    return fileHandleFactory_.generate(FileHandleKey{path});
  }

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr&,
      const ConnectorTableHandlePtr&,
      const ColumnHandleMap&,
      ConnectorQueryCtx*) override {
    VELOX_UNSUPPORTED("Test connector does not create data sources");
  }

  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr,
      ConnectorInsertTableHandlePtr,
      ConnectorQueryCtx*,
      CommitStrategy) override {
    VELOX_UNSUPPORTED("Test connector does not create data sinks");
  }
};

TEST(FileConnectorTest, resources) {
  filesystems::registerLocalFileSystem();
  const auto path = common::testutil::TempFilePath::create();
  folly::InlineExecutor executor;
  for (bool cacheEnabled : {false, true}) {
    SCOPED_TRACE(cacheEnabled);
    const auto config = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{
            {FileConfig::kEnableFileHandleCache,
             cacheEnabled ? "true" : "false"},
            {FileConfig::kNumCacheFileHandles, "10"},
            {FileConfig::kFileHandleExpirationDurationMs, "60000"}});
    TestFileConnector connector(config, &executor);
    EXPECT_EQ(connector.connectorConfig(), config);
    EXPECT_EQ(connector.ioExecutor(), &executor);
    EXPECT_EQ(connector.fileHandleCacheStats().maxSize, cacheEnabled ? 10 : 0);
    EXPECT_EQ(
        connector.fileHandleCacheStats().expireDurationMs,
        cacheEnabled ? 60000 : 0);
    {
      auto first = connector.open(path->getPath());
      auto second = connector.open(path->getPath());
      EXPECT_EQ(first->file->size(), 0);
      EXPECT_EQ(first->file == second->file, cacheEnabled);
    }
    EXPECT_EQ(
        connector.fileHandleCacheStats().numElements, cacheEnabled ? 1 : 0);
    EXPECT_EQ(connector.clearFileHandleCache().numElements, 0);
  }
}

TEST(FileConnectorTest, defaultResources) {
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  TestFileConnector connector(config, nullptr);
  EXPECT_EQ(connector.ioExecutor(), nullptr);
  EXPECT_EQ(connector.fileHandleCacheStats().maxSize, 20'000);
  EXPECT_EQ(connector.fileHandleCacheStats().expireDurationMs, 0);
  EXPECT_FALSE(connector.canAddDynamicFilter());
  EXPECT_FALSE(connector.supportsSplitPreload());
  EXPECT_FALSE(connector.supportsIndexLookup());
}

} // namespace
} // namespace facebook::velox::connector::hive
