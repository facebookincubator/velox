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
#pragma once

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/storage_adapters/hdfs/HdfsWriteFile.h"
#include "velox/core/Context.h"
#include "velox/dwio/common/DataSink.h"

namespace facebook::velox {

class HdfsFileSink : public facebook::velox::dwio::common::DataSink {
 public:
  explicit HdfsFileSink(
      const std::string& fullDestinationPath,
      const facebook::velox::dwio::common::MetricsLogPtr& metricLogger =
          facebook::velox::dwio::common::MetricsLog::voidLog(),
      facebook::velox::dwio::common::IoStatistics* stats = nullptr)
      : facebook::velox::dwio::common::DataSink{
            "HdfsFileSink",
            metricLogger,
            stats} {
    auto destinationPathStartPos = fullDestinationPath.substr(7).find("/", 0);
    std::string destinationPath =
        fullDestinationPath.substr(destinationPathStartPos + 7);
    auto hdfsFileSystem =
        filesystems::getFileSystem(fullDestinationPath, nullptr);
    file_ = hdfsFileSystem->openFileForWrite(destinationPath);
  }

  ~HdfsFileSink() override {
    destroy();
  }

  using facebook::velox::dwio::common::DataSink::write;

  void write(std::vector<facebook::velox::dwio::common::DataBuffer<char>>&
                 buffers) override;

  static void registerFactory();

 protected:
  void doClose() override {
    file_->close();
  }

 private:
  std::unique_ptr<WriteFile> file_;
};
} // namespace facebook::velox
