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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

namespace facebook::velox::parquet::test {

class ParquetTableScanTestBase : public exec::test::HiveConnectorTestBase {
 protected:
  static std::string parquetSessionProperty(std::string_view key) {
    return dwio::common::formatConfigPrefix(
               dwio::common::FileFormat::PARQUET, "_") +
        std::string(key);
  }

  void SetUp() override {
    exec::test::HiveConnectorTestBase::SetUp();
    registerParquetReaderFactory();
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& filePath,
      const std::optional<
          std::unordered_map<std::string, std::optional<std::string>>>&
          partitionKeys = std::nullopt,
      const std::optional<std::unordered_map<std::string, std::string>>&
          infoColumns = std::nullopt) {
    return makeHiveConnectorSplits(
        filePath,
        1,
        dwio::common::FileFormat::PARQUET,
        partitionKeys,
        infoColumns)[0];
  }

  void writeToParquetFile(
      const std::string& path,
      const std::vector<RowVectorPtr>& data,
      ParquetWriterOptions options = {}) {
    dwio::common::WriterOptions writerOptions;
    writeToParquetFile(
        path, data, std::move(writerOptions), std::move(options));
  }

  void writeToParquetFile(
      const std::string& path,
      const std::vector<RowVectorPtr>& data,
      dwio::common::WriterOptions writerOptions,
      ParquetWriterOptions options) {
    VELOX_CHECK_GT(data.size(), 0);

    auto writeFile = std::make_unique<LocalWriteFile>(path, true, false);
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), path);
    auto childPool =
        rootPool_->addAggregateChild("ParquetTableScanTestBase.Writer");
    writerOptions.memoryPool = childPool.get();
    writerOptions.formatSpecificOptions =
        std::make_shared<ParquetWriterOptions>(std::move(options));
    auto writer = std::make_unique<Writer>(
        std::move(sink), writerOptions, asRowType(data[0]->type()));

    for (const auto& vector : data) {
      writer->write(vector);
    }
    writer->close();
  }
};

} // namespace facebook::velox::parquet::test
