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

#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

namespace facebook::velox::connector::hive::iceberg {

class HiveIcebergDataSource : public HiveDataSource {
 public:
  HiveIcebergDataSource(
      const RowTypePtr& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      FileHandleFactory* fileHandleFactory,
      core::ExpressionEvaluator* expressionEvaluator,
      cache::AsyncDataCache* cache,
      const std::string& scanId,
      folly::Executor* executor,
      const dwio::common::ReaderOptions& options,
      std::shared_ptr<velox::connector::ConnectorSplit> connectorSplit);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;
  uint64_t readNext(uint64_t size) override;

 private:
  void readPositionalDeletes(const IcebergDeleteFile& deleteFile);
  void readEqualityDeletes(const IcebergDeleteFile& deleteFile);
  void openDeletes(
      const IcebergDeleteFile& deleteFile,
      const RowTypePtr& deleteRowType,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& deleteColumnHandles);

  std::string connectorId_;
  BufferPtr deleteBitmap_;
  // The read offset to the beginning of the split in number of rows for the
  // current batch
  uint64_t readOffset_;
  // The file row position for the first row in the split
  uint64_t firstRowInSplit_;
  // Materialized delete file contents, one RowVectorPtr for each delete file
  std::vector<RowVectorPtr> deleteFileContents_;
  // The index into deleteFileContents_ vector for the current batch
  uint32_t deleteFileContentsOffset_;
  // The index within each delete file
  uint32_t deleteRowsOffset_;
  // Total number of delete rows in the split
  uint64_t numDeleteRowsInSplit_;
};

} // namespace facebook::velox::connector::hive::iceberg