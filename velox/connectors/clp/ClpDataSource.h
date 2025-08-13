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

#include <set>

#include "velox/connectors/Connector.h"
#include "velox/connectors/clp/ClpConfig.h"
#include "velox/connectors/clp/search_lib/ClpCursor.h"

namespace clp_s {
class BaseColumnReader;
} // namespace clp_s

namespace facebook::velox::connector::clp {

class ClpDataSource : public DataSource {
 public:
  ClpDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      velox::memory::MemoryPool* pool,
      std::shared_ptr<const ClpConfig>& clpConfig);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  std::optional<RowVectorPtr> next(uint64_t size, velox::ContinueFuture& future)
      override;

  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) override {
    VELOX_NYI("Dynamic filters not supported by ClpConnector.");
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override {
    return {};
  }

 private:
  /// Recursively adds fields from the column type to the list of fields to be
  /// retrieved from the data source.
  ///
  /// @param columnType
  /// @param parentName The name of the parent field (used for nested fields).
  void addFieldsRecursively(
      const TypePtr& columnType,
      const std::string& parentName);

  /// Creates a Vector of the specified type and size.
  ///
  /// This method recursively creates vectors for complex types like ROW. For
  /// primitive types, it creates a LazyVector that will load the data from the
  /// underlying data source when it is accessed.
  ///
  /// @param vectorType
  /// @param vectorSize
  /// @param projectedColumns The readers of the projected columns.
  /// @param filteredRows The rows to be read.
  /// @param readerIndex The index of the column reader.
  /// @return A Vector of the specified type and size.
  VectorPtr createVector(
      const TypePtr& vectorType,
      size_t vectorSize,
      const std::vector<clp_s::BaseColumnReader*>& projectedColumns,
      const std::shared_ptr<std::vector<uint64_t>>& filteredRows,
      size_t& readerIndex);

  ClpConfig::StorageType storageType_;
  velox::memory::MemoryPool* pool_;
  RowTypePtr outputType_;
  std::set<std::string> columnUntypedNames_;
  uint64_t completedRows_{0};
  uint64_t completedBytes_{0};

  std::vector<search_lib::Field> fields_;

  std::unique_ptr<search_lib::ClpCursor> cursor_;
};

} // namespace facebook::velox::connector::clp
