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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileColumnHandle.h"
#include "velox/core/ITypedExpr.h"
#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"

namespace facebook::velox::connector::hive {

/// Common table metadata and accessors for file-based connectors. Used by
/// FileDataSource and FileSplitReader. Connector-specific handles extend this
/// class with their own metadata and serialization.
class FileTableHandle : public ConnectorTableHandle {
 public:
  using ConnectorTableHandle::ConnectorTableHandle;

  const std::string& tableName() const {
    return tableName_;
  }

  const std::string& name() const override {
    return tableName_;
  }

  /// Single-field filters that can be applied efficiently during file reading.
  virtual const common::SubfieldFilters& subfieldFilters() const {
    return subfieldFilters_;
  }

  /// Remaining filter expression that cannot be converted into subfield
  /// filters. Usually less efficient but supports arbitrary boolean
  /// expressions.
  virtual const core::TypedExprPtr& remainingFilter() const {
    return remainingFilter_;
  }

  /// Sampling rate between 0 and 1 (excluding 0). 0.1 means 10% sampling.
  /// 1.0 means no sampling.
  virtual double sampleRate() const {
    return sampleRate_;
  }

  /// Subset of schema stored in data files (non-partitioning columns).
  /// Needed for reading TEXTFILE, handling schema evolution, etc.
  virtual const RowTypePtr& dataColumns() const {
    return dataColumns_;
  }

  /// Extra parameters passed down to the file format reader layer.
  virtual const std::unordered_map<std::string, std::string>& tableParameters()
      const {
    return tableParameters_;
  }

  /// Extra columns used in filters but not in the output. Returns column
  /// handles as FileColumnHandlePtr for use in the generic scan pipeline.
  /// Concrete handles own the storage to also support typed accessors.
  virtual std::vector<FileColumnHandlePtr> filterColumnHandles() const = 0;

  /// Database or namespace name for this table.
  virtual const std::string& dbName() const {
    return dbName_;
  }

 protected:
  FileTableHandle(
      std::string connectorId,
      std::string tableName,
      common::SubfieldFilters subfieldFilters,
      core::TypedExprPtr remainingFilter,
      RowTypePtr dataColumns,
      std::unordered_map<std::string, std::string> tableParameters,
      double sampleRate,
      std::string dbName);

  /// Serializes table identity, filters and sampling fields. Derived handles
  /// append their own fields without changing the shared query-plan format.
  folly::dynamic serializeFileFields(std::string_view typeName) const;

  static void deserializeFileFields(
      const folly::dynamic& obj,
      void* context,
      std::string& connectorId,
      std::string& tableName,
      common::SubfieldFilters& subfieldFilters,
      core::TypedExprPtr& remainingFilter,
      std::vector<FileColumnHandlePtr>& filterColumnHandles,
      double& sampleRate,
      std::string& dbName);

 private:
  const std::string tableName_;
  const common::SubfieldFilters subfieldFilters_;
  const core::TypedExprPtr remainingFilter_;
  const RowTypePtr dataColumns_;
  const std::unordered_map<std::string, std::string> tableParameters_;
  const double sampleRate_{1.0};
  const std::string dbName_;
};

using FileTableHandlePtr = std::shared_ptr<const FileTableHandle>;

} // namespace facebook::velox::connector::hive
