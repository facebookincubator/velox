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

/// Base class for table handles in file-based connectors.
///
/// Define the common interface for table-level metadata needed by
/// FileDataSource and FileSplitReader. Connector-specific table handles
/// (HiveTableHandle, etc.) extend this class.
class FileTableHandle : public ConnectorTableHandle {
 public:
  using ConnectorTableHandle::ConnectorTableHandle;

  /// Single-field filters that can be applied efficiently during file reading.
  virtual const common::SubfieldFilters& subfieldFilters() const = 0;

  /// Remaining filter expression that cannot be converted into subfield
  /// filters. Usually less efficient but supports arbitrary boolean
  /// expressions.
  virtual const core::TypedExprPtr& remainingFilter() const = 0;

  /// Sampling rate between 0 and 1 (excluding 0). 0.1 means 10% sampling.
  /// 1.0 means no sampling.
  virtual double sampleRate() const = 0;

  /// Subset of schema stored in data files (non-partitioning columns).
  /// Needed for reading TEXTFILE, handling schema evolution, etc.
  virtual const RowTypePtr& dataColumns() const = 0;

  /// Extra parameters passed down to the file format reader layer.
  virtual const std::unordered_map<std::string, std::string>& tableParameters()
      const = 0;

  /// Extra columns used in filters but not in the output. Returns column
  /// handles as FileColumnHandlePtr for use in the generic scan pipeline.
  virtual std::vector<FileColumnHandlePtr> filterColumnHandles() const = 0;

  /// Database or namespace name for this table.
  virtual const std::string& dbName() const = 0;
};

using FileTableHandlePtr = std::shared_ptr<const FileTableHandle>;

} // namespace facebook::velox::connector::hive
