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

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/connectors/common/FileConnectorUtil.h"
#include "velox/connectors/common/FileScanState.h"
#include "velox/connectors/common/FileTableHandle.h"

namespace facebook::velox::exec {
class ExprSet;
}

namespace facebook::velox::connector {

class FileConfig;

/// Immutable column and predicate requirements for a file scan. Owns the
/// handles and subfields backing its derived requirements, but no query
/// context, compiled expression, reader, or mutable ScanSpec. DataSource owns
/// the execution ExprSet and lends it to construction only to derive column
/// demands.
///
/// A configuration can create independent mutable states for the same scan
/// requirements. Each state retains this configuration for borrowed inputs;
/// published filter payloads are shared and must not be modified in place.
class FileScanConfig {
 public:
  using Subfields =
      folly::F14FastMap<std::string, std::vector<const common::Subfield*>>;
  using ColumnHandles = std::unordered_map<std::string, FileColumnHandlePtr>;

  struct Options {
    double sampleRate{1.0};
    bool disableStatsBasedFilterReorder{false};
  };

  /// Resolves and checks the input columns before filter extraction or
  /// expression compilation, preserving the FileDataSource initialization
  /// order. Returns the handles to classify, including filter-only columns.
  static std::vector<FileColumnHandlePtr> resolveColumns(
      const RowTypePtr& outputType,
      const FileTableHandlePtr& tableHandle,
      const ColumnHandleMap& assignments,
      const ConnectorQueryCtx* context,
      const FileConfig& fileConfig);

  /// Inputs have been checked by resolveColumns. The remaining expression and
  /// filters are the results of the existing filter-extraction step.
  /// remainingFilterExprSet compiles remainingFilter (null iff absent), and is
  /// borrowed only during this constructor.
  FileScanConfig(
      const RowTypePtr& outputType,
      const FileTableHandlePtr& tableHandle,
      const ColumnHandleMap& assignments,
      const std::vector<FileColumnHandlePtr>& columns,
      common::SubfieldFilters filters,
      core::TypedExprPtr remainingFilter,
      const exec::ExprSet* remainingFilterExprSet,
      Options options);

  FileScanConfig(const FileScanConfig&) = delete;
  FileScanConfig& operator=(const FileScanConfig&) = delete;
  FileScanConfig(FileScanConfig&&) = delete;
  FileScanConfig& operator=(FileScanConfig&&) = delete;

  /// Creates independent mutable trees and filter-map entries without compiling
  /// an execution ExprSet. Does not prescribe when a caller creates or reuses
  /// state; Hive continues to reuse its state across serial splits.
  static std::unique_ptr<FileScanState> createState(
      const std::shared_ptr<const FileScanConfig>& config,
      const ConnectorQueryCtx* context);

  const RowTypePtr& outputType() const {
    return outputType_;
  }

  const RowTypePtr& readerOutputType() const {
    return readerOutputType_;
  }

  const FileTableHandlePtr& tableHandle() const {
    return tableHandle_;
  }

  const core::TypedExprPtr& remainingFilter() const {
    return remainingFilter_;
  }

  const common::SubfieldFilters& filters() const {
    return filters_;
  }

  double sampleRate() const {
    return options_.sampleRate;
  }

  const Subfields& subfields() const {
    return subfields_;
  }

  const ColumnHandles& partitionKeys() const {
    return partitionKeys_;
  }

  const ColumnHandles& infoColumns() const {
    return infoColumns_;
  }

  const SpecialColumnNames& specialColumns() const {
    return specialColumns_;
  }

  const auto& multiReferencedFields() const {
    return multiReferencedFields_;
  }

  const auto& remainingFilterColumns() const {
    return remainingFilterColumns_;
  }

 private:
  void processColumnHandle(const FileColumnHandlePtr& handle);

  void configureExtractionColumns(
      FileScanState& state,
      memory::MemoryPool* pool) const;

  const RowTypePtr outputType_;
  const FileTableHandlePtr tableHandle_;
  const ColumnHandleMap assignments_;
  const common::SubfieldFilters filters_;
  const core::TypedExprPtr remainingFilter_;
  const Options options_;
  RowTypePtr readerOutputType_;
  // Preserve the initial tree / metadata construction before extraction
  // rebuilds.
  RowTypePtr preExtractionType_;
  ColumnHandles partitionKeys_;
  ColumnHandles infoColumns_;
  SpecialColumnNames specialColumns_{};
  Subfields subfields_;
  std::vector<common::Subfield> remainingFilterSubfields_;
  folly::F14FastMap<column_index_t, const FileColumnHandle*> extractionColumns_;
  std::vector<column_index_t> multiReferencedFields_;
  folly::F14FastSet<std::string> remainingFilterColumns_;
};

} // namespace facebook::velox::connector
