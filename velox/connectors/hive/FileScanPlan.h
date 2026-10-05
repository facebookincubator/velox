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

#include "velox/connectors/hive/FileTableHandle.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"

namespace facebook::velox::connector::hive {

class FileConfig;

struct FileScanOptions {
  // When false, keep the entire remaining expression for post-read evaluation.
  // Explicit table-handle subfield filters still apply to physical reads.
  bool extractRemainingFilter{true};
};

/// Mutable state for exactly one physical reader. Never share this state
/// between concurrently active readers, even when they share a FileScanPlan.
struct FileScanState {
  common::SubfieldFilters filters;
  std::shared_ptr<common::ScanSpec> scanSpec;
  std::shared_ptr<common::MetadataFilter> metadataFilter;
  RowTypePtr readerProducedType;
};

/// Immutable column and predicate preparation shared by physical readers.
/// Owns the handles and subfields referenced by the derived column demands.
/// It owns no vectors, compiled ExprSets, query context or mutable ScanSpec.
class FileScanPlan {
 public:
  using Subfields =
      folly::F14FastMap<std::string, std::vector<const common::Subfield*>>;
  using ColumnHandles = std::unordered_map<std::string, FileColumnHandlePtr>;

  FileScanPlan(
      const RowTypePtr& outputType,
      const FileTableHandlePtr& tableHandle,
      const ColumnHandleMap& assignments,
      const ConnectorQueryCtx* context,
      const std::shared_ptr<FileConfig>& config,
      FileScanOptions options = {});

  ~FileScanPlan() = default;

  FileScanPlan(const FileScanPlan&) = delete;
  FileScanPlan& operator=(const FileScanPlan&) = delete;
  FileScanPlan(FileScanPlan&&) = delete;
  FileScanPlan& operator=(FileScanPlan&&) = delete;

  FileScanState newFileScanState(const ConnectorQueryCtx* context) const;

  /// Build a state with connector-specific physical column demands, e.g.
  /// Hive bucket conversion columns. Does not modify the logical plan.
  FileScanState newFileScanState(
      const RowTypePtr& readerOutputType,
      const Subfields& subfields,
      const common::SubfieldFilters& filters,
      const ConnectorQueryCtx* context) const;

  const RowTypePtr& outputType() const {
    return outputType_;
  }
  const RowTypePtr& readerOutputType() const {
    return readerOutputType_;
  }
  const FileTableHandlePtr& tableHandle() const {
    return tableHandle_;
  }
  const ColumnHandleMap& assignments() const {
    return assignments_;
  }
  const core::TypedExprPtr& originalRemainingFilter() const {
    return tableHandle_->remainingFilter();
  }
  const core::TypedExprPtr& remainingFilter() const {
    return remainingFilter_;
  }
  common::SubfieldFilters originalFilters() const;
  common::SubfieldFilters filters() const;
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
  const auto& extractionColumns() const {
    return extractionColumns_;
  }
  const auto& columnPostProcessors() const {
    return columnPostProcessors_;
  }
  const auto& multiReferencedFields() const {
    return multiReferencedFields_;
  }
  const auto& remainingFilterColumns() const {
    return remainingFilterColumns_;
  }
  double sampleRate() const {
    return sampleRate_;
  }

 private:
  void processColumnHandle(const FileColumnHandlePtr& handle);
  RowTypePtr configureExtractionColumns(
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const RowTypePtr& readerOutputType,
      memory::MemoryPool* pool) const;

  const RowTypePtr outputType_;
  const FileTableHandlePtr tableHandle_;
  const ColumnHandleMap assignments_;
  const bool disableStatsBasedFilterReorder_;
  RowTypePtr readerOutputType_;
  common::SubfieldFilters originalFilters_;
  common::SubfieldFilters filters_;
  core::TypedExprPtr remainingFilter_;
  double sampleRate_;
  ColumnHandles partitionKeys_;
  ColumnHandles infoColumns_;
  SpecialColumnNames specialColumns_{};
  Subfields subfields_;
  std::vector<common::Subfield> remainingFilterSubfields_;
  folly::F14FastMap<column_index_t, const FileColumnHandle*> extractionColumns_;
  std::vector<std::function<void(VectorPtr&)>> columnPostProcessors_;
  std::vector<column_index_t> multiReferencedFields_;
  folly::F14FastSet<std::string> remainingFilterColumns_;
};

} // namespace facebook::velox::connector::hive
