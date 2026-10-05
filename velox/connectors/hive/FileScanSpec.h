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
/// between concurrently active readers, even when they share a FileScanSpec.
struct FileScanState {
  common::SubfieldFilters filters;
  std::shared_ptr<common::ScanSpec> scanSpec;
  std::shared_ptr<common::MetadataFilter> metadataFilter;
  RowTypePtr readerProducedType;
};

/// Immutable connector-level description of the columns and predicates for a
/// logical scan. Multiple data sources or physical readers with the same scan
/// requirements can share a std::shared_ptr<const FileScanSpec>.
///
/// Resolves output assignments to physical column names and types, including
/// partition and synthesized columns, required subfields, extraction chains,
/// and post-read column processing. Preserves the original predicates and
/// derives the physical filters, remaining expression, and column demands
/// according to FileScanOptions. Columns needed only by the remaining
/// expression are included in the reader's input requirements.
///
/// The DWIO common::ScanSpec is a mutable tree describing how one physical
/// reader accesses columns. newFileScanState() creates a fresh tree, cloned
/// filters, extraction configuration, and an optional MetadataFilter for each
/// reader. The overload accepting physical column demands supports additions
/// such as Hive bucket conversion columns. File-specific constants, filter
/// adaptation, and other reader mutations remain local to that FileScanState.
///
/// Owns the handles, typed expressions, and subfields backing its derived
/// column demands. Readers borrowing these inputs must retain this object's
/// lifetime. The query context and expression evaluator are used during
/// preparation; this object retains no query context, compiled ExprSet, vector,
/// reader, or mutable common::ScanSpec. Each reader obtains its execution state
/// using its own context and memory pool.
///
/// File enumeration, split scheduling, and format-specific I/O are handled by
/// the connector and reader layers that consume this specification.
class FileScanSpec {
 public:
  using Subfields =
      folly::F14FastMap<std::string, std::vector<const common::Subfield*>>;
  using ColumnHandles = std::unordered_map<std::string, FileColumnHandlePtr>;

  FileScanSpec(
      const RowTypePtr& outputType,
      const FileTableHandlePtr& tableHandle,
      const ColumnHandleMap& assignments,
      const ConnectorQueryCtx* context,
      const std::shared_ptr<FileConfig>& config,
      FileScanOptions options = {});

  ~FileScanSpec() = default;

  FileScanSpec(const FileScanSpec&) = delete;
  FileScanSpec& operator=(const FileScanSpec&) = delete;
  FileScanSpec(FileScanSpec&&) = delete;
  FileScanSpec& operator=(FileScanSpec&&) = delete;

  FileScanState newFileScanState(const ConnectorQueryCtx* context) const;

  /// Build a state with connector-specific physical column demands, e.g.
  /// Hive bucket conversion columns. Does not modify this specification.
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
