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
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/Reader.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::velox::connector {
class ConnectorQueryCtx;
} // namespace facebook::velox::connector

namespace facebook::velox::connector::hive {

struct HiveConnectorSplit;
class HiveTableHandle;
class HiveColumnHandle;
class HiveConfig;

/// HiveIndexReader handles index lookups for Hive tables with cluster indexes.
/// It focuses on:
/// - Creating index bounds from join conditions
/// - Delegating actual index lookups to the format-specific IndexReader
///
/// The format-specific IndexReader (e.g., SelectiveNimbleIndexReader) handles:
/// - Encoding keys into format-specific representations
/// - Stripe iteration and row range computation
/// - Data reading and output assembly
class HiveIndexReader {
 public:
  HiveIndexReader(
      const std::vector<std::shared_ptr<const HiveConnectorSplit>>& hiveSplits,
      const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const std::vector<core::IndexLookupConditionPtr>& joinConditions,
      const RowTypePtr& requestType,
      const RowTypePtr& outputType,
      const std::shared_ptr<io::IoStatistics>& ioStats,
      const std::shared_ptr<IoStats>& fsStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor);

  virtual ~HiveIndexReader() = default;

  using Request = IndexSource::Request;
  using Result = IndexSource::Result;

  /// Sets the input for index lookup. Each row in 'input' will be converted
  /// to index bounds and passed to the format-specific IndexReader.
  ///
  /// @param request The lookup request containing input row vector with lookup
  /// keys.
  void startLookup(const Request& request);

  /// Returns true if there are more results to fetch from the current lookup.
  bool hasNext() const;

  /// Returns the next batch of matching rows for the current input rows.
  /// The result from a single request row is never split across multiple
  /// calls to next().
  ///
  /// @param maxOutputRows Maximum number of output rows to return.
  /// @return Result containing matched rows and input hit indices, or nullptr
  /// if no more results.
  std::unique_ptr<Result> next(vector_size_t maxOutputRows);

  std::string toString() const;

 private:
  // Creates the file reader for reading file metadata and schema.
  std::unique_ptr<dwio::common::Reader> createFileReader();

  // Creates the format-specific index reader.
  std::unique_ptr<dwio::common::IndexReader> createIndexReader();

  // Parses join conditions to extract column indices and constant values.
  void parseJoinConditions();

  // Builds IndexBounds from the request row vector.
  serializer::IndexBounds buildRequestIndexBounds(const RowVectorPtr& request);

  std::shared_ptr<const HiveConnectorSplit> hiveSplit_;
  const std::shared_ptr<const HiveTableHandle> tableHandle_;
  const ConnectorQueryCtx* connectorQueryCtx_;
  const std::shared_ptr<const HiveConfig> hiveConfig_;
  FileHandleFactory* const fileHandleFactory_;
  const RowTypePtr requestType_;
  const RowTypePtr outputType_;

  const std::shared_ptr<io::IoStatistics> ioStatistics_;
  const std::shared_ptr<IoStats> ioStats_;
  folly::Executor* const ioExecutor_;
  memory::MemoryPool* const pool_;

  const std::shared_ptr<common::ScanSpec> scanSpec_;
  const std::unique_ptr<dwio::common::Reader> fileReader_;
  const std::unique_ptr<dwio::common::IndexReader> indexReader_;
  // Join conditions (including equal conditions converted from join keys).
  const std::vector<core::IndexLookupConditionPtr> joinConditions_;

  // Request column indices for each join condition (for probe side columns).
  // For EqualIndexLookupCondition, stores {valueIndex}.
  // For BetweenIndexLookupCondition, stores {lowerIndex, upperIndex}.
  std::vector<std::vector<column_index_t>> requestColumnIndices_;

  // For BetweenIndexLookupCondition with constant bounds, stores the constant
  // values directly. The outer vector is indexed by join condition index. The
  // inner vector has size 2 for between conditions (lower, upper). If a bound
  // is a constant, the corresponding optional contains the value; otherwise
  // it's std::nullopt and the value should be decoded from request.
  std::vector<std::vector<std::optional<variant>>> constantBoundValues_;

  // Cached row type for index bounds (column names and types from join
  // conditions).
  RowTypePtr indexBoundType_;

  // Reusable column vectors for building index bounds.
  std::vector<VectorPtr> lowerBoundColumns_;
  std::vector<VectorPtr> upperBoundColumns_;
};

} // namespace facebook::velox::connector::hive
