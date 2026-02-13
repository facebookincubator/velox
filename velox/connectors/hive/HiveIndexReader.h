/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Reader.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::connector {
class ConnectorQueryCtx;
} // namespace facebook::velox::connector

namespace facebook::velox::connector::hive {

struct HiveConnectorSplit;
class HiveTableHandle;
class HiveColumnHandle;
class HiveConfig;

/// HiveIndexReader is similar to SplitReader but supports index lookup API.
/// It takes a request row vector, converts each row into filters inserted into
/// the scan spec, and reads matching data from the underlying reader. The
/// result includes matched rows and a buffer containing the count of matches
/// for each input request row.
///
/// The HiveIndexReader is designed to be reusable across multiple lookup
/// requests on the same split.
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
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor);

  virtual ~HiveIndexReader() = default;

  using Request = IndexSource::Request;
  using Result = IndexSource::Result;

  /// Sets the input for index lookup. Each row in 'input' will be converted
  /// to filters on key columns and used to query matching rows from the
  /// underlying data.
  ///
  /// @param request The lookup request containing input row vector with lookup
  /// keys.
  void setRequest(const Request& request);

  /// Returns true if there are more input rows to process.
  bool hasNext() const;

  /// Reads the next batch of matching rows for the current input rows.
  /// The result from a single request row is never split across multiple
  /// calls to next().
  ///
  /// @param maxOutputRows Maximum number of output rows to return.
  /// @return Result containing matched rows and input hit indices, or nullptr
  /// if no more results.
  std::unique_ptr<Result> next(vector_size_t maxOutputRows);

  std::string toString() const;

 private:
  // Resets filter caches for reuse.
  void resetFilterCaches();

  // Creates the file reader for reading file metadata and schema.
  // NOTE: Called from constructor initializer list, so only accesses members
  // declared before fileReader_.
  std::unique_ptr<dwio::common::Reader> createFileReader();

  // Creates the row reader.
  // NOTE: Called from constructor initializer list, so only accesses members
  // declared before rowReader_.
  std::unique_ptr<dwio::common::RowReader> createRowReader();

  // Initializes joinIndexColumnSpecs_ and requestColumnIndices_ from join
  // conditions.
  void initJoinConditions();

  // Converts the input row at the given index to filters on key columns
  // and applies them to the scan spec.
  void applyFiltersFromRequest(vector_size_t row);

  // Clears filters applied to key columns in the scan spec.
  void clearKeyFilters();

  // Reads the next batch of rows based on the current filters.
  // Returns the number of rows read. The output vector is passed by reference
  // and will be populated with the matching rows.
  uint64_t readNext(VectorPtr& output);

  // Resets request_ and requestRow_.
  void reset();

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
  const std::unique_ptr<dwio::common::RowReader> rowReader_;
  // Join conditions (including equal conditions converted from join keys).
  const std::vector<core::IndexLookupConditionPtr> joinConditions_;

  // Cached ScanSpec children for index columns used in join conditions.
  std::vector<common::ScanSpec*> joinIndexColumnSpecs_;
  // Request column indices for each join condition (for probe side columns).
  // For EqualIndexLookupCondition, stores {valueIndex}.
  // For BetweenIndexLookupCondition, stores {lowerIndex, upperIndex}.
  std::vector<std::vector<column_index_t>> requestColumnIndices_;

  // Current request for lookup.
  RowVectorPtr request_;
  // Current row index in the request being processed.
  vector_size_t requestRow_{0};

  // Decoded vectors for input columns used in join conditions.
  // Indexed by join condition index and then by column index within that
  // condition (0 for equal condition value, 0/1 for between condition
  // lower/upper).
  std::vector<std::vector<DecodedVector>> decodedRequestVectors_;

  // For BetweenIndexLookupCondition with constant bounds, stores the constant
  // values directly. The outer vector is indexed by join condition index. The
  // inner vector has size 2 for between conditions (lower, upper). If a bound
  // is a constant, the corresponding optional contains the value; otherwise
  // it's std::nullopt and the value should be decoded from request.
  std::vector<std::vector<std::optional<variant>>> constantBoundValues_;
};

} // namespace facebook::velox::connector::hive
