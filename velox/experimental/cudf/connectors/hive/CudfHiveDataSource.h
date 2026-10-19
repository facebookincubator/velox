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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/base/RandomUtil.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/type/Type.h"

#include <cudf/ast/expressions.hpp>

#include <mutex>
#include <string_view>
#include <unordered_set>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

class CudfHiveDataSource : public DataSource, public NvtxHelper {
 public:
  /// DWIO bytes read, preserved separately from ReadFile bytes.
  static constexpr std::string_view kDwioStorageReadBytes{
      "dwio.storageReadBytes"};

  CudfHiveDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      facebook::velox::FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& CudfHiveConfig);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void setFromDataSource(std::unique_ptr<DataSource> source) override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<facebook::velox::common::Filter>& /*filter*/)
      override {
    VELOX_NYI(
        "Dynamic filters not yet implemented by cudf::CudfHiveConnector.");
  }

  std::optional<RowVectorPtr> next(
      uint64_t size,
      velox::ContinueFuture& /* future */) override;

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  const common::SubfieldFilters* getFilters() const override {
    return &subfieldFilters_;
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

 protected:
  // Indicates the component that turns the partition keys of a split into
  // columns.
  enum class PartitionKeySource {
    // Partition key columns are left out of the split reader's projection:
    // pushed-down filters on them are tested once per split against the
    // split's partition value, and the columns are materialized as constants
    // after the read.
    kDataSource,
    // Partition key columns stay in the split reader's projection, for table
    // formats whose readers resolve them per data file.
    kSplitReader,
  };

  // Same as the public constructor, with 'partitionKeySource' choosing which
  // component resolves partition keys.
  CudfHiveDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      facebook::velox::FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
      PartitionKeySource partitionKeySource);

  // Virtual method to create a `CudfSplitReader` or subclass for the data
  // source.
  virtual std::unique_ptr<CudfSplitReader> createCudfSplitReader();

  // Virtual method to convert the input `ConnectorSplit` to appropriate
  // subclass(es).
  virtual void convertSplit(std::shared_ptr<ConnectorSplit> split);

  std::shared_ptr<CudfHiveConnectorSplit> split_;
  std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
      tableHandle_;

  const std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  facebook::velox::FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  const ConnectorQueryCtx* const connectorQueryCtx_;

  // Carries the session settings the remaining filter is folded and compiled
  // against. Kept for the data source's lifetime: createCudfExpression borrows
  // its config, and an evaluator may hold on to it.
  std::shared_ptr<core::QueryCtx> optimizeQueryCtx_;

  // Columns to read.
  std::vector<std::string> readColumnNames_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<velox::IoStats> ioStats_;

  // The row type for the data source output, not including filter-only columns.
  const RowTypePtr outputType_;

  // Cached combined AST filter expression compiled from 'subfieldFilters_',
  // owned by 'subfieldTree_'.
  const cudf::ast::expression* subfieldFilterAst_{nullptr};
  // Whether any subfield filter targets a decimal column. If so, addSplit()
  // rebuilds the pushdown AST against each split's Parquet decimal types.
  bool hasDecimalSubfieldFilter_{false};

 private:
  // A partition key the data source resolves from the split.
  struct PartitionKey {
    TypePtr type;
    // Whether a DATE value is a count of days since the epoch rather than an
    // ISO date.
    bool dateValueDaysSinceEpoch;
  };

  // Construct and cache a RowTypePtr for the table column names and types.
  const RowTypePtr getTableRowType();

  // Returns the type of a column read from the file or materialized by the
  // data source, by table column name.
  const TypePtr& columnType(const std::string& name) const;

  // Returns the current split's value of a partition key. A key the split
  // does not carry reads as NULL.
  std::optional<std::string> partitionKeyValue(const std::string& name) const;

  // Returns whether every pushed-down partition key filter accepts the current
  // split.
  bool partitionKeyFiltersAcceptSplit() const;

  // Returns the current split's partition key values as scalars, in
  // 'partitionKeyNames_' order, each parsed per its key's type.
  std::vector<std::unique_ptr<cudf::scalar>> makePartitionKeyScalars(
      cuda::stream_ref stream) const;

  // Returns the materialized partition key columns of the current split, each
  // a constant column of 'numRows' rows built from 'partitionKeyScalars_'.
  std::vector<std::unique_ptr<cudf::column>> makePartitionKeyColumns(
      cudf::size_type numRows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  // Returns whether TIMESTAMP partition values are read as local time.
  bool readTimestampPartitionValueAsLocalTime() const;

  // Counts the current split as skipped without opening it.
  void recordSkippedSplit();

  RowTypePtr cachedTableRowType_{};

  memory::MemoryPool* const pool_;

  size_t completedRows_{0};
  size_t completedBytes_{0};

  dwio::common::RuntimeStats runtimeStats_;

  std::unique_ptr<CudfSplitReader> cudfSplitReader_;

  // Optimized remaining-filter expression, or null when there is no remaining
  // filter or it folded to a constant. Gates remaining-filter evaluation in
  // next().
  core::TypedExprPtr optimizedRemainingFilter_;

  // Whether the remaining filter folded to a constant that no row satisfies,
  // in which case splits are skipped without opening them.
  bool remainingFilterRejectsAllRows_{false};

  // Compiled cuDF evaluator for the remaining filter, applied post-read in
  // next(). Null when there is no remaining filter.
  std::shared_ptr<velox::cudf_velox::CudfExpression>
      cudfRemainingFilterExpression_;

  std::atomic<uint64_t> totalRemainingFilterTime_{0};

  std::unordered_set<std::string> readColumnSet_;

  // Expression evaluator for remaining filter.
  core::ExpressionEvaluator* const expressionEvaluator_;

  // Logical and split-specific physical AST storage for subfield filters.
  std::vector<std::unique_ptr<cudf::scalar>> subfieldScalars_;
  cudf::ast::tree subfieldTree_;
  std::vector<std::unique_ptr<cudf::scalar>> pushdownFilterScalars_;
  cudf::ast::tree pushdownFilterTree_;

  // The table handle's subfield filters, merged with the ones extracted from
  // its remaining filter, except the ones on partition keys the data source
  // resolves, which are kept in 'partitionKeyFilters_'.
  common::SubfieldFilters subfieldFilters_;

  const PartitionKeySource partitionKeySource_;

  // Table column name of each output column, in 'outputType_' order.
  std::vector<std::string> outputSourceNames_;

  // Types of the output and materialized columns, by table column name.
  std::unordered_map<std::string, TypePtr> columnTypes_;

  // Row type of what the split reader returns: 'readColumnNames_' with their
  // types.
  RowTypePtr readerOutputType_;

  // Partition keys the data source resolves from the split, by table column
  // name.
  std::unordered_map<std::string, PartitionKey> partitionKeys_;

  // The partition keys materialized as columns, those of the output and of
  // the remaining filter, in the order they follow the read columns in the
  // table the remaining filter sees.
  std::vector<std::string> partitionKeyNames_;

  // Pushed-down filters on partition keys the data source resolves. Each
  // decides for the whole split, so none is compiled into the reader's
  // filter.
  common::SubfieldFilters partitionKeyFilters_;

  // The current split's partition key values as scalars, in
  // 'partitionKeyNames_' order. Built once per split in addSplit(); next()
  // expands each into a constant column per batch.
  std::vector<std::unique_ptr<cudf::scalar>> partitionKeyScalars_;

  // Whether a partition key filter rejected the current split, which then has
  // no split reader.
  bool splitSkipped_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
