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
#include "velox/experimental/cudf/connectors/hive/CudfIntegerMembership.h"
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
#include <cudf/column/column.hpp>

#include <mutex>
#include <string_view>
#include <unordered_map>
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
      column_index_t outputChannel,
      const std::shared_ptr<facebook::velox::common::Filter>& filter) override;

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

  // Columns to read.
  std::vector<std::string> readColumnNames_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<velox::IoStats> ioStats_;

  // The row type for the data source output, not including filter-only columns.
  const RowTypePtr outputType_;

  // Cached combined AST filter expression compiled from 'subfieldFilters_',
  // owned by 'subfieldTree_'.
  const cudf::ast::expression* subfieldFilterAst_{nullptr};

 private:
  // Construct and cache a RowTypePtr for the table column names and types.
  const RowTypePtr getTableRowType();
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

  // Expression evaluator for subfield filter.
  std::vector<std::unique_ptr<cudf::scalar>> subfieldScalars_;
  cudf::ast::tree subfieldTree_;

  // The table handle's subfield filters, merged with the ones extracted from
  // its remaining filter.
  common::SubfieldFilters subfieldFilters_;

  // Holds an exact integer filter and its lazily copied device values.
  struct DynamicIntegerFilter {
    // Sorted host values for reader bounds and device representation.
    std::vector<int64_t> values;

    // Preserves SQL filter semantics for null probe values.
    bool nullAllowed;

    // Avoids repeated uploads; stores input-width values or UINT32 bitmap
    // words.
    std::unique_ptr<cudf::column> deviceValues;

    // The CPU bitmask filter's exact representation, copied once to
    // deviceValues.
    std::vector<uint32_t> bitmap{};
    int64_t bitmapMinimum{0};

    // Sparse filters retain one hash table across decoded batches.
    std::unique_ptr<CudfIntegerHashSet> hashSet{};
  };

  // Tracks columns whose storage types must match accepted dynamic filters.
  std::unordered_set<column_index_t> dynamicFilterChannels_;

  // Keeps ranges and other merged filters by physical subfield.
  common::SubfieldFilters dynamicFilters_;

  // Keeps exact integer filters indexed by their output channel.
  std::unordered_map<column_index_t, DynamicIntegerFilter>
      dynamicIntegerFilters_;

  // Owns all AST nodes referenced by 'dynamicFilterExpr_'.
  std::unique_ptr<cudf::ast::tree> dynamicFilterTree_;

  // Owns all scalars referenced by 'dynamicFilterExpr_'.
  std::vector<std::unique_ptr<cudf::scalar>> dynamicFilterScalars_;

  // Points to the AST used for filter kinds without a direct integer path.
  const cudf::ast::expression* dynamicFilterExpr_{nullptr};

  // No dynamic filter has changed since this data source created its reader.
  bool readerFiltersCurrent_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
