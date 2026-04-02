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

#include "velox/experimental/cudf/benchmarks/PreloadedScanOperator.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/TableScan.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"

#include <cudf/copying.hpp>

namespace facebook::velox::cudf_velox {

// --- PreloadedTableStore ---

PreloadedTableStore& PreloadedTableStore::getInstance() {
  static PreloadedTableStore instance;
  return instance;
}

void PreloadedTableStore::store(
    const std::string& tableName,
    std::vector<RowVectorPtr> batches) {
  tables_[tableName] = std::move(batches);
}

const std::vector<RowVectorPtr>* PreloadedTableStore::getBatches(
    const std::string& tableName) const {
  auto it = tables_.find(tableName);
  return it != tables_.end() ? &it->second : nullptr;
}

bool PreloadedTableStore::isGpuData() const {
  for (const auto& [name, batches] : tables_) {
    if (!batches.empty()) {
      return std::dynamic_pointer_cast<CudfVector>(batches.front()) != nullptr;
    }
  }
  return false;
}

void PreloadedTableStore::clear() {
  tables_.clear();
}

// --- PreloadedScanOperator ---

PreloadedScanOperator::PreloadedScanOperator(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::TableScanNode>& scanNode)
    : SourceOperator(
          ctx,
          scanNode->outputType(),
          operatorId,
          scanNode->id(),
          "PreloadedScan"),
      scanOutputType_(scanNode->outputType()),
      tableName_(
          dynamic_cast<const velox::connector::hive::HiveTableHandle*>(
              scanNode->tableHandle().get())
              ->tableName()) {
  auto& store = PreloadedTableStore::getInstance();
  batches_ = store.getBatches(tableName_);
  VELOX_CHECK_NOT_NULL(
      batches_,
      "Preloaded data not found for table: {}",
      tableName_);
  VELOX_CHECK(!batches_->empty(), "Preloaded batches empty for: {}", tableName_);

  const auto& storeType = batches_->front()->type()->asRow();
  for (auto i = 0; i < scanOutputType_->size(); ++i) {
    const auto& name = scanOutputType_->nameOf(i);
    auto idx = storeType.getChildIdxIfExists(name);
    VELOX_CHECK(
        idx.has_value(),
        "Column '{}' not found in preloaded table '{}'",
        name,
        tableName_);
    columnMapping_.push_back(idx.value());
  }
}

RowVectorPtr PreloadedScanOperator::selectColumns(
    const RowVectorPtr& batch) const {
  if (auto cudfVec = std::dynamic_pointer_cast<CudfVector>(batch)) {
    auto tableView = cudfVec->getTableView();
    std::vector<cudf::column_view> selectedCols;
    selectedCols.reserve(columnMapping_.size());
    for (auto idx : columnMapping_) {
      selectedCols.push_back(tableView.column(idx));
    }
    auto selectedView = cudf::table_view(selectedCols);
    auto selectedTable =
        std::make_unique<cudf::table>(selectedView, cudfVec->stream());
    return std::make_shared<CudfVector>(
        pool(),
        scanOutputType_,
        batch->size(),
        std::move(selectedTable),
        cudfVec->stream());
  }

  std::vector<VectorPtr> children;
  children.reserve(columnMapping_.size());
  for (auto idx : columnMapping_) {
    children.push_back(batch->childAt(idx));
  }
  return std::make_shared<RowVector>(
      pool(), scanOutputType_, nullptr, batch->size(), std::move(children));
}

RowVectorPtr PreloadedScanOperator::getOutput() {
  if (finished_) {
    return nullptr;
  }
  if (currentBatch_ >= batches_->size()) {
    finished_ = true;
    return nullptr;
  }
  return selectColumns((*batches_)[currentBatch_++]);
}

// --- PreloadedTableScanAdapter ---

namespace {

class PreloadedTableScanAdapter : public OperatorAdapter {
 public:
  PreloadedTableScanAdapter() : OperatorAdapter("TableScan") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::TableScan*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    if (PreloadedTableStore::getInstance().isPopulated()) {
      return std::dynamic_pointer_cast<const core::TableScanNode>(planNode) !=
          nullptr;
    }
    return canRunOnGPUFallback(planNode);
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    auto& store = PreloadedTableStore::getInstance();
    if (!store.isPopulated()) {
      return true;
    }
    return store.isGpuData();
  }

  bool keepOperator() const override {
    return !PreloadedTableStore::getInstance().isPopulated();
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto scanNode =
        std::dynamic_pointer_cast<const core::TableScanNode>(planNode);
    VELOX_CHECK_NOT_NULL(scanNode);
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<PreloadedScanOperator>(operatorId, ctx, scanNode));
    return result;
  }

 private:
  static bool canRunOnGPUFallback(const core::PlanNodePtr& planNode) {
    auto tableScanNode =
        std::dynamic_pointer_cast<const core::TableScanNode>(planNode);
    if (!tableScanNode) {
      return false;
    }
    auto const& conn = velox::connector::getConnector(
        tableScanNode->tableHandle()->connectorId());
    return std::dynamic_pointer_cast<
               cudf_velox::connector::hive::CudfHiveConnector>(conn) !=
        nullptr;
  }
};

} // namespace

void registerPreloadedTableScanAdapter() {
  OperatorAdapterRegistry::getInstance().registerAdapter(
      std::make_unique<PreloadedTableScanAdapter>(), /*overwrite=*/true);
}

} // namespace facebook::velox::cudf_velox
