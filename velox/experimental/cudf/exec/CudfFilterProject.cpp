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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/memory/Memory.h"
#include "velox/expression/Expr.h"
#include "velox/expression/FieldReference.h"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <iostream>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

namespace {

void debugPrintTree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    int indent = 0,
    std::ostream& os = std::cout) {
  if (indent == 0)
    os << "=== Expression Tree ===" << std::endl;
  os << std::string(indent, ' ') << expr->name() << "("
     << expr->type()->toString() << ")" << std::endl;
  for (auto& input : expr->inputs()) {
    debugPrintTree(input, indent + 2, os);
  }
}

bool checkAddIdentityProjection(
    const core::TypedExprPtr& projection,
    const RowTypePtr& inputType,
    column_index_t outputChannel,
    std::vector<exec::IdentityProjection>& identityProjections) {
  if (auto field = core::TypedExprs::asFieldAccess(projection)) {
    const auto& inputs = field->inputs();
    if (inputs.empty() ||
        (inputs.size() == 1 &&
         dynamic_cast<const core::InputTypedExpr*>(inputs[0].get()))) {
      const auto inputChannel = inputType->getChildIdx(field->name());
      identityProjections.emplace_back(inputChannel, outputChannel);
      return true;
    }
  }

  return false;
}

// Split stats to attrbitute cardinality reduction to the Filter node.
std::vector<exec::OperatorStats> splitStats(
    const exec::OperatorStats& combinedStats,
    const core::PlanNodeId& filterNodeId) {
  exec::OperatorStats filterStats;

  filterStats.operatorId = combinedStats.operatorId;
  filterStats.pipelineId = combinedStats.pipelineId;
  filterStats.planNodeId = filterNodeId;
  filterStats.operatorType = combinedStats.operatorType;
  filterStats.numDrivers = combinedStats.numDrivers;

  filterStats.inputBytes = combinedStats.inputBytes;
  filterStats.inputPositions = combinedStats.inputPositions;
  filterStats.inputVectors = combinedStats.inputVectors;

  // Estimate Filter's output bytes based on cardinality change.
  const double filterRate = combinedStats.inputPositions > 0
      ? (combinedStats.outputPositions * 1.0 / combinedStats.inputPositions)
      : 1.0;

  filterStats.outputBytes = (uint64_t)(filterStats.inputBytes * filterRate);
  filterStats.outputPositions = combinedStats.outputPositions;
  filterStats.outputVectors = combinedStats.outputVectors;

  auto projectStats = combinedStats;
  projectStats.inputBytes = filterStats.outputBytes;
  projectStats.inputPositions = filterStats.outputPositions;
  projectStats.inputVectors = filterStats.outputVectors;

  return {std::move(projectStats), std::move(filterStats)};
}

} // namespace

bool canBeEvaluatedByCudf(
    const std::vector<core::TypedExprPtr>& exprs,
    core::QueryCtx* queryCtx) {
  if (exprs.empty()) {
    return true;
  }

  auto precompilePool =
      memory::memoryManager()->addLeafPool("", /*threadSafe*/ false);
  core::ExecCtx precompileCtx(precompilePool.get(), queryCtx);

  bool lazyDereference = false;
  std::vector<core::TypedExprPtr> exprsCopy = exprs;
  std::unique_ptr<exec::ExprSet> exprSet = exec::makeExprSetFromFlag(
      std::move(exprsCopy), &precompileCtx, lazyDereference);

  for (const auto& e : exprSet->exprs()) {
    if (!canBeEvaluatedByCudf(e)) {
      return false;
    }
  }
  return true;
}

CudfFilterProject::CudfFilterProject(
    int32_t operatorId,
    velox::exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project)
    : Operator(
          driverCtx,
          project ? project->outputType() : filter->outputType(),
          operatorId,
          project ? project->id() : filter->id(),
          "CudfFilterProject"),
      NvtxHelper(
          nvtx3::rgb{220, 20, 60}, // Crimson
          operatorId,
          fmt::format("[{}]", project ? project->id() : filter->id())),
      hasFilter_(filter != nullptr),
      project_(project),
      filter_(filter) {
  if (filter_ != nullptr && project_ != nullptr) {
    folly::Synchronized<exec::OperatorStats>& opStats = Operator::stats();
    opStats.withWLock([&](auto& stats) {
      stats.setStatSplitter(
          [filterId = filter_->id()](const auto& combinedStats) {
            return splitStats(combinedStats, filterId);
          });
    });
  }
}

void CudfFilterProject::initialize() {
  Operator::initialize();

  std::vector<core::TypedExprPtr> allExprs;
  if (hasFilter_) {
    VELOX_CHECK_NOT_NULL(filter_);
    allExprs.push_back(filter_->filter());
  }

  if (project_) {
    const auto& inputType = project_->sources()[0]->outputType();

    for (column_index_t i = 0; i < project_->projections().size(); i++) {
      auto& projection = project_->projections()[i];
      bool identityProjection = checkAddIdentityProjection(
          projection, inputType, i, identityProjections_);
      if (!identityProjection) {
        allExprs.push_back(projection);
        resultProjections_.emplace_back(allExprs.size() - 1, i);
      }
    }
  } else {
    for (column_index_t i = 0; i < outputType_->size(); ++i) {
      identityProjections_.emplace_back(i, i);
    }
    isIdentityProjection_ = true;
  }

  auto lazyDereference =
      (dynamic_cast<const core::LazyDereferenceNode*>(project_.get()) !=
       nullptr);
  VELOX_CHECK(!(lazyDereference && filter_));
  auto expr = exec::makeExprSetFromFlag(
      std::move(allExprs), operatorCtx_->execCtx(), lazyDereference);

  const auto inputType = project_ ? project_->sources()[0]->outputType()
                                  : filter_->sources()[0]->outputType();

  // convert to AST
  if (CudfConfig::getInstance().debugEnabled) {
    int i = 0;
    for (const auto& expr : expr->exprs()) {
      LOG(INFO) << "expr[" << i++ << "] " << expr->toString();
      debugPrintTree(expr, 0, LOG(INFO));
    }
  }
  if (hasFilter_) {
    // First expr is Filter, rest are Project
    filterEvaluator_ = createCudfExpression(expr->exprs()[0], inputType);
    std::transform(
        expr->exprs().begin() + 1,
        expr->exprs().end(),
        std::back_inserter(projectEvaluators_),
        [inputType](const auto& expr) {
          return createCudfExpression(expr, inputType);
        });
  } else {
    std::transform(
        expr->exprs().begin(),
        expr->exprs().end(),
        std::back_inserter(projectEvaluators_),
        [inputType](const auto& expr) {
          return createCudfExpression(expr, inputType);
        });
  }

  filter_.reset();
  project_.reset();
}

void CudfFilterProject::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr CudfFilterProject::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (allInputProcessed()) {
    return nullptr;
  }
  if (input_->size() == 0) {
    input_.reset();
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  auto inputTableColumns = cudfInput->release()->release();

  if (hasFilter_) {
    filter(inputTableColumns, stream);
  }
  vector_size_t filteredRowCount = 0;
  if (!inputTableColumns.empty()) {
    filteredRowCount = inputTableColumns.front()->size();
  } else {
    filteredRowCount = input_->size();
  }
  auto outputColumns = project(inputTableColumns, stream);

  if (outputColumns.empty()) {
    if (filteredRowCount == 0) {
      input_.reset();
      return nullptr;
    }
    auto outputTable = std::make_unique<cudf::table>(std::move(outputColumns));
    stream.synchronize();
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(1) << "cudfProject Output: " << filteredRowCount
              << " rows, 0 columns";
    }
    auto cudfOutput = std::make_shared<CudfVector>(
        input_->pool(),
        outputType_,
        filteredRowCount,
        std::move(outputTable),
        stream);
    input_.reset();
    return cudfOutput;
  }

  auto outputTable = std::make_unique<cudf::table>(std::move(outputColumns));
  stream.synchronize();
  auto const numColumns = outputTable->num_columns();
  auto const size = outputTable->num_rows();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "cudfProject Output: " << size << " rows, " << numColumns
            << " columns";
  }

  auto cudfOutput = std::make_shared<CudfVector>(
      input_->pool(), outputType_, size, std::move(outputTable), stream);
  input_.reset();
  if (size == 0) {
    return nullptr;
  }
  return cudfOutput;
}

void CudfFilterProject::filter(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    rmm::cuda_stream_view stream) {
  // Evaluate the Filter
  std::vector<cudf::column_view> inputViews;
  inputViews.reserve(inputTableColumns.size());
  for (auto& col : inputTableColumns) {
    inputViews.push_back(col->view());
  }
  auto filterColumn = filterEvaluator_->eval(
      inputViews, stream, cudf::get_current_device_resource_ref(), true);
  auto filterColumnView = asView(filterColumn);
  bool shouldApplyFilter = [&]() {
    if (filterColumnView.has_nulls()) {
      return true;
    }
    // check if all values in filterColumnView are true
    auto isAllTrue = cudf::reduce(
        filterColumnView,
        *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        cudf::get_current_device_resource_ref());
    using ScalarType = cudf::scalar_type_t<bool>;
    auto result = static_cast<ScalarType*>(isAllTrue.get());
    // If filter is not all true, apply the filter
    return !(result->is_valid(stream) && result->value(stream));
  }();
  if (shouldApplyFilter) {
    auto filterTable =
        std::make_unique<cudf::table>(std::move(inputTableColumns));
    auto filteredTable =
        cudf::apply_boolean_mask(*filterTable, filterColumnView, stream);
    inputTableColumns = filteredTable->release();
  }
}

std::vector<std::unique_ptr<cudf::column>> CudfFilterProject::project(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::column_view> inputViews;
  inputViews.reserve(inputTableColumns.size());
  for (auto& col : inputTableColumns) {
    inputViews.push_back(col->view());
  }
  std::vector<ColumnOrView> columns;
  for (auto& projectEvaluator : projectEvaluators_) {
    columns.push_back(projectEvaluator->eval(
        inputViews, stream, cudf::get_current_device_resource_ref(), true));
  }

  // Rearrange columns to match outputType_
  std::vector<std::unique_ptr<cudf::column>> outputColumns(outputType_->size());
  // computed resultProjections
  for (int i = 0; i < resultProjections_.size(); i++) {
    auto& columnOrView = columns[i];
    if (std::holds_alternative<std::unique_ptr<cudf::column>>(columnOrView)) {
      // Move the owned column
      outputColumns[resultProjections_[i].outputChannel] =
          std::move(std::get<std::unique_ptr<cudf::column>>(columnOrView));
    } else {
      // Materialize the column_view into an owned column
      auto view = std::get<cudf::column_view>(columnOrView);
      outputColumns[resultProjections_[i].outputChannel] =
          std::make_unique<cudf::column>(
              view, stream, cudf::get_current_device_resource_ref());
    }
  }

  // Count occurrences of each inputChannel, and move columns if they occur only
  // once
  std::unordered_map<column_index_t, int> inputChannelCount;
  for (const auto& identity : identityProjections_) {
    inputChannelCount[identity.inputChannel]++;
  }

  // identityProjections (input to output copy)
  for (auto const& identity : identityProjections_) {
    VELOX_CHECK_NOT_NULL(inputTableColumns[identity.inputChannel]);
    if (inputChannelCount[identity.inputChannel] == 1) {
      // Move the column if it occurs only once
      outputColumns[identity.outputChannel] =
          std::move(inputTableColumns[identity.inputChannel]);
    } else {
      // Otherwise, copy the column and decrement the count
      outputColumns[identity.outputChannel] = std::make_unique<cudf::column>(
          *inputTableColumns[identity.inputChannel],
          stream,
          cudf::get_current_device_resource_ref());
    }
    VELOX_CHECK_GT(inputChannelCount[identity.inputChannel], 0);
    inputChannelCount[identity.inputChannel]--;
  }

  return outputColumns;
}

bool CudfFilterProject::allInputProcessed() {
  return !input_;
}

bool CudfFilterProject::isFinished() {
  return noMoreInput_ && allInputProcessed();
}

} // namespace facebook::velox::cudf_velox
