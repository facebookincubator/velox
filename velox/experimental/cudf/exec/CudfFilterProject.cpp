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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Validation.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/expression/ExprOptimizer.h"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

namespace {

void debugPrintTree(
    const core::TypedExprPtr& expr,
    int indent = 0,
    std::ostream& os = std::cout) {
  if (indent == 0)
    os << "=== Expression Tree ===" << std::endl;
  os << std::string(indent, ' ') << core::ExprKindName::toName(expr->kind())
     << "(" << expr->type()->toString() << ")" << std::endl;
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

CudfFilterProject::CudfFilterProject(
    int32_t operatorId,
    velox::exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          project ? project->outputType() : filter->outputType(),
          project ? project->id() : filter->id(),
          "CudfFilterProject",
          nvtx3::rgb{220, 20, 60}, // Crimson
          NvtxMethodFlag::kAll,
          std::nullopt,
          project ? std::static_pointer_cast<const core::PlanNode>(project)
                  : std::static_pointer_cast<const core::PlanNode>(filter)),
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

  const auto inputType = project_ ? project_->sources()[0]->outputType()
                                  : filter_->sources()[0]->outputType();

  // convert to AST
  if (CudfConfig::getInstance().debugEnabled) {
    int i = 0;
    for (const auto& expr : allExprs) {
      LOG(INFO) << "expr[" << i++ << "] " << expr->toString();
      debugPrintTree(expr, 0, LOG(INFO));
    }
  }
  // Optimize (rewrites + constant folding) each expression before evaluator
  // selection so CudfFunctions never see scalar-only operand sets, then
  // compile. The operator pool owns the folded constants for the evaluator's
  // lifetime.
  auto* const queryCtx = operatorCtx_->execCtx()->queryCtx();
  auto* const pool = operatorCtx_->pool();
  inputRowType_ = inputType;
  const auto optimizeAndCompile =
      [this, inputType, queryCtx, pool](const core::TypedExprPtr& expr) {
        auto optimized = expression::optimize(expr, queryCtx, pool);
        // Kept for the CPU re-run, which evaluates the tree the GPU compiled.
        cpuExprSource_.push_back(optimized);
        return createCudfExpression(
            optimized, inputType, pool, queryCtx->queryConfig());
      };
  if (hasFilter_) {
    // First expr is Filter, rest are Project.
    filterEvaluator_ = optimizeAndCompile(allExprs.front());
    std::transform(
        allExprs.begin() + 1,
        allExprs.end(),
        std::back_inserter(projectEvaluators_),
        optimizeAndCompile);
  } else {
    std::transform(
        allExprs.begin(),
        allExprs.end(),
        std::back_inserter(projectEvaluators_),
        optimizeAndCompile);
  }

  filter_.reset();
  project_.reset();
}

// Builds the operator's output from CPU results as project() does on the
// device: computed expressions on their result channels, pass-through columns
// copied, and `selected` the filter's verdict. Assembled by hand because this
// class's resultProjections_ and identityProjections_ shadow the base class's,
// so Operator::fillOutput would see empty ones.
RowVectorPtr CudfFilterProject::assembleCpuOutput(
    const RowVectorPtr& hostInput,
    std::vector<VectorPtr>& results,
    const SelectivityVector& selected,
    cuda::stream_ref stream) {
  auto* const pool = operatorCtx_->pool();
  const auto numSelected = selected.countSelected();

  // Row numbers the filter kept, so every column can be gathered the same way.
  BufferPtr indices = allocateIndices(numSelected, pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  vector_size_t next = 0;
  selected.applyToSelected(
      [&](vector_size_t row) { rawIndices[next++] = row; });

  // Encodings stay as Velox produced them; toCudfTable flattens them.
  const auto wrap = [&](const VectorPtr& source) -> VectorPtr {
    if (numSelected == source->size()) {
      return source;
    }
    return BaseVector::wrapInDictionary(
        /*nulls=*/nullptr, indices, numSelected, source);
  };

  std::vector<VectorPtr> children(outputType_->size());
  for (const auto& projection : resultProjections_) {
    // inputChannel indexes the expression list, which is the order of
    // `results`, with the filter at index 0 when there is one.
    children[projection.outputChannel] = wrap(results[projection.inputChannel]);
  }
  for (const auto& identity : identityProjections_) {
    children[identity.outputChannel] =
        wrap(hostInput->childAt(identity.inputChannel));
  }

  auto output = std::make_shared<RowVector>(
      pool, outputType_, nullptr, numSelected, std::move(children));

  // The operator emits CudfVectors, so the CPU answer goes back to the device.
  auto table = with_arrow::toCudfTable(output, pool, stream, get_output_mr());
  return std::make_shared<CudfVector>(
      pool, outputType_, numSelected, std::move(table), stream);
}

RowVectorPtr CudfFilterProject::evaluateOnCpu(
    std::vector<std::unique_ptr<cudf::column>> columns,
    bool applyFilter,
    cuda::stream_ref stream) {
  // Velox's evaluator raises the real error and narrows rows through any
  // conditional, so a row the GPU declined but a conditional discards raises
  // nothing.
  std::vector<cudf::column_view> views;
  views.reserve(columns.size());
  for (const auto& column : columns) {
    views.push_back(column->view());
  }

  // The whole input row, with the input's field names: a FieldReference in the
  // expression resolves by name.
  auto hostInput = with_arrow::toVeloxColumn(
      cudf::table_view{views},
      operatorCtx_->pool(),
      std::static_pointer_cast<const Type>(inputRowType_),
      stream,
      get_temp_mr());
  stream.sync();

  auto* const execCtx = operatorCtx_->execCtx();
  if (cpuExprs_ == nullptr) {
    // Built on first use, since most queries never reach this path.
    auto source = cpuExprSource_;
    cpuExprs_ = velox::exec::makeExprSetFromFlag(std::move(source), execCtx);
  }

  velox::exec::LocalSelectivityVector rowsHolder(*execCtx, hostInput->size());
  auto* const rows = rowsHolder.get();
  rows->setAll();
  velox::exec::EvalCtx evalCtx(execCtx, cpuExprs_.get(), hostInput.get());

  std::vector<VectorPtr> results;
  cpuExprs_->eval(*rows, evalCtx, results);

  // The declined row did not raise on the CPU, so a conditional discarded it:
  // the CPU's answer replaces the GPU's for this batch.
  auto selected = rowsHolder.get();
  if (applyFilter) {
    const auto& filterResult = results.front();
    velox::exec::LocalSelectivityVector filteredHolder(
        *execCtx, hostInput->size());
    auto* const filtered = filteredHolder.get();
    filtered->clearAll();
    auto* const decoded = filterResult->as<SimpleVector<bool>>();
    VELOX_CHECK_NOT_NULL(
        decoded, "A filter must evaluate to a flat boolean on the CPU path");
    for (vector_size_t row = 0; row < hostInput->size(); ++row) {
      if (!decoded->isNullAt(row) && decoded->valueAt(row)) {
        filtered->setValid(row, true);
      }
    }
    filtered->updateBounds();
    return assembleCpuOutput(hostInput, results, *filtered, stream);
  }
  return assembleCpuOutput(hostInput, results, *selected, stream);
}

void CudfFilterProject::doAddInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr CudfFilterProject::doGetOutput() {
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
  auto outputSize = input_->size();

  if (hasFilter_ && !filter(inputTableColumns, stream)) {
    // The filter declined a row, and its own error is the one to raise, so
    // the CPU redoes the filter as well as the projections.
    auto output = evaluateOnCpu(
        std::move(inputTableColumns), /*applyFilter=*/true, stream);
    // Retired here as on every other exit: the columns were released above, so
    // the input now holds a null table.
    input_.reset();
    return output;
  }
  if (!inputTableColumns.empty()) {
    outputSize = inputTableColumns.front()->size();
  }
  auto projected = project(inputTableColumns, stream);
  if (!projected.has_value()) {
    // A projection declined. The filter already ran, so re-project its
    // survivors, as the CPU operator does.
    auto output = evaluateOnCpu(
        std::move(inputTableColumns), /*applyFilter=*/false, stream);
    input_.reset();
    return output;
  }
  auto outputColumns = std::move(projected).value();

  auto outputTable = std::make_unique<cudf::table>(std::move(outputColumns));
  auto const numColumns = outputTable->num_columns();
  auto const size = numColumns > 0 ? outputTable->num_rows() : outputSize;
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "cudfProject Output: " << size << " rows, " << numColumns
            << " columns";
  }
  if (size == 0) {
    input_.reset();
    return nullptr;
  }
  auto cudfOutput = std::make_shared<CudfVector>(
      input_->pool(), outputType_, size, std::move(outputTable), stream);
  input_.reset();
  return cudfOutput;
}

bool CudfFilterProject::filter(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    cuda::stream_ref stream) {
  // Evaluate the Filter
  std::vector<cudf::column_view> inputViews;
  inputViews.reserve(inputTableColumns.size());
  for (auto& col : inputTableColumns) {
    inputViews.push_back(col->view());
  }
  gpu_sfi::GpuSfiErrors errors(stream, get_temp_mr());
  auto filterColumn =
      filterEvaluator_->eval(inputViews, stream, get_temp_mr(), true, &errors);
  // Checked before the retention mask drops the declined row, while the input
  // is still whole.
  if (errors.resolve() != gpu_sfi::ErrorClass::kNone) {
    return false;
  }
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
        get_temp_mr());
    using ScalarType = cudf::scalar_type_t<bool>;
    auto result = static_cast<ScalarType*>(isAllTrue.get());
    // If filter is not all true, apply the filter
    return !(result->is_valid(stream) && result->value(stream));
  }();
  if (shouldApplyFilter) {
    auto filterTable =
        std::make_unique<cudf::table>(std::move(inputTableColumns));
    auto filteredTable = cudf::apply_retention_mask(
        *filterTable, filterColumnView, stream, get_output_mr());
    inputTableColumns = filteredTable->release();
  }
  return true;
}

std::optional<std::vector<std::unique_ptr<cudf::column>>>
CudfFilterProject::project(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    cuda::stream_ref stream) {
  std::vector<cudf::column_view> inputViews;
  inputViews.reserve(inputTableColumns.size());
  for (auto& col : inputTableColumns) {
    inputViews.push_back(col->view());
  }
  std::vector<ColumnOrView> columns;
  // One word for the whole batch: resolve() synchronizes the stream, and a
  // declined row discards every projection's result anyway. Each projection
  // reads only the input columns, so a declined row in one cannot reach
  // another.
  gpu_sfi::GpuSfiErrors errors(stream, get_temp_mr());
  for (auto& projectEvaluator : projectEvaluators_) {
    columns.push_back(projectEvaluator->eval(
        inputViews, stream, get_output_mr(), true, &errors));
  }
  // Checked while the input columns are still whole -- the identity moves
  // below have not run.
  if (errors.resolve() != gpu_sfi::ErrorClass::kNone) {
    return std::nullopt;
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
          std::make_unique<cudf::column>(view, stream, get_output_mr());
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
          *inputTableColumns[identity.inputChannel], stream, get_output_mr());
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
