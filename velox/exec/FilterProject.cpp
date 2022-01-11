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
#include "velox/exec/FilterProject.h"
#include "velox/core/Expressions.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {
namespace {
bool checkAddIdentityProjection(
    const std::shared_ptr<const core::ITypedExpr>& projection,
    const std::shared_ptr<const RowType>& inputType,
    ChannelIndex outputChannel,
    std::vector<IdentityProjection>& identityProjections) {
  if (auto field = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
          projection)) {
    if (field->inputs().empty()) {
      const auto inputChannel = inputType->getChildIdx(field->name());
      identityProjections.emplace_back(inputChannel, outputChannel);
      return true;
    }
  }

  return false;
}
} // namespace

FilterProject::FilterProject(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project)
    : Operator(
          driverCtx,
          project ? project->outputType() : filter->outputType(),
          operatorId,
          project ? project->id() : filter->id(),
          "FilterProject"),
      hasFilter_(filter != nullptr) {
  std::vector<std::shared_ptr<const core::ITypedExpr>> allExprs;
  if (hasFilter_) {
    allExprs.push_back(filter->filter());
  }
  if (project) {
    auto inputType = project->sources()[0]->outputType();
    for (ChannelIndex i = 0; i < project->projections().size(); i++) {
      auto projection = project->projections()[i];
      bool identityProjection = checkAddIdentityProjection(
          projection, inputType, i, identityProjections_);
      if (!identityProjection) {
        allExprs.push_back(projection);
        resultProjections_.emplace_back(allExprs.size() - 1, i);
      }
    }
  } else {
    for (ChannelIndex i = 0; i < outputType_->size(); ++i) {
      identityProjections_.emplace_back(i, i);
    }
    isIdentityProjection_ = true;
  }
  numExprs_ = allExprs.size();
  exprs_ = makeExprSetFromFlag(std::move(allExprs), operatorCtx_->execCtx());
}

void FilterProject::addInput(RowVectorPtr input) {
  input_ = std::move(input);
  numProcessedInputRows_ = 0;
  getResultVectors(&results_);
  int32_t firstProjected;
  if (resultProjections_.empty()) {
    firstProjected = results_.size();
  } else {
    firstProjected = resultProjections_[0].inputChannel;
  }

  for (int32_t i = 0; i < firstProjected; ++i) {
    if (results_[i]) {
      if (results_[i]->encoding() == VectorEncoding::Simple::FLAT) {
        results_[i]->clear();
      } else {
        results_[i] = nullptr;
      }
    }
  }
}

bool FilterProject::allInputProcessed() {
  if (!input_) {
    return true;
  }
  if (numProcessedInputRows_ == input_->size()) {
    inputProcessed();
    return true;
  }
  return false;
}

bool FilterProject::isFinished() {
  return noMoreInput_ && allInputProcessed();
}

RowVectorPtr FilterProject::getOutput() {
  if (allInputProcessed()) {
    clearIdentityProjectedOutput();
    clearNonReusableOutput();
    return nullptr;
  }
  vector_size_t size = input_->size();
  LocalSelectivityVector localRows(operatorCtx_->execCtx(), size);
  auto* rows = localRows.get();
  rows->setAll();
  EvalCtx evalCtx(operatorCtx_->execCtx(), exprs_.get(), input_.get());
  if (!hasFilter_) {
    numProcessedInputRows_ = size;
    VELOX_CHECK(!isIdentityProjection_);
    project(*rows, &evalCtx);

    if (results_.size() > 0) {
      auto outCol = results_[0];
      if (outCol && outCol->isCodegenOutput()) {
        // codegen can output different size when it merged filter + projection
        size = outCol->size();
        if (size == 0) { // all filtered out
          return nullptr;
        }
      }
    }

    return fillOutput(size, nullptr);
  }

  // evaluate filter
  auto numOut = filter(&evalCtx, *rows);
  numProcessedInputRows_ = size;
  if (numOut == 0) { // no rows passed the filer
    inputProcessed();
    return nullptr;
  }

  bool allRowsSelected = (numOut == size);

  // evaluate projections (if present)
  if (!isIdentityProjection_) {
    if (!allRowsSelected) {
      rows->setFromBits(filterEvalCtx_.selectedBits->as<uint64_t>(), size);
    }
    project(*rows, &evalCtx);
  }

  return fillOutput(
      numOut, allRowsSelected ? nullptr : filterEvalCtx_.selectedIndices);
}

void FilterProject::clearNonReusableOutput() {
  if (!output_) {
    return;
  }
  if (!output_.unique()) {
    output_ = nullptr;
    return;
  }
  for (auto& child : output_->children()) {
    if (!child) {
      continue;
    }
    if (!BaseVector::isReusableFlatVector(child)) {
      child = nullptr;
    }
  }
}

void FilterProject::project(const SelectivityVector& rows, EvalCtx* evalCtx) {
  // Make sure LazyVectors are loaded for all the "rows".
  //
  // Consider projection with 2 expressions: f(a) AND g(b), h(b)
  // If b is a LazyVector and f(a) AND g(b) expression is evaluated first, it
  // will load b only for rows where f(a) is true. However, h(b) projection
  // needs all rows for "b".
  //
  // This works, but may load more rows than necessary. E.g. if we only have
  // f(a) AND g(b) expression and b is not used anywhere else, it is sufficient
  // to load b for a subset of rows where f(a) is true.
  *evalCtx->mutableIsFinalSelection() = false;
  *evalCtx->mutableFinalSelection() = &rows;

  exprs_->eval(
      hasFilter_ ? 1 : 0, numExprs_, !hasFilter_, rows, evalCtx, &results_);
}

vector_size_t FilterProject::filter(
    EvalCtx* evalCtx,
    const SelectivityVector& allRows) {
  exprs_->eval(0, 1, true, allRows, evalCtx, &results_);
  return processFilterResults(results_[0], allRows, filterEvalCtx_, pool());
}
} // namespace facebook::velox::exec
