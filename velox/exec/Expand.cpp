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
#include "velox/exec/Expand.h"

namespace facebook::velox::exec {

Expand::Expand(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::ExpandNode>& expandNode)
    : Operator(
          driverCtx,
          expandNode->outputType(),
          operatorId,
          expandNode->id(),
          "Expand") {
  const auto& inputType = expandNode->inputType();
  const auto numRows = expandNode->projections().size();
  fieldProjections_.reserve(numRows);
  constantProjections_.reserve(numRows);
  constantOutputs_.reserve(numRows);
  const auto numColumns = expandNode->names().size();
  for (const auto& rowProjections : expandNode->projections()) {
    std::vector<column_index_t> rowProjection;
    rowProjection.reserve(numColumns);
    std::vector<std::shared_ptr<const core::ConstantTypedExpr>>
        constantProjection;
    constantProjection.reserve(numColumns);
    for (const auto& columnProjection : rowProjections) {
      if (auto field = core::TypedExprs::asFieldAccess(columnProjection)) {
        rowProjection.push_back(inputType->getChildIdx(field->name()));
        constantProjection.push_back(nullptr);
      } else if (
          auto constant = core::TypedExprs::asConstant(columnProjection)) {
        rowProjection.push_back(kConstantChannel);
        constantProjection.push_back(constant);
      } else {
        VELOX_USER_FAIL(
            "Expand operator doesn't support this expression. Only column references and constants are supported. {}",
            columnProjection->toString());
      }
    }

    fieldProjections_.emplace_back(std::move(rowProjection));
    constantProjections_.emplace_back(std::move(constantProjection));
  }
}

void Expand::initialize() {
  if (constantProjections_.empty()) {
    return;
  }
  const auto numColumns = constantProjections_[0].size();
  for (const auto& projections : constantProjections_) {
    std::vector<VectorPtr> constantOutput;
    constantOutput.reserve(numColumns);
    for (const auto& constant : projections) {
      if (constant) {
        constantOutput.push_back(constant->toConstantVector(pool()));
      } else {
        constantOutput.push_back(nullptr);
      }
    }
    constantOutputs_.emplace_back(std::move(constantOutput));
  }
}

bool Expand::needsInput() const {
  return !noMoreInput_ && input_ == nullptr;
}

void Expand::addInput(RowVectorPtr input) {
  // Load Lazy vectors.
  for (auto& child : input->children()) {
    child->loadedVector();
  }

  input_ = std::move(input);
}

RowVectorPtr Expand::getOutput() {
  if (!input_) {
    return nullptr;
  }

  const auto numInput = input_->size();

  std::vector<VectorPtr> outputColumns(outputType_->size());

  const auto& rowProjection = fieldProjections_[rowIndex_];
  const auto& constantProjection = constantOutputs_[rowIndex_];
  const auto numColumns = rowProjection.size();

  for (auto i = 0; i < numColumns; ++i) {
    if (rowProjection[i] == kConstantChannel) {
      outputColumns[i] =
          BaseVector::wrapInConstant(numInput, 0, constantProjection[i]);
    } else {
      outputColumns[i] = input_->childAt(rowProjection[i]);
    }
  }

  ++rowIndex_;
  if (rowIndex_ == fieldProjections_.size()) {
    rowIndex_ = 0;
    input_ = nullptr;
  }

  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, numInput, std::move(outputColumns));
}

} // namespace facebook::velox::exec
