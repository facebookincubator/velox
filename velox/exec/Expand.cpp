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
  const auto& inputType = expandNode->sources()[0]->outputType();
  auto numProjectSets = expandNode->projectSets().size();
  projectMappings_.reserve(numProjectSets);
  constantMappings_.reserve(numProjectSets);
  auto numProjects = expandNode->names().size();
  for (const auto& projectSet : expandNode->projectSets()) {
    std::vector<column_index_t> projectMapping;
    projectMapping.reserve(numProjects);
    std::vector<ConstantTypedExprPtr> constantMapping;
    constantMapping.reserve(numProjects);
    for (const auto& project : projectSet) {
      if (auto field =
              std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                  project)) {
        projectMapping.push_back(inputType->getChildIdx(field->name()));
        constantMapping.push_back(nullptr);
      } else if (
          auto constant =
              std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                  project)) {
        projectMapping.push_back(kUnMapedProject);
        constantMapping.push_back(constant);
      } else {
        VELOX_FAIL("Unexpted expression for Expand");
      }
    }

    projectMappings_.emplace_back(std::move(projectMapping));
    constantMappings_.emplace_back(std::move(constantMapping));
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

  // Make a copy of input for the grouping set at 'projectSetIndex_'.
  auto numInput = input_->size();

  std::vector<VectorPtr> outputColumns(outputType_->size());

  const auto& projectMapping = projectMappings_[projectSetIndex_];
  const auto& constantMapping = constantMappings_[projectSetIndex_];
  auto numGroupingKeys = projectMapping.size();

  for (auto i = 0; i < numGroupingKeys; ++i) {
    if (projectMapping[i] == kUnMapedProject) {
      auto constantExpr = constantMapping[i];
      if (constantExpr->value().isNull()) {
        // Add null column.
        outputColumns[i] = BaseVector::createNullConstant(
            outputType_->childAt(i), numInput, pool());
      } else {
        // Add constant column: gid, gpos, etc.
        outputColumns[i] = BaseVector::createConstant(
            constantExpr->type(), constantExpr->value(), numInput, pool());
      }
    } else {
      outputColumns[i] = input_->childAt(projectMapping[i]);
    }
  }

  ++projectSetIndex_;
  if (projectSetIndex_ == projectMappings_.size()) {
    projectSetIndex_ = 0;
    input_ = nullptr;
  }

  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, numInput, std::move(outputColumns));
}

} // namespace facebook::velox::exec
