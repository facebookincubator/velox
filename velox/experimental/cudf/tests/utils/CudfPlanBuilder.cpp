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

#include "velox/experimental/cudf/tests/utils/CudfPlanBuilder.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/vector/ComplexVector.h"

using namespace facebook::velox;

namespace facebook::velox::cudf_velox::test {

namespace {
RowTypePtr concat(const RowTypePtr& a, const RowTypePtr& b) {
  std::vector<std::string> names = a->names();
  std::vector<TypePtr> types = a->children();
  names.insert(names.end(), b->names().begin(), b->names().end());
  types.insert(types.end(), b->children().begin(), b->children().end());
  return ROW(std::move(names), std::move(types));
}

RowTypePtr extract(
    const RowTypePtr& type,
    const std::vector<std::string>& childNames) {
  std::vector<std::string> names = childNames;

  std::vector<TypePtr> types;
  types.reserve(childNames.size());
  for (const auto& name : childNames) {
    types.emplace_back(type->findChild(name));
  }
  return ROW(std::move(names), std::move(types));
}

// TODO: The field and fields functions are static members of PlanBuilder but
// are private
std::shared_ptr<const core::FieldAccessTypedExpr> field(
    const RowTypePtr& inputType,
    column_index_t index) {
  auto name = inputType->names()[index];
  auto type = inputType->childAt(index);
  return std::make_shared<core::FieldAccessTypedExpr>(type, name);
}

std::shared_ptr<const core::FieldAccessTypedExpr> field(
    const RowTypePtr& inputType,
    const std::string& name) {
  column_index_t index = inputType->getChildIdx(name);
  return field(inputType, index);
}

std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> fields_(
    const RowTypePtr& inputType,
    const std::vector<std::string>& names) {
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> fields;
  for (const auto& name : names) {
    fields.push_back(field(inputType, name));
  }
  return fields;
}

std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> fields_(
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& indices) {
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> fields;
  for (auto& index : indices) {
    fields.push_back(field(inputType, index));
  }
  return fields;
}
} // namespace

CudfPlanBuilder::CudfPlanBuilder(
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
    memory::MemoryPool* pool)
    : PlanBuilder(planNodeIdGenerator, pool) {}

CudfPlanBuilder& CudfPlanBuilder::hashJoin(
    const std::vector<std::string>& leftKeys,
    const std::vector<std::string>& rightKeys,
    const core::PlanNodePtr& build,
    const std::string& filter,
    const std::vector<std::string>& outputLayout,
    core::JoinType joinType,
    bool nullAware) {
  std::cout << "Calling CudfPlanBuilder::hashJoin" << std::endl;

  VELOX_CHECK_NOT_NULL(planNode_, "CudfHashJoin cannot be the source node");
  VELOX_CHECK_EQ(leftKeys.size(), rightKeys.size());

  auto leftType = planNode_->outputType();
  auto rightType = build->outputType();
  auto resultType = concat(leftType, rightType);
  core::TypedExprPtr filterExpr;
  /*
  // TODO: Can't use pool_ because it is private. Skipping filterExpr.
  if (!filter.empty()) {
    filterExpr = parseExpr(filter, resultType, options_, pool_);
  }
  */

  RowTypePtr outputType;
  if (isLeftSemiProjectJoin(joinType) || isRightSemiProjectJoin(joinType)) {
    std::vector<std::string> names = outputLayout;

    // Last column in 'outputLayout' must be a boolean 'match'.
    std::vector<TypePtr> types;
    types.reserve(outputLayout.size());
    for (auto i = 0; i < outputLayout.size() - 1; ++i) {
      types.emplace_back(resultType->findChild(outputLayout[i]));
    }
    types.emplace_back(BOOLEAN());

    outputType = ROW(std::move(names), std::move(types));
  } else {
    outputType = extract(resultType, outputLayout);
  }

  auto leftKeyFields = fields_(leftType, leftKeys);
  auto rightKeyFields = fields_(rightType, rightKeys);

  planNode_ = std::make_shared<cudf_velox::CudfHashJoinNode>(
      nextPlanNodeId(),
      joinType,
      nullAware,
      leftKeyFields,
      rightKeyFields,
      std::move(filterExpr),
      std::move(planNode_),
      build,
      outputType);

  return *this;
}

} // namespace facebook::velox::cudf_velox::test
