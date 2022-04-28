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

#include "velox/substrait/VeloxToSubstraitPlan.h"

namespace facebook::velox::substrait {

::substrait::Plan* VeloxToSubstraitPlanConvertor::toSubstrait(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const PlanNode>& plan) {
  // Assume only accepts a single plan fragment.
  // TODO add root_rel
  ::substrait::Plan* substraitPlan =
      google::protobuf::Arena::CreateMessage<::substrait::Plan>(&arena);
  ::substrait::Rel* rel = substraitPlan->add_relations()->mutable_rel();
  toSubstrait(arena, plan, rel);

  return substraitPlan;
}

void VeloxToSubstraitPlanConvertor::toSubstrait(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const PlanNode>& planNode,
    ::substrait::Rel* rel) {
  if (auto filterNode = std::dynamic_pointer_cast<const FilterNode>(planNode)) {
    auto filterRel = rel->mutable_filter();
    toSubstrait(arena, filterNode, filterRel);
    return;
  }
  if (auto valuesNode = std::dynamic_pointer_cast<const ValuesNode>(planNode)) {
    ::substrait::ReadRel* readRel = rel->mutable_read();
    toSubstrait(arena, valuesNode, readRel);
    return;
  }
  if (auto projectNode =
          std::dynamic_pointer_cast<const ProjectNode>(planNode)) {
    ::substrait::ProjectRel* projectRel = rel->mutable_project();
    toSubstrait(arena, projectNode, projectRel);
    return;
  }
}

void VeloxToSubstraitPlanConvertor::toSubstrait(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const FilterNode>& filterNode,
    ::substrait::FilterRel* filterRel) {
  std::vector<std::shared_ptr<const PlanNode>> sources = filterNode->sources();

  // Check there only have one input.
  VELOX_USER_CHECK_EQ(
      1, sources.size(), "Filter plan node must have extract one source.");
  auto source = sources[0];

  ::substrait::Rel* filterInput = filterRel->mutable_input();
  // Build source.
  toSubstrait(arena, source, filterInput);

  // Construct substrait expr(Filter condition).
  auto filterCondition = filterNode->filter();
  auto inputType = source->outputType();
  filterRel->set_allocated_condition(
      exprConvertor_.toSubstraitExpr(arena, filterCondition, inputType));

  filterRel->mutable_common()->mutable_direct();
}

void VeloxToSubstraitPlanConvertor::toSubstrait(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const ValuesNode>& valuesNode,
    ::substrait::ReadRel* readRel) {
  const auto& outputType = valuesNode->outputType();

  ::substrait::ReadRel_VirtualTable* virtualTable =
      readRel->mutable_virtual_table();

  readRel->set_allocated_base_schema(
      typeConvertor_.toSubstraitNamedStruct(arena, outputType));
  // The row number of the input data.
  int64_t numRows = valuesNode->values().size();

  // There have multiple rows in the data and each row is a RowVectorPtr.
  for (int64_t row = 0; row < numRows; ++row) {
    // The row data.
    ::substrait::Expression_Literal_Struct* litValue =
        virtualTable->add_values();
    const auto& rowValue = valuesNode->values().at(row);
    // The column number of the row data.
    int64_t numColumns = rowValue->childrenSize();

    for (int64_t column = 0; column < numColumns; ++column) {
      ::substrait::Expression_Literal* substraitField =
          google::protobuf::Arena::CreateMessage<
              ::substrait::Expression_Literal>(&arena);

      VectorPtr children = rowValue->childAt(column);

      substraitField->MergeFrom(*exprConvertor_.toSubstraitExpr(
          arena, std::make_shared<ConstantTypedExpr>(children), litValue));
    }
  }
  readRel->mutable_common()->mutable_direct();
}

void VeloxToSubstraitPlanConvertor::toSubstrait(
    google::protobuf::Arena& arena,
    const std::shared_ptr<const ProjectNode>& projectNode,
    ::substrait::ProjectRel* projectRel) {
  std::vector<std::shared_ptr<const ITypedExpr>> projections =
      projectNode->projections();

  std::vector<std::shared_ptr<const PlanNode>> sources = projectNode->sources();
  // Check there only have one input.
  VELOX_USER_CHECK_EQ(
      1, sources.size(), "Project plan node must have extract one source.");
  // The previous node.
  std::shared_ptr<const PlanNode> source = sources[0];

  // Process the source Node.
  ::substrait::Rel* projectRelInput = projectRel->mutable_input();
  toSubstrait(arena, source, projectRelInput);

  // Remap the output.
  ::substrait::RelCommon_Emit* projRelEmit =
      projectRel->mutable_common()->mutable_emit();

  int64_t projectionSize = projections.size();

  auto inputType = source->outputType();
  int64_t projectEmitReMapId = inputType->size();

  for (int64_t i = 0; i < projectionSize; i++) {
    std::shared_ptr<const ITypedExpr>& veloxExpr = projections.at(i);

    projectRel->add_expressions()->MergeFrom(
        *exprConvertor_.toSubstraitExpr(arena, veloxExpr, inputType));

    // Add outputMapping for each expression.
    projRelEmit->add_output_mapping(projectEmitReMapId);
    projectEmitReMapId += 1;
  }

  return;
}

} // namespace facebook::velox::substrait
