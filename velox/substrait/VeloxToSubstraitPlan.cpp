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

// Velox Plan to Substrait
void VeloxToSubstraitPlanConvertor::toSubstrait(
    std::shared_ptr<const PlanNode> vPlan,
    ::substrait::Plan& sPlan) {
  // Assume only accepts a single plan fragment
  // TODO: convert the Split RootNode get from dispatcher to RootRel
  ::substrait::Rel* sRel = sPlan.add_relations()->mutable_rel();
  toSubstrait(vPlan, sRel);
}

void VeloxToSubstraitPlanConvertor::toSubstrait(
    std::shared_ptr<const PlanNode> vPlanNode,
    ::substrait::Rel* sRel) {
  if (auto filterNode =
          std::dynamic_pointer_cast<const FilterNode>(vPlanNode)) {
    auto sFilterRel = sRel->mutable_filter();
    toSubstrait(filterNode, sFilterRel);
    return;
  }
  if (auto vValuesNode =
          std::dynamic_pointer_cast<const ValuesNode>(vPlanNode)) {
    ::substrait::ReadRel* sReadRel = sRel->mutable_read();
    toSubstrait(vValuesNode, sReadRel);
    return;
  }
  if (auto vProjNode =
          std::dynamic_pointer_cast<const ProjectNode>(vPlanNode)) {
    ::substrait::ProjectRel* sProjRel = sRel->mutable_project();
    toSubstrait(vProjNode, sProjRel);
    return;
  }
}

void VeloxToSubstraitPlanConvertor::toSubstrait(
    std::shared_ptr<const FilterNode> vFilterNode,
    ::substrait::FilterRel* sFilterRel) {
  const PlanNodeId vId = vFilterNode->id();
  std::shared_ptr<const PlanNode> vSource;
  std::vector<std::shared_ptr<const PlanNode>> vSources =
      vFilterNode->sources();
  // check how many inputs there have
  int64_t vSourceSize = vSources.size();
  if (vSourceSize == 0) {
    VELOX_FAIL("Filter Node must have input");
  } else if (vSourceSize == 1) {
    vSource = vSources[0];
  } else {
    // TODO
    // select one in the plan fragment pass to transformVExpr
    //  and the other change into root or simpleCapture.
    VELOX_NYI("Haven't support multiple inputs now.");
  }
  std::shared_ptr<const ITypedExpr> vFilterCondition = vFilterNode->filter();

  ::substrait::Rel* sFilterInput = sFilterRel->mutable_input();
  ::substrait::Expression* sFilterCondition = sFilterRel->mutable_condition();
  //   Build source
  toSubstrait(vSource, sFilterInput);

  RowTypePtr vPreNodeOutPut = vSource->outputType();
  //   Construct substrait expr
  v2SExprConvertor_.toSubstraitExpr(
      sFilterCondition, vFilterCondition, vPreNodeOutPut);
  sFilterRel->mutable_common()->mutable_direct();
}

void VeloxToSubstraitPlanConvertor::toSubstrait(
    std::shared_ptr<const ValuesNode> vValuesNode,
    ::substrait::ReadRel* sReadRel) {
  const RowTypePtr vOutPut = vValuesNode->outputType();

  ::substrait::ReadRel_VirtualTable* sVirtualTable =
      sReadRel->mutable_virtual_table();

  ::substrait::NamedStruct* sBaseSchema = sReadRel->mutable_base_schema();
  v2STypeConvertor_.toSubstraitNamedStruct(vOutPut, sBaseSchema);

  const PlanNodeId id = vValuesNode->id();
  // sread.virtual_table().values_size(); multi rows
  int64_t numRows = vValuesNode->values().size();
  // should be the same value.kFieldsFieldNumber  = vOutputType->size();
  int64_t numColumns;
  // multi rows, each row is a RowVectorPrt

  for (int64_t row = 0; row < numRows; ++row) {
    // the specfic row
    ::substrait::Expression_Literal_Struct* sLitValue =
        sVirtualTable->add_values();
    RowVectorPtr rowValue = vValuesNode->values().at(row);
    // the column numbers in the specfic row.
    numColumns = rowValue->childrenSize();

    for (int64_t column = 0; column < numColumns; ++column) {
      ::substrait::Expression_Literal* sField;

      VectorPtr children = rowValue->childAt(column);
      sField = v2STypeConvertor_.processVeloxValueByType(
          sLitValue, sField, children);
    }
  }
  sReadRel->mutable_common()->mutable_direct();
}

void VeloxToSubstraitPlanConvertor::toSubstrait(
    std::shared_ptr<const ProjectNode> vProjNode,
    ::substrait::ProjectRel* sProjRel) {
  // the info from vProjNode
  const PlanNodeId vId = vProjNode->id();
  std::vector<std::string> vNames = vProjNode->names();
  std::vector<std::shared_ptr<const ITypedExpr>> vProjections =
      vProjNode->projections();
  const RowTypePtr vOutput = vProjNode->outputType();

  // check how many inputs there have
  std::vector<std::shared_ptr<const PlanNode>> vSources = vProjNode->sources();
  // the PreNode
  std::shared_ptr<const PlanNode> vSource;
  int64_t vSourceSize = vSources.size();
  if (vSourceSize == 0) {
    VELOX_FAIL("Project Node must have input");
  } else if (vSourceSize == 1) {
    vSource = vSources[0];
  } else {
    // TODO
    // select one in the plan fragment pass to transformVExpr
    //  and the other change into root or simpleCapture.
    VELOX_NYI("Haven't support multiple inputs now.");
  }

  // process the source Node.
  ::substrait::Rel* sProjInput = sProjRel->mutable_input();
  toSubstrait(vSource, sProjInput);

  // remapping the output
  ::substrait::RelCommon_Emit* sProjEmit =
      sProjRel->mutable_common()->mutable_emit();

  int64_t vProjectionSize = vProjections.size();

  RowTypePtr vPreNodeOutPut = vSource->outputType();
  std::vector<std::string> vPreNodeColNames = vPreNodeOutPut->names();
  std::vector<std::shared_ptr<const velox::Type>> vPreNodeColTypes =
      vPreNodeOutPut->children();
  int64_t vPreNodeColNums = vPreNodeColNames.size();
  int64_t sProjEmitReMapId = vPreNodeColNums;

  for (int64_t i = 0; i < vProjectionSize; i++) {
    std::shared_ptr<const ITypedExpr>& vExpr = vProjections.at(i);
    ::substrait::Expression* sExpr = sProjRel->add_expressions();

    v2SExprConvertor_.toSubstraitExpr(sExpr, vExpr, vPreNodeOutPut);
    // add outputMapping for each vExpr
    const std::shared_ptr<const Type> vExprType = vExpr->type();

    bool sProjEmitReMap = false;
    for (int64_t j = 0; j < vPreNodeColNums; j++) {
      if (vExprType == vPreNodeColTypes[j] &&
          vOutput->nameOf(i) == vPreNodeColNames[j]) {
        sProjEmit->add_output_mapping(j);
        sProjEmitReMap = true;
        break;
      }
    }
    if (!sProjEmitReMap) {
      sProjEmit->add_output_mapping(sProjEmitReMapId++);
    }
  }

  return;
}

} // namespace facebook::velox::substrait
