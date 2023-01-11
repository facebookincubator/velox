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

#include "velox/substrait/tests/SubstraitTestPlanBuilder.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/substrait/VeloxToSubstraitExpr.h"

#include "velox/type/Type.h"

namespace facebook::velox::substrait {

using facebook::velox::exec::test::AssertQueryBuilder;

std::unique_ptr<SubstraitRelCommon> MakeRelCommon(
    std::vector<int> emitIndices) {
  std::unique_ptr<SubstraitRelCommon> relCommon =
      std::make_unique<SubstraitRelCommon>();
  std::unique_ptr<SubstraitRelCommonEmit> relCommonEmit =
      std::make_unique<SubstraitRelCommonEmit>();
  for (int idx : emitIndices) {
    relCommonEmit->add_output_mapping(idx);
  }
  relCommon->set_allocated_emit(relCommonEmit.release());
  return relCommon;
}

std::unique_ptr<SubstraitReadRel> MakeReadRel(
    const SubstraitExtensionCollectorPtr& extensionCollector,
    google::protobuf::Arena& arena,
    std::vector<RowVectorPtr> data,
    std::vector<int> emitIndices = {}) {
  std::shared_ptr<VeloxToSubstraitTypeConvertor> typeConvertor;
  std::unique_ptr<SubstraitReadRel> readRel =
      std::make_unique<SubstraitReadRel>();
  // create Virtual Table
  size_t numVectors = data.size();
  std::unique_ptr<SubstraitVirtualTable> virtualTable =
      std::make_unique<SubstraitVirtualTable>();
  // Construct the expression converter.
  auto exprConvertor =
      std::make_shared<VeloxToSubstraitExprConvertor>(extensionCollector);
  VELOX_CHECK(numVectors > 0, "No data provided");

  // here we process the required information to create a RowTypePtr to
  // update the ReadRel schema

  // first vector
  const auto& zeroRowVector = data.at(0);
  // The column number of the row data.
  int64_t numColumns = zeroRowVector->childrenSize();
  std::vector<std::string> names(numColumns);
  std::vector<TypePtr> types(numColumns);
  for (int64_t row = 0; row < numVectors; row++) {
    // The row data.
    ::substrait::Expression_Literal_Struct* litValue =
        virtualTable->add_values();

    const auto& rowVector = data.at(row);

    for (int64_t column = 0; column < numColumns; column++) {
      std::unique_ptr<SubstraitExprLit> substraitField =
          std::make_unique<SubstraitExprLit>();

      const VectorPtr& child = rowVector->childAt(column);
      if (row == 0) {
        names[column] = "f" + std::to_string(row);
        types[column] = child->type();
      }
      substraitField->MergeFrom(exprConvertor->toSubstraitExpr(
          arena, std::make_shared<core::ConstantTypedExpr>(child), litValue));
    }
  }

  RowTypePtr rowPtr = ROW(std::move(names), std::move(types));
  readRel->mutable_base_schema()->MergeFrom(
      typeConvertor->toSubstraitNamedStruct(arena, std::move(rowPtr)));
  readRel->mutable_virtual_table()->MergeFrom(*virtualTable.release());
  if (emitIndices.size() > 0) {
    readRel->set_allocated_common(MakeRelCommon(emitIndices).release());
  } else {
    readRel->mutable_common()->mutable_direct();
  }

  return readRel;
}

void SubstraitTestPlanBuilder::ValidateReadRelWithEmit(
    const std::vector<RowVectorPtr> data,
    const std::vector<RowVectorPtr>& expected,
    const std::vector<int> emitIndices) {
  google::protobuf::Arena arena;
  // make PlanRel
  std::unique_ptr<SubstraitPlanRel> planRel =
      std::make_unique<SubstraitPlanRel>();
  // make Rel
  std::unique_ptr<::substrait::Rel> rel = std::make_unique<::substrait::Rel>();
  // make ReadRel
  auto readRel = MakeReadRel(extensionCollector(), arena, data, emitIndices);
  rel->mutable_read()->MergeFrom(*readRel);
  // add rel to PlanRel
  planRel->mutable_rel()->MergeFrom(*rel);
  std::unique_ptr<SubstraitPlan> substraitPlan =
      std::make_unique<SubstraitPlan>();
  substraitPlan->mutable_relations()->AddAllocated(planRel.release());
  // add Extensions
  extensionCollector()->addExtensionsToPlan(substraitPlan.get());
  // convert Substrait plan to Velox Plan
  auto veloxPlan = planConverter()->toVeloxPlan(*substraitPlan);
  // assert Results
  AssertQueryBuilder(veloxPlan).assertResults(expected);
}

} // namespace facebook::velox::substrait
