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

#pragma once

#include <string>
#include <typeinfo>

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"
#include "velox/type/Type.h"

#include "velox/substrait/VeloxToSubstraitExpr.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"
#include "velox/substrait/proto/substrait/plan.pb.h"

using namespace facebook::velox::core;

namespace facebook::velox::substrait {

class VeloxToSubstraitPlanConvertor {
 public:
  void veloxToSubstraitIR(
      std::shared_ptr<const PlanNode> vPlan,
      ::substrait::Plan& sPlan);

 private:
  void veloxToSubstraitIR(
      std::shared_ptr<const PlanNode> vPlanNode,
      ::substrait::Rel* sRel);

  void transformVeloxFilter(
      std::shared_ptr<const FilterNode> vFilterNode,
      ::substrait::FilterRel* sFilterRel);

  void transformVeloxValuesNode(
      std::shared_ptr<const ValuesNode> vValuesNode,
      ::substrait::ReadRel* sReadRel);

  void transformVeloxProjNode(
      std::shared_ptr<const ProjectNode> vProjNode,
      ::substrait::ProjectRel* sProjRel);

  void transformVeloxAggregateNode(
      std::shared_ptr<const AggregationNode> vAggNode,
      ::substrait::AggregateRel* sAggRel);

  void transformVeloxOrderBy(
      std::shared_ptr<const OrderByNode> vOrderbyNode,
      ::substrait::SortRel* sSortRel);

  VeloxToSubstraitExprConvertor v2SExprConvertor_;
  VeloxToSubstraitTypeConvertor v2STypeConvertor_;
};
} // namespace facebook::velox::substrait
