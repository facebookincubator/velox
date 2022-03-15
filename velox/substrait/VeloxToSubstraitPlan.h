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

#include "velox/substrait/proto/substrait/algebra.pb.h"
#include "velox/substrait/proto/substrait/plan.pb.h"

#include "connectors/hive/HiveConnector.h"
#include "connectors/hive/HivePartitionFunction.h"
#include "core/PlanNode.h"
#include "exec/HashPartitionFunction.h"
#include "exec/RoundRobinPartitionFunction.h"
#include "type/Type.h"
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"

#include "VeloxToSubstraitExpr.h"

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

  void transformVFilter(
      std::shared_ptr<const FilterNode> vFilterNode,
      ::substrait::FilterRel* sFilterRel);

  void transformVValuesNode(
      std::shared_ptr<const ValuesNode> vValuesNode,
      ::substrait::ReadRel* sReadRel);

  void transformVProjNode(
      std::shared_ptr<const ProjectNode> vProjNode,
      ::substrait::ProjectRel* sProjRel);

  void transformVAggregateNode(
      std::shared_ptr<const AggregationNode> vAggNode,
      ::substrait::AggregateRel* sAggRel);

  void transformVOrderBy(
      std::shared_ptr<const OrderByNode> vOrderbyNode,
      ::substrait::SortRel* sSortRel);

  VeloxToSubstraitExprConvertor v2SExprConvertor_;
  VeloxToSubstraitTypeConvertor v2STypeConvertor_;
};
} // namespace facebook::velox::substrait
