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

#include "SubstraitUtils.h"
#include "velox/connectors/hive/HiveConnector.h"

namespace facebook::velox::substrait {

/// This class is used to convert Substrait representations to Velox
/// expressions.
class SubstraitVeloxExprConverter {
 public:
  /// subParser: A Substrait parser used to convert Substrait representations
  /// into recognizable representations. functionMap: A pre-constructed map
  /// storing the relations between the function id and the function name.
  SubstraitVeloxExprConverter(
      const std::shared_ptr<SubstraitParser>& subParser,
      const std::unordered_map<uint64_t, std::string>& functionMap)
      : subParser_(subParser), functionMap_(functionMap) {}

  /// Used to convert Substrait Field into Velox Field Expression.
  std::shared_ptr<const core::FieldAccessTypedExpr> toVeloxExpr(
      const ::substrait::Expression::FieldReference& sField,
      const int32_t& inputPlanNodeId);

  /// Used to convert Substrait ScalarFunction into Velox Expression.
  std::shared_ptr<const core::ITypedExpr> toVeloxExpr(
      const ::substrait::Expression::ScalarFunction& sFunc,
      const int32_t& inputPlanNodeId);

  /// Used to convert Substrait Literal into Velox Expression.
  std::shared_ptr<const core::ConstantTypedExpr> toVeloxExpr(
      const ::substrait::Expression::Literal& sLit);

  /// Used to convert Substrait Expression into Velox Expression.
  std::shared_ptr<const core::ITypedExpr> toVeloxExpr(
      const ::substrait::Expression& sExpr,
      const int32_t& inputPlanNodeId);

  /// Used to convert Substrait Filter into Velox SubfieldFilters.
  connector::hive::SubfieldFilters toVeloxFilter(
      const std::vector<std::string>& inputNameList,
      const std::vector<TypePtr>& inputTypeList,
      const ::substrait::Expression& sFilter);

 private:
  /// Multiple conditions are connected to a binary tree structure with
  /// the relation key words, including AND, OR, and etc. Currently, only
  /// AND is supported. This function is used to extract all the Substrait
  /// conditions in the binary tree structure into a vector.
  void getFlatConditions(
      const ::substrait::Expression& sFilter,
      std::vector<::substrait::Expression_ScalarFunction>* scalarFunctions);

  /// The Substrait parser used to convert Substrait representations into
  /// recognizable representations.
  std::shared_ptr<SubstraitParser> subParser_;
  /// The map storing the relations between the function id and the function
  /// name.
  std::unordered_map<uint64_t, std::string> functionMap_;
  /// This class contains the needed infos for Filter Pushdown.
  class FilterInfo;
};

} // namespace facebook::velox::substrait
