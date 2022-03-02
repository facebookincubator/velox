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

#include "SubstraitToVeloxExpr.h"
#include "TypeUtils.h"

namespace facebook::velox::substrait {

std::shared_ptr<const core::FieldAccessTypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::FieldReference& sField,
    const int32_t& inputPlanNodeId) {
  auto typeCase = sField.reference_type_case();
  switch (typeCase) {
    case ::substrait::Expression::FieldReference::ReferenceTypeCase::
        kDirectReference: {
      auto dRef = sField.direct_reference();
      int32_t colIdx = subParser_->parseReferenceSegment(dRef);
      auto fieldName = subParser_->makeNodeName(inputPlanNodeId, colIdx);
      // TODO: Get the input type and support different types here.
      return std::make_shared<const core::FieldAccessTypedExpr>(
          DOUBLE(), fieldName);
    }
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Reference '{}'", typeCase);
  }
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::ScalarFunction& sFunc,
    const int32_t& inputPlanNodeId) {
  std::vector<std::shared_ptr<const core::ITypedExpr>> params;
  for (const auto& sArg : sFunc.args()) {
    auto expr = toVeloxExpr(sArg, inputPlanNodeId);
    params.push_back(expr);
  }
  auto functionId = sFunc.function_reference();
  auto veloxFunction = subParser_->findVeloxFunction(functionMap_, functionId);
  auto subType = subParser_->parseType(sFunc.output_type());
  auto veloxType = toVeloxType(subType->type);
  return std::make_shared<const core::CallTypedExpr>(
      veloxType, std::move(params), veloxFunction);
}

std::shared_ptr<const core::ConstantTypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Literal& sLit) {
  auto typeCase = sLit.literal_type_case();
  switch (typeCase) {
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp64: {
      return std::make_shared<core::ConstantTypedExpr>(sLit.fp64());
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kBoolean: {
      return std::make_shared<core::ConstantTypedExpr>(sLit.boolean());
    }
    default:
      VELOX_NYI(
          "Substrait conversion not supported for type case '{}'", typeCase);
  }
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression& sExpr,
    const int32_t& inputPlanNodeId) {
  std::shared_ptr<const core::ITypedExpr> veloxExpr;
  auto typeCase = sExpr.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kLiteral: {
      veloxExpr = toVeloxExpr(sExpr.literal());
      break;
    }
    case ::substrait::Expression::RexTypeCase::kScalarFunction: {
      veloxExpr = toVeloxExpr(sExpr.scalar_function(), inputPlanNodeId);
      break;
    }
    case ::substrait::Expression::RexTypeCase::kSelection: {
      veloxExpr = toVeloxExpr(sExpr.selection(), inputPlanNodeId);
      break;
    }
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Expression '{}'", typeCase);
  }
  return veloxExpr;
}

// TODO: Support different types here.
class SubstraitVeloxExprConverter::FilterInfo {
 public:
  FilterInfo() {}
  // Used to set the left bound.
  void setLeft(double left, bool isExclusive) {
    left_ = left;
    leftExclusive_ = isExclusive;
    if (!isInitialized_) {
      isInitialized_ = true;
    }
  }

  // Used to set the right bound.
  void setRight(double right, bool isExclusive) {
    right_ = right;
    rightExclusive_ = isExclusive;
    if (!isInitialized_) {
      isInitialized_ = true;
    }
  }

  // Will fordis Null value if called once.
  void forbidsNull() {
    nullAllowed_ = false;
    if (!isInitialized_) {
      isInitialized_ = true;
    }
  }

  // Return the initialization status.
  bool isInitialized() {
    return isInitialized_ ? true : false;
  }

  // The left bound.
  std::optional<double> left_ = std::nullopt;
  // The right bound.
  std::optional<double> right_ = std::nullopt;
  // The Null allowing.
  bool nullAllowed_ = true;
  // If true, left bound will be exclusive.
  bool leftExclusive_ = false;
  // If true, right bound will be exclusive.
  bool rightExclusive_ = false;

 private:
  bool isInitialized_ = false;
};

void SubstraitVeloxExprConverter::getFlatConditions(
    const ::substrait::Expression& sFilter,
    std::vector<::substrait::Expression_ScalarFunction>* scalarFunctions) {
  auto typeCase = sFilter.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kScalarFunction: {
      auto sFunc = sFilter.scalar_function();
      auto filterName = subParser_->findSubstraitFunction(
          functionMap_, sFunc.function_reference());
      // TODO: Only AND relation is supported here.
      if (filterName == "AND") {
        for (const auto& sCondition : sFunc.args()) {
          getFlatConditions(sCondition, scalarFunctions);
        }
      } else {
        (*scalarFunctions).push_back(sFunc);
      }
      break;
    }
    default:
      VELOX_NYI("GetFlatConditions not supported for type '{}'", typeCase);
  }
}

connector::hive::SubfieldFilters SubstraitVeloxExprConverter::toVeloxFilter(
    const std::vector<std::string>& inputNameList,
    const std::vector<TypePtr>& inputTypeList,
    const ::substrait::Expression& sFilter) {
  connector::hive::SubfieldFilters filters;
  std::unordered_map<int, std::shared_ptr<FilterInfo>> colInfoMap;
  for (int idx = 0; idx < inputNameList.size(); idx++) {
    auto filterInfo = std::make_shared<FilterInfo>();
    colInfoMap[idx] = filterInfo;
  }
  std::vector<::substrait::Expression_ScalarFunction> scalarFunctions;
  getFlatConditions(sFilter, &scalarFunctions);
  for (const auto& scalarFunction : scalarFunctions) {
    auto filterName = subParser_->findSubstraitFunction(
        functionMap_, scalarFunction.function_reference());
    int32_t colIdx;
    // TODO: Add different types' support here.
    double val;
    for (auto& param : scalarFunction.args()) {
      auto typeCase = param.rex_type_case();
      switch (typeCase) {
        case ::substrait::Expression::RexTypeCase::kSelection: {
          auto sel = param.selection();
          // TODO: Only direct reference is considered here.
          auto dRef = sel.direct_reference();
          colIdx = subParser_->parseReferenceSegment(dRef);
          break;
        }
        case ::substrait::Expression::RexTypeCase::kLiteral: {
          auto sLit = param.literal();
          // TODO: Only double is considered here.
          val = sLit.fp64();
          break;
        }
        default:
          VELOX_NYI(
              "Substrait conversion not supported for arg type '{}'", typeCase);
      }
    }
    if (filterName == "IS_NOT_NULL") {
      colInfoMap[colIdx]->forbidsNull();
    } else if (filterName == "GREATER_THAN_OR_EQUAL") {
      colInfoMap[colIdx]->setLeft(val, false);
    } else if (filterName == "GREATER_THAN") {
      colInfoMap[colIdx]->setLeft(val, true);
    } else if (filterName == "LESS_THAN_OR_EQUAL") {
      colInfoMap[colIdx]->setRight(val, false);
    } else if (filterName == "LESS_THAN") {
      colInfoMap[colIdx]->setRight(val, true);
    } else {
      VELOX_NYI(
          "Substrait conversion not supported for filter name '{}'",
          filterName);
    }
  }
  // Construct the Filters.
  for (int idx = 0; idx < inputNameList.size(); idx++) {
    auto filterInfo = colInfoMap[idx];
    double leftBound = 0;
    double rightBound = 0;
    bool leftUnbounded = true;
    bool rightUnbounded = true;
    bool leftExclusive = false;
    bool rightExclusive = false;
    if (filterInfo->isInitialized()) {
      if (filterInfo->left_) {
        leftUnbounded = false;
        leftBound = filterInfo->left_.value();
        leftExclusive = filterInfo->leftExclusive_;
      }
      if (filterInfo->right_) {
        rightUnbounded = false;
        rightBound = filterInfo->right_.value();
        rightExclusive = filterInfo->rightExclusive_;
      }
      bool nullAllowed = filterInfo->nullAllowed_;
      filters[common::Subfield(inputNameList[idx])] =
          std::make_unique<common::DoubleRange>(
              leftBound,
              leftUnbounded,
              leftExclusive,
              rightBound,
              rightUnbounded,
              rightExclusive,
              nullAllowed);
    }
  }
  return filters;
}

} // namespace facebook::velox::substrait
