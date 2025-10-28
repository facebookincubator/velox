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
#include "velox/connectors/hive/iceberg/TransformExprBuilder.h"
#include "velox/core/Expressions.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

constexpr char const* kBucketFunction{"iceberg_bucket"};
constexpr char const* kTruncateFunction{"iceberg_truncate"};
constexpr char const* kYearFunction{"iceberg_years"};
constexpr char const* kMonthFunction{"iceberg_months"};
constexpr char const* kDayFunction{"iceberg_days"};
constexpr char const* kHourFunction{"iceberg_hours"};

} // namespace

core::TypedExprPtr TransformExprBuilder::toExpression(
    const IcebergPartitionSpec::Field& field,
    const std::string& inputFieldName) {
  // For identity transform, just return a field access expression.
  if (field.transformType == TransformType::kIdentity) {
    return std::make_shared<core::FieldAccessTypedExpr>(
        field.type, inputFieldName);
  }

  // For other transforms, build a CallTypedExpr with the appropriate function.
  std::string functionName;
  TypePtr resultType;
  switch (field.transformType) {
    case TransformType::kBucket:
      functionName = kBucketFunction;
      resultType = INTEGER();
      break;
    case TransformType::kTruncate:
      functionName = kTruncateFunction;
      resultType = field.type;
      break;
    case TransformType::kYear:
      functionName = kYearFunction;
      resultType = INTEGER();
      break;
    case TransformType::kMonth:
      functionName = kMonthFunction;
      resultType = INTEGER();
      break;
    case TransformType::kDay:
      functionName = kDayFunction;
      resultType = DATE();
      break;
    default:
      functionName = kHourFunction;
      resultType = INTEGER();
      break;
  }

  // Build the expression arguments.
  std::vector<core::TypedExprPtr> exprArgs;
  if (field.parameter.has_value()) {
    exprArgs.emplace_back(
        std::make_shared<core::ConstantTypedExpr>(
            INTEGER(), Variant(field.parameter.value())));
  }
  exprArgs.emplace_back(
      std::make_shared<core::FieldAccessTypedExpr>(field.type, inputFieldName));

  return std::make_shared<core::CallTypedExpr>(
      resultType, std::move(exprArgs), functionName);
}

std::vector<core::TypedExprPtr> TransformExprBuilder::toExpressions(
    const IcebergPartitionSpecPtr& partitionSpec,
    const std::vector<column_index_t>& partitionChannels,
    const RowTypePtr& inputType) {
  VELOX_CHECK_EQ(
      partitionSpec->fields.size(),
      partitionChannels.size(),
      "Number of partition fields must match number of partition channels");

  const auto numTransforms = partitionChannels.size();
  std::vector<core::TypedExprPtr> transformExprs;
  transformExprs.reserve(numTransforms);

  for (auto i = 0; i < numTransforms; i++) {
    const auto channel = partitionChannels[i];
    transformExprs.emplace_back(
        toExpression(partitionSpec->fields.at(i), inputType->nameOf(channel)));
  }

  return transformExprs;
}

} // namespace facebook::velox::connector::hive::iceberg
