/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

/// Converts a single partition field to a typed expression.
///
/// Builds an expression tree for one partition transform. Identity transforms
/// become FieldAccessTypedExpr, while other transforms (bucket, truncate,
/// year, month, day, hour) become CallTypedExpr with appropriate function
/// names and parameters.
///
/// @param field Partition field containing transform type, source column
/// type, and optional parameter (e.g., bucket count, truncate width).
/// @param inputFieldName Name of the source column in the input RowVector.
/// @param icebergFuncPrefix Prefix of iceberg transform function names.
/// @return Typed expression representing the transform.
core::TypedExprPtr toExpression(
    const IcebergPartitionSpec::Field& field,
    const std::string& inputFieldName,
    const std::string& icebergFuncPrefix) {
  // For identity transform, just return a field access expression.
  if (field.transformType == TransformType::kIdentity) {
    return std::make_shared<core::FieldAccessTypedExpr>(
        field.type, inputFieldName);
  }

  // For other transforms, build a CallTypedExpr with the appropriate function.
  std::string functionName;
  switch (field.transformType) {
    case TransformType::kBucket:
      functionName = icebergFuncPrefix + "bucket";
      break;
    case TransformType::kTruncate:
      functionName = icebergFuncPrefix + "truncate";
      break;
    case TransformType::kYear:
      functionName = icebergFuncPrefix + "years";
      break;
    case TransformType::kMonth:
      functionName = icebergFuncPrefix + "months";
      break;
    case TransformType::kDay:
      functionName = icebergFuncPrefix + "days";
      break;
    case TransformType::kHour:
      functionName = icebergFuncPrefix + "hours";
      break;
    case TransformType::kIdentity:
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
      field.resultType(), std::move(exprArgs), functionName);
}

} // namespace

std::vector<core::TypedExprPtr> TransformExprBuilder::toExpressions(
    const IcebergPartitionSpecPtr& partitionSpec,
    const std::vector<column_index_t>& partitionChannels,
    const RowTypePtr& inputType,
    const std::string& icebergFuncPrefix) {
  VELOX_CHECK_EQ(
      partitionSpec->fields.size(),
      partitionChannels.size(),
      "Number of partition fields must match number of partition channels");

  const auto numTransforms = partitionChannels.size();
  std::vector<core::TypedExprPtr> transformExprs;
  transformExprs.reserve(numTransforms);

  for (auto i = 0; i < numTransforms; i++) {
    const auto channel = partitionChannels[i];
    transformExprs.emplace_back(toExpression(
        partitionSpec->fields.at(i),
        inputType->nameOf(channel),
        icebergFuncPrefix));
  }

  return transformExprs;
}

} // namespace facebook::velox::connector::hive::iceberg
