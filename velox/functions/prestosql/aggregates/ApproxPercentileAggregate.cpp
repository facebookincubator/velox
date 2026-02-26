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
#include "velox/functions/prestosql/aggregates/ApproxPercentileAggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/aggregates/ApproxPercentileAggregateBase.h"

namespace facebook::velox::aggregate::prestosql {

using functions::aggregate::ApproxPercentileAggregateBase;
using Idx = functions::aggregate::ApproxPercentileIntermediateTypeChildIndex;

namespace {

/// Presto accuracy policy: accuracy is a double in (0, 1] representing
/// the error bound for the approximation.
struct PrestoAccuracyPolicy {
  static constexpr double kMissingNormalizedValue = -1;
  static constexpr double kDefaultAccuracy = kMissingNormalizedValue;

  static bool isDefaultAccuracy(double accuracy) {
    return accuracy == kMissingNormalizedValue;
  }

  template <typename T>
  static void setOnAccumulator(
      functions::aggregate::detail::KllSketchAccumulator<T>* accumulator,
      double accuracy) {
    if (accuracy != kMissingNormalizedValue) {
      accumulator->setAccuracy(accuracy);
    }
  }

  static void checkSetAccuracy(
      DecodedVector& decodedAccuracy,
      const SelectivityVector& rows,
      double& accuracy) {
    if (decodedAccuracy.isConstantMapping()) {
      VELOX_USER_CHECK(!decodedAccuracy.isNullAt(0), "Accuracy cannot be null");
      checkSetPrestoAccuracy(accuracy, decodedAccuracy.valueAt<double>(0));
    } else {
      rows.applyToSelected([&](auto row) {
        VELOX_USER_CHECK(
            !decodedAccuracy.isNullAt(row), "Accuracy cannot be null");
        const auto value = decodedAccuracy.valueAt<double>(row);
        if (accuracy == kMissingNormalizedValue) {
          checkSetPrestoAccuracy(accuracy, value);
        }
        VELOX_USER_CHECK_EQ(
            value,
            accuracy,
            "Accuracy argument must be constant for all input rows");
      });
    }
  }

  static void checkAndSetFromIntermediate(
      double& accuracy,
      double inputAccuracy) {
    checkSetPrestoAccuracy(accuracy, inputAccuracy);
  }

 private:
  static void checkSetPrestoAccuracy(double& accuracy, double inputAccuracy) {
    VELOX_USER_CHECK(
        0 < inputAccuracy && inputAccuracy <= 1,
        "Accuracy must be between 0 and 1");
    if (accuracy == kMissingNormalizedValue) {
      accuracy = inputAccuracy;
    } else {
      VELOX_USER_CHECK_EQ(
          inputAccuracy,
          accuracy,
          "Accuracy argument must be constant for all input rows");
    }
  }
};

template <typename T>
using ApproxPercentileAggregate = ApproxPercentileAggregateBase<
    T,
    /*kHasWeight=*/true,
    PrestoAccuracyPolicy>;

bool validPercentileType(const Type& type) {
  if (type.kind() == TypeKind::DOUBLE) {
    return true;
  }
  if (type.kind() != TypeKind::ARRAY) {
    return false;
  }
  return type.as<TypeKind::ARRAY>().elementType()->kind() == TypeKind::DOUBLE;
}

void addSignatures(
    const std::string& inputType,
    const std::string& percentileType,
    const std::string& returnType,
    std::vector<std::shared_ptr<exec::AggregateFunctionSignature>>&
        signatures) {
  auto intermediateType = fmt::format(
      "row(array(double), boolean, double, integer, bigint, {0}, {0}, array({0}), array(integer))",
      inputType);
  // (value, percentile)
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .returnType(returnType)
                           .intermediateType(intermediateType)
                           .argumentType(inputType)
                           .argumentType(percentileType)
                           .build());
  // (value, weight, percentile)
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .returnType(returnType)
                           .intermediateType(intermediateType)
                           .argumentType(inputType)
                           .argumentType("bigint")
                           .argumentType(percentileType)
                           .build());
  // (value, percentile, accuracy)
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .returnType(returnType)
                           .intermediateType(intermediateType)
                           .argumentType(inputType)
                           .argumentType(percentileType)
                           .argumentType("double")
                           .build());
  // (value, weight, percentile, accuracy)
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .returnType(returnType)
                           .intermediateType(intermediateType)
                           .argumentType(inputType)
                           .argumentType("bigint")
                           .argumentType(percentileType)
                           .argumentType("double")
                           .build());
}

} // namespace

void registerApproxPercentileAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  for (const auto& inputType :
       {"tinyint", "smallint", "integer", "bigint", "real", "double"}) {
    addSignatures(inputType, "double", inputType, signatures);
    addSignatures(
        inputType,
        "array(double)",
        fmt::format("array({})", inputType),
        signatures);
  }
  exec::registerAggregateFunction(
      names,
      std::move(signatures),
      [names](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        const std::string& name = names.front();
        const auto isRawInput = exec::isRawInput(step);
        const auto hasWeight =
            argTypes.size() >= 2 && argTypes[1]->kind() == TypeKind::BIGINT;
        const bool hasAccuracy = argTypes.size() == (hasWeight ? 4 : 3);
        const auto fixedRandomSeed =
            config.debugAggregationApproxPercentileFixedRandomSeed();

        if (isRawInput) {
          VELOX_USER_CHECK_EQ(
              argTypes.size(),
              2 + hasWeight + hasAccuracy,
              "Wrong number of arguments passed to {}",
              name);
          if (hasWeight) {
            VELOX_USER_CHECK_EQ(
                argTypes[1]->kind(),
                TypeKind::BIGINT,
                "The type of the weight argument of {} must be BIGINT",
                name);
          }
          if (hasAccuracy) {
            VELOX_USER_CHECK_EQ(
                argTypes.back()->kind(),
                TypeKind::DOUBLE,
                "The type of the accuracy argument of {} must be DOUBLE",
                name);
          }
          VELOX_USER_CHECK(
              validPercentileType(*argTypes[argTypes.size() - 1 - hasAccuracy]),
              "The type of the percentile argument of {} must be DOUBLE or ARRAY(DOUBLE)",
              name);
        } else {
          VELOX_USER_CHECK_EQ(
              argTypes.size(),
              1,
              "The type of partial result for {} must be ROW",
              name);
          VELOX_USER_CHECK_EQ(
              argTypes[0]->kind(),
              TypeKind::ROW,
              "The type of partial result for {} must be ROW",
              name);
        }

        TypePtr type;
        if (!isRawInput && exec::isPartialOutput(step)) {
          type = argTypes[0]->asRow().childAt(static_cast<int>(Idx::kMinValue));
        } else if (isRawInput) {
          type = argTypes[0];
        } else if (resultType->isArray()) {
          type = resultType->as<TypeKind::ARRAY>().elementType();
        } else {
          type = resultType;
        }

        switch (type->kind()) {
          case TypeKind::TINYINT:
            return std::make_unique<ApproxPercentileAggregate<int8_t>>(
                hasWeight, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::SMALLINT:
            return std::make_unique<ApproxPercentileAggregate<int16_t>>(
                hasWeight, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::INTEGER:
            return std::make_unique<ApproxPercentileAggregate<int32_t>>(
                hasWeight, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::BIGINT:
            return std::make_unique<ApproxPercentileAggregate<int64_t>>(
                hasWeight, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::REAL:
            return std::make_unique<ApproxPercentileAggregate<float>>(
                hasWeight, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::DOUBLE:
            return std::make_unique<ApproxPercentileAggregate<double>>(
                hasWeight, hasAccuracy, resultType, fixedRandomSeed);
          default:
            VELOX_USER_FAIL(
                "Unsupported input type for {} aggregation {}",
                name,
                type->toString());
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
