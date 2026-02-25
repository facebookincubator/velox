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

#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/aggregates/ApproxPercentileAggregateBase.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"

namespace facebook::velox::functions::aggregate::sparksql {

using Idx = ApproxPercentileIntermediateTypeChildIndex;

namespace {

/// Spark accuracy policy: accuracy is an int32 representing the reciprocal
/// of epsilon (e.g., 10000 means epsilon = 1/10000).
struct SparkAccuracyPolicy {
  static constexpr double kDefaultAccuracy = 10000;

  static bool isDefaultAccuracy(double /*accuracy*/) {
    // Spark always has a valid accuracy (defaults to 10000), never "missing".
    return false;
  }

  template <typename T>
  static void setOnAccumulator(
      KllSketchAccumulator<T>* accumulator,
      double accuracy) {
    // Spark accuracy is the reciprocal of epsilon.
    double epsilon = 1.0 / static_cast<double>(static_cast<int32_t>(accuracy));
    accumulator->setAccuracy(epsilon);
  }

  static void checkSetAccuracy(
      DecodedVector& decodedAccuracy,
      const SelectivityVector& rows,
      double& accuracy) {
    if (decodedAccuracy.isConstantMapping()) {
      VELOX_USER_CHECK(!decodedAccuracy.isNullAt(0), "Accuracy cannot be null");
      checkAndSet(accuracy, decodedAccuracy.valueAt<int32_t>(0));
    } else {
      rows.applyToSelected([&](auto row) {
        VELOX_USER_CHECK(
            !decodedAccuracy.isNullAt(row), "Accuracy cannot be null");
        const auto currentAccuracy = decodedAccuracy.valueAt<int32_t>(row);
        if (accuracy == kDefaultAccuracy) {
          checkAndSet(accuracy, currentAccuracy);
        }
        VELOX_USER_CHECK_EQ(
            currentAccuracy,
            static_cast<int32_t>(accuracy),
            "Accuracy argument must be constant");
      });
    }
  }

  static void checkAndSetFromIntermediate(
      double& accuracy,
      double inputAccuracy) {
    checkAndSet(accuracy, static_cast<int32_t>(inputAccuracy));
  }

 private:
  static void checkAndSet(double& accuracy, int32_t inputAccuracy) {
    VELOX_USER_CHECK(
        inputAccuracy > 0 &&
            inputAccuracy <= std::numeric_limits<int32_t>::max(),
        "Accuracy must be greater than 0 and less than or equal to "
        "Int32.MaxValue, got {}",
        inputAccuracy);
    accuracy = static_cast<double>(inputAccuracy);
  }
};

template <typename T>
using ApproxPercentileAggregate = ApproxPercentileAggregateBase<
    T,
    /*kHasWeight=*/false,
    SparkAccuracyPolicy>;

void addSignatures(
    const std::string& inputType,
    std::vector<std::shared_ptr<exec::AggregateFunctionSignature>>&
        signatures) {
  std::string intermediateType = fmt::format(
      "row(array(double), boolean, double, integer, bigint, {0}, {0}, array({0}), array(integer))",
      inputType);

  // Signature 1: approx_percentile(T, double) -> T (single percentile, default
  // accuracy).
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType(inputType)
          .intermediateType(intermediateType)
          .argumentType(inputType)
          .argumentType("double")
          .build());
  // Signature 2: approx_percentile(T, double, integer) -> T (single percentile,
  // explicit accuracy).
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType(inputType)
          .intermediateType(intermediateType)
          .argumentType(inputType)
          .argumentType("double")
          .argumentType("integer")
          .build());
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType(inputType)
          .intermediateType(intermediateType)
          .argumentType(inputType)
          .argumentType("double")
          .argumentType("bigint")
          .build());
  // Signature 3: approx_percentile(T, array(double)) -> array(T) (percentile
  // array, default accuracy).
  std::string arrayReturnType = fmt::format("array({})", inputType);
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType(arrayReturnType)
          .intermediateType(intermediateType)
          .argumentType(inputType)
          .argumentType("array(double)")
          .build());
  // Signature 4: approx_percentile(T, array(double), integer) -> array(T)
  // (percentile array, explicit accuracy).
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType(arrayReturnType)
          .intermediateType(intermediateType)
          .argumentType(inputType)
          .argumentType("array(double)")
          .argumentType("integer")
          .build());
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType(arrayReturnType)
          .intermediateType(intermediateType)
          .argumentType(inputType)
          .argumentType("array(double)")
          .argumentType("bigint")
          .build());
}

} // namespace

exec::AggregateRegistrationResult registerApproxPercentileAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  const std::vector<std::string> kSupportedInputTypes = {
      "tinyint", "smallint", "integer", "bigint", "real", "double"};
  for (const auto& inputType : kSupportedInputTypes) {
    addSignatures(inputType, signatures);
  }

  auto functionName = prefix + velox::aggregate::kApproxPercentile;
  return exec::registerAggregateFunction(
      functionName,
      std::move(signatures),
      [functionName = functionName](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        const auto isRawInput = exec::isRawInput(step);
        const auto fixedRandomSeed =
            config.debugAggregationApproxPercentileFixedRandomSeed();
        const bool hasAccuracy = isRawInput ? (argTypes.size() == 3) : false;

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
                /*hasWeight=*/false, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::SMALLINT:
            return std::make_unique<ApproxPercentileAggregate<int16_t>>(
                /*hasWeight=*/false, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::INTEGER:
            return std::make_unique<ApproxPercentileAggregate<int32_t>>(
                /*hasWeight=*/false, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::BIGINT:
            return std::make_unique<ApproxPercentileAggregate<int64_t>>(
                /*hasWeight=*/false, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::REAL:
            return std::make_unique<ApproxPercentileAggregate<float>>(
                /*hasWeight=*/false, hasAccuracy, resultType, fixedRandomSeed);
          case TypeKind::DOUBLE:
            return std::make_unique<ApproxPercentileAggregate<double>>(
                /*hasWeight=*/false, hasAccuracy, resultType, fixedRandomSeed);
          default:
            VELOX_USER_FAIL(
                "Unsupported input type for {}: {}",
                functionName,
                argTypes[0]->toString());
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
