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
#include "velox/functions/prestosql/aggregates/MergeAggregate.h"
#include "velox/common/base/Exceptions.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/functions/prestosql/aggregates/HyperLogLogAggregate.h"
#include "velox/functions/prestosql/aggregates/MergeKHyperLogLogAggregate.h"
#include "velox/functions/prestosql/aggregates/MergeQDigestAggregate.h"
#include "velox/functions/prestosql/aggregates/MergeTDigestAggregate.h"
#include "velox/functions/prestosql/aggregates/SfmSketchAggregate.h"
#include "velox/functions/prestosql/types/HyperLogLogRegistration.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/functions/prestosql/types/KHyperLogLogRegistration.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"
#include "velox/functions/prestosql/types/QDigestRegistration.h"
#include "velox/functions/prestosql/types/SfmSketchRegistration.h"
#include "velox/functions/prestosql/types/SfmSketchType.h"
#include "velox/functions/prestosql/types/TDigestRegistration.h"
#include "velox/functions/prestosql/types/TDigestType.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

std::vector<exec::AggregateRegistrationResult> registerMerge(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite,
    double defaultError,
    const std::vector<MergeSketchType>& additionalSketchTypes) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  auto inputTypes = std::vector<std::string>{
      "hyperloglog",
      "khyperloglog",
      "sfmsketch",
      "tdigest(double)",
      "qdigest(bigint)",
      "qdigest(real)",
      "qdigest(double)"};
  signatures.reserve(inputTypes.size() + additionalSketchTypes.size());
  for (const auto& inputType : inputTypes) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(inputType)
            .intermediateType("varbinary")
            .argumentType(inputType)
            .build());
  }
  // Built-in sketch types an injected type must not collide with.
  const std::vector<TypePtr> builtinTypes = {
      HYPERLOGLOG(),
      KHYPERLOGLOG(),
      SFMSKETCH(),
      TDIGEST(DOUBLE()),
      QDIGEST(BIGINT()),
      QDIGEST(REAL()),
      QDIGEST(DOUBLE()),
  };
  std::vector<TypePtr> seenSketchTypes;
  seenSketchTypes.reserve(additionalSketchTypes.size());
  for (const auto& sketch : additionalSketchTypes) {
    VELOX_CHECK_NOT_NULL(sketch.signature);
    VELOX_CHECK_NOT_NULL(sketch.type);
    VELOX_CHECK(sketch.factory);
    VELOX_CHECK(
        !sketch.signature->argumentTypes().empty(),
        "merge() sketch signature must have at least one argument type: {}",
        sketch.type->toString());
    // Signature's argument type must resolve to 'type', matched at dispatch.
    const auto argType = exec::SignatureBinder::tryResolveType(
        sketch.signature->argumentTypes()[0], {}, {});
    VELOX_CHECK(
        argType != nullptr && *argType == *sketch.type,
        "merge() sketch signature argument type must match its type: {}",
        sketch.type->toString());
    // Signature's return type must resolve to 'type', matched at dispatch.
    const auto returnType = exec::SignatureBinder::tryResolveType(
        sketch.signature->returnType(), {}, {});
    VELOX_CHECK(
        returnType != nullptr && *returnType == *sketch.type,
        "merge() sketch signature return type must match its type: {}",
        sketch.type->toString());
    // An extension type must not shadow a built-in dispatched below it.
    for (const auto& builtinType : builtinTypes) {
      VELOX_CHECK(
          *sketch.type != *builtinType,
          "merge() sketch type collides with a built-in sketch type: {}",
          sketch.type->toString());
    }
    // Injected extension types must be unique among themselves.
    for (const auto& seenType : seenSketchTypes) {
      VELOX_CHECK(
          *sketch.type != *seenType,
          "merge() sketch type is registered more than once: {}",
          sketch.type->toString());
    }
    seenSketchTypes.push_back(sketch.type);
    signatures.push_back(sketch.signature);
  }
  bool hllAsRawInput = true;
  bool hllAsFinalResult = true;
  return exec::registerAggregateFunction(
      names,
      signatures,
      [hllAsFinalResult, hllAsRawInput, defaultError, additionalSketchTypes](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        for (const auto& sketch : additionalSketchTypes) {
          if (*argTypes[0] == *sketch.type) {
            return sketch.factory(resultType);
          }
        }
        if (*argTypes[0] == *TDIGEST(DOUBLE())) {
          return createMergeTDigestAggregate(resultType);
        }
        if (*argTypes[0] == *QDIGEST(BIGINT()) ||
            *argTypes[0] == *QDIGEST(REAL()) ||
            *argTypes[0] == *QDIGEST(DOUBLE())) {
          return createMergeQDigestAggregate(resultType, argTypes[0]);
        }
        if (argTypes[0] == SFMSKETCH()) {
          if (exec::isPartialOutput(step)) {
            return std::make_unique<SfmSketchAggregate<true, false, true>>(
                VARBINARY());
          }
          return std::make_unique<SfmSketchAggregate<true, false, true>>(
              resultType);
        }
        if (argTypes[0] == KHYPERLOGLOG()) {
          return std::make_unique<MergeKHyperLogLogAggregate>(resultType);
        }
        if (argTypes[0]->isUnknown()) {
          return std::make_unique<HyperLogLogAggregate<UnknownValue, true>>(
              resultType, hllAsRawInput, defaultError);
        }
        if (exec::isPartialInput(step) && argTypes[0]->isTinyint()) {
          // This condition only applies to approx_distinct(boolean).
          return std::make_unique<HyperLogLogAggregate<bool, false>>(
              resultType, hllAsRawInput, defaultError);
        }
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createHyperLogLogAggregate,
            argTypes[0]->kind(),
            resultType,
            hllAsFinalResult,
            hllAsRawInput,
            defaultError);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerMergeAggregate(
    const std::vector<std::string>& names,
    bool /* withCompanionFunctions */,
    bool overwrite,
    const std::vector<MergeSketchType>& additionalSketchTypes) {
  registerSfmSketchType();
  registerHyperLogLogType();
  registerKHyperLogLogType();
  registerTDigestType();
  registerQDigestType();
  // merge is companion function for approx_distinct. Don't register companion
  // functions for it.
  registerMerge(
      names,
      false,
      overwrite,
      common::hll::kDefaultApproxSetStandardError,
      additionalSketchTypes);
}

} // namespace facebook::velox::aggregate::prestosql
