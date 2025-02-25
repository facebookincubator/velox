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

#include "velox/expression/fuzzer/ArgValuesGenerators.h"

#include "velox/common/fuzzer/ConstrainedGenerators.h"
#include "velox/common/fuzzer/Utils.h"
#include "velox/core/Expressions.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::fuzzer {

namespace {

const std::vector<TypePtr>& jsonScalarTypes() {
  static const std::vector<TypePtr> kScalarTypes{
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      VARBINARY(),
      TIMESTAMP(),
  };
  return kScalarTypes;
}

} // namespace

std::vector<core::TypedExprPtr> JsonParseArgValuesGenerator::generate(
    const CallableSignature& signature,
    const VectorFuzzer::Options& options,
    FuzzerGenerator& rng,
    ExpressionFuzzerState& state) {
  VELOX_CHECK_EQ(signature.args.size(), 1);
  populateInputTypesAndNames(signature, state);

  // Populate state.customInputGenerators_ for the argument that requires custom
  // generation. A nullptr should be added at state.customInputGenerators_[i] if
  // the i-th argument does not require custom input generation.
  const auto representedType =
      facebook::velox::randType(rng, jsonScalarTypes(), 3);
  const auto seed = rand<uint32_t>(rng);
  const auto nullRatio = options.nullRatio;
  state.customInputGenerators_.emplace_back(
      std::make_shared<fuzzer::JsonInputGenerator>(
          seed,
          signature.args[0],
          nullRatio,
          fuzzer::getRandomInputGenerator(seed, representedType, nullRatio),
          true));

  // Populate inputExpressions_ for the argument that requires custom
  // generation. A nullptr should be added at inputExpressions[i] if the i-th
  // argument does not require custom input generation.
  std::vector<core::TypedExprPtr> inputExpressions{
      signature.args.size(), nullptr};
  inputExpressions[0] = std::make_shared<core::FieldAccessTypedExpr>(
      signature.args[0], state.inputRowNames_.back());
  return inputExpressions;
}

std::vector<core::TypedExprPtr> JsonExtractArgValuesGenerator::generate(
    const CallableSignature& signature,
    const VectorFuzzer::Options& options,
    FuzzerGenerator& rng,
    ExpressionFuzzerState& state) {
  VELOX_CHECK_EQ(signature.args.size(), 2);
  std::vector<core::TypedExprPtr> inputExpressions;

  for (auto i = 0; i < signature.args.size(); ++i) {
    state.inputRowTypes_.emplace_back(signature.args[i]);
    state.inputRowNames_.emplace_back(
        fmt::format("c{}", state.inputRowTypes_.size() - 1));
    inputExpressions.push_back(std::make_shared<core::FieldAccessTypedExpr>(
        signature.args[i], state.inputRowNames_.back()));
  }

  const auto nullRatio = options.nullRatio;
  const auto seed = rand<uint32_t>(rng);
  const auto maxContainerSize = rand<uint32_t>(rng, 0, 10);
  TypePtr representedType;
  std::vector<variant> mapKeys;
  // 60% of times generate VARCHAR map keys, otherwise generate BIGINT map keys.
  if (coinToss(rng, 0.6)) {
    representedType = facebook::velox::fuzzer::randType(
        rng, jsonScalarTypes(), 3, {VARCHAR()});
    RandomInputGenerator<StringView> mapKeyGenerator{seed, VARCHAR(), 0.0, 10};
    for (auto i = 0; i < maxContainerSize + 2; ++i) {
      mapKeys.push_back(mapKeyGenerator.generate());
    }
  } else {
    representedType = facebook::velox::fuzzer::randType(
        rng, jsonScalarTypes(), 3, {BIGINT()});
    RandomInputGenerator<int64_t> mapKeyGenerator{seed, BIGINT(), 0.0};
    for (auto i = 0; i < maxContainerSize + 2; ++i) {
      mapKeys.push_back(mapKeyGenerator.generate());
    }
  }

  state.customInputGenerators_.emplace_back(
      std::make_shared<fuzzer::JsonInputGenerator>(
          seed,
          signature.args[0],
          nullRatio,
          fuzzer::getRandomInputGenerator(
              seed, representedType, nullRatio, mapKeys, maxContainerSize),
          true));
  state.customInputGenerators_.emplace_back(
      std::make_shared<fuzzer::JsonPathGenerator>(
          seed,
          signature.args[1],
          nullRatio,
          representedType,
          mapKeys,
          maxContainerSize,
          true));
  return inputExpressions;
}

} // namespace facebook::velox::fuzzer
