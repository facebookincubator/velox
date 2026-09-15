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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "velox/common/fuzzer/ConstrainedGenerators.h"
#include "velox/core/Expressions.h"
#include "velox/expression/fuzzer/FuzzerToolkit.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::functions::sparksql::fuzzer {

class PadArgValuesGenerator : public velox::fuzzer::ArgValuesGenerator {
 public:
  std::vector<core::TypedExprPtr> generate(
      const velox::fuzzer::CallableSignature& signature,
      const VectorFuzzer::Options& options,
      velox::fuzzer::FuzzerGenerator& rng,
      velox::fuzzer::ExpressionFuzzerState& state) override {
    VELOX_CHECK(signature.args.size() == 2 || signature.args.size() == 3);
    VELOX_CHECK_EQ(signature.args[0]->kind(), TypeKind::VARCHAR);
    VELOX_CHECK_EQ(signature.args[1]->kind(), TypeKind::INTEGER);
    if (signature.args.size() == 3) {
      VELOX_CHECK_EQ(signature.args[2]->kind(), TypeKind::VARCHAR);
    }

    populateInputTypesAndNames(signature, state);

    std::vector<core::TypedExprPtr> inputExpressions;
    inputExpressions.reserve(signature.args.size());
    const auto firstInputIndex =
        state.inputRowNames_.size() - signature.args.size();
    const auto nullRatio = options.nullRatio;
    const auto stringLength = std::max<size_t>(1, options.stringLength);
    constexpr int32_t kMaxPadSize = 1024 * 1024;

    for (size_t i = 0; i < signature.args.size(); ++i) {
      const auto seed = velox::fuzzer::rand<uint32_t>(rng);
      if (i == 1) {
        state.customInputGenerators_.emplace_back(
            std::make_shared<velox::fuzzer::RangeConstrainedGenerator<int32_t>>(
                seed, signature.args[i], nullRatio, 0, kMaxPadSize));
      } else if (i == 2) {
        state.customInputGenerators_.emplace_back(
            std::make_shared<velox::fuzzer::NotEqualConstrainedGenerator>(
                seed,
                signature.args[i],
                variant(std::string("")),
                std::make_unique<
                    velox::fuzzer::RandomInputGenerator<StringView>>(
                    seed, signature.args[i], nullRatio, stringLength)));
      } else {
        state.customInputGenerators_.emplace_back(nullptr);
      }

      inputExpressions.emplace_back(
          std::make_shared<core::FieldAccessTypedExpr>(
              signature.args[i], state.inputRowNames_[firstInputIndex + i]));
    }

    return inputExpressions;
  }
};

} // namespace facebook::velox::functions::sparksql::fuzzer
