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

#include <gtest/gtest.h>

#include "velox/core/Expressions.h"
#include "velox/expression/fuzzer/ArgValuesGenerators.h"

namespace facebook::velox::fuzzer::test {
namespace {

TEST(ArgValuesGeneratorsTest, inverseFCdf) {
  const CallableSignature signature{
      .name = "inverse_f_cdf",
      .args = {DOUBLE(), DOUBLE(), DOUBLE()},
      .returnType = DOUBLE(),
      .constantArgs = {false, false, false},
  };
  VectorFuzzer::Options options;
  options.nullRatio = 0;

  FuzzerGenerator rng{0};
  ExpressionFuzzerState state{rng, 5};
  InverseFCdfArgValuesGenerator generator;

  for (auto i = 0; i < 100; ++i) {
    state.reset();
    const auto args = generator.generate(signature, options, rng, state);

    // Every argument must read an input column with a custom generator.
    // Arguments left null would be filled with arbitrary expressions, which
    // can produce degrees of freedom large enough to hang the function.
    ASSERT_EQ(args.size(), 3);
    ASSERT_EQ(state.customInputGenerators_.size(), 3);
    for (auto j = 0; j < 3; ++j) {
      const auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(args[j]);
      ASSERT_NE(field, nullptr);
      EXPECT_EQ(field->name(), state.inputRowNames_[j]);
      ASSERT_NE(state.customInputGenerators_[j], nullptr);
    }

    for (auto j = 0; j < 100; ++j) {
      const auto numerator =
          state.customInputGenerators_[0]->generate().value<double>();
      const auto denominator =
          state.customInputGenerators_[1]->generate().value<double>();
      const auto probability =
          state.customInputGenerators_[2]->generate().value<double>();
      EXPECT_GE(numerator, 0);
      EXPECT_LE(numerator, 1'000'000);
      EXPECT_GE(denominator, 0);
      EXPECT_LE(denominator, 1'000'000);
      EXPECT_GE(probability, 0);
      EXPECT_LE(probability, 1);
    }
  }
}

} // namespace
} // namespace facebook::velox::fuzzer::test
