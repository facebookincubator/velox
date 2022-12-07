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

#include "velox/expression/VectorFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {

/// Wraps input in a constant encoding that repeats the first element, then in a
/// dictionary that reverses the order of rows.
class TestingDictionaryOverConstFunction : public exec::VectorFunction {
 public:
  TestingDictionaryOverConstFunction() {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    const auto size = rows.size();
    auto constant = BaseVector::wrapInConstant(size, 0, args[0]);

    auto indices = makeIndicesInReverse(size, context.pool());
    auto nulls = allocateNulls(size, context.pool());
    result =
        BaseVector::wrapInDictionary(nulls, indices, size, std::move(constant));
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T -> T
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("T")
                .argumentType("T")
                .build()};
  }
};

} // namespace facebook::velox::test
