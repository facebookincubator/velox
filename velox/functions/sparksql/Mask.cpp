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

#include <utility>

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/lib/Re2Functions.h"

namespace facebook::velox::functions::sparksql {
namespace {
class MaskFunction final : public exec::VectorFunction {
 public:
  MaskFunction() {}
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // Get the decoded vectors out of arguments.
    exec::DecodedArgs decodedArgs(rows, args, context);
    DecodedVector* strings = decodedArgs.at(0);
    DecodedVector* upperChars = decodedArgs.at(1);
    DecodedVector* lowerChars = decodedArgs.at(2);
    DecodedVector* digitChars = decodedArgs.at(3);
    DecodedVector* otherChars = decodedArgs.at(4);
    BaseVector::ensureWritable(rows, VARCHAR(), context.pool(), result);
    auto* results = result->as<FlatVector<StringView>>();
    // Optimization for the (flat, const, const, const, const) case.
    if (strings->isIdentityMapping() and upperChars->isConstantMapping() and
        lowerChars->isConstantMapping() and digitChars->isConstantMapping() and
        otherChars->isConstantMapping()) {
      // TODO: enable the inpalce if possible
      const auto* rawStrings = strings->data<StringView>();
      const auto upperChar = upperChars->valueAt<StringView>(0);
      const auto lowerChar = lowerChars->valueAt<StringView>(0);
      const auto digitChar = digitChars->valueAt<StringView>(0);
      const auto otherChar = otherChars->valueAt<StringView>(0);
      rows.applyToSelected([&](vector_size_t row) {
        auto proxy = exec::StringWriter<>(results, row);
        applyInner(
            rawStrings[row],
            upperChar,
            lowerChar,
            digitChar,
            otherChar,
            row,
            proxy);
        proxy.finalize();
      });
    } else {
      // The rest of the cases are handled through this general path and no
      // direct access.
      rows.applyToSelected([&](vector_size_t row) {
        auto proxy = exec::StringWriter<>(results, row);
        applyInner(
            strings->valueAt<StringView>(row),
            upperChars->valueAt<StringView>(row),
            lowerChars->valueAt<StringView>(row),
            digitChars->valueAt<StringView>(row),
            otherChars->valueAt<StringView>(row),
            row,
            proxy);
        proxy.finalize();
      });
    }
  }

  inline void applyInner(
      StringView input,
      const StringView upperChar,
      const StringView lowerChar,
      const StringView digitChar,
      const StringView otherChar,
      vector_size_t row,
      facebook::velox::exec::StringWriter<false>& result) const {
    const auto inputSize = input.size();
    auto inputBuffer = input.data();
    result.reserve(inputSize);
    auto outputBuffer = result.data();
    VELOX_CHECK_EQ(upperChar.size(), 1);
    VELOX_CHECK_EQ(lowerChar.size(), 1);
    VELOX_CHECK_EQ(digitChar.size(), 1);
    VELOX_CHECK_EQ(otherChar.size(), 1);

    auto upperCharBuffer = upperChar.data();
    auto lowerCharBuffer = lowerChar.data();
    auto digitCharBuffer = digitChar.data();
    auto otherCharBuffer = otherChar.data();

    for (auto i = 0; i < inputSize; i++) {
      unsigned char p = inputBuffer[i];
      if (isupper(p)) {
        outputBuffer[i] = upperCharBuffer[0];
      } else if (islower(p)) {
        outputBuffer[i] = lowerCharBuffer[0];
      } else if (isdigit(p)) {
        outputBuffer[i] = digitCharBuffer[0];
      } else {
        outputBuffer[i] = otherCharBuffer[0];
      }
    }
    result.resize(inputSize);
  }
};

std::shared_ptr<exec::VectorFunction> createMask(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  VELOX_CHECK_EQ(inputArgs.size(), 5);
  return std::make_shared<MaskFunction>();
}

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;

  signatures.emplace_back(exec::FunctionSignatureBuilder()
                              .returnType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .build());
  return signatures;
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(mask, signatures(), createMask);
} // namespace facebook::velox::functions::sparksql