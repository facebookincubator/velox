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

// mask(string) -> string
// mask(string, upperChar) -> string
// mask(string, upperChar, lowerChar) -> string
// mask(string, upperChar, lowerChar, digitChar) -> string
// mask(string, upperChar, lowerChar, digitChar, otherChar) -> string
//
// Masks the characters of the given string value with the provided specific
// characters respectively. Upper-case characters are replaced with the second
// argument. Default value is 'X'. Lower-case characters are replaced with the
// third argument. Default value is 'x'. Digit characters are replaced with the
// fourth argument. Default value is 'n'. Other characters are replaced with the
// last argument. Default value is NULL and the original character is retained.
// If the provided nth argument is NULL, the related original character is
// retained.
class MaskFunction final : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // Get the decoded vectors out of arguments.
    exec::DecodedArgs decodedArgs(rows, args, context);
    DecodedVector* strings = decodedArgs.at(0);
    DecodedVector* upperChars = args.size() >= 2 ? decodedArgs.at(1) : nullptr;
    DecodedVector* lowerChars = args.size() >= 3 ? decodedArgs.at(2) : nullptr;
    DecodedVector* digitChars = args.size() >= 4 ? decodedArgs.at(3) : nullptr;
    DecodedVector* otherChars = args.size() >= 5 ? decodedArgs.at(4) : nullptr;

    BaseVector::ensureWritable(rows, VARCHAR(), context.pool(), result);
    auto* flatResult = result->as<FlatVector<StringView>>();
    auto getMaskedChar = [&](DecodedVector* inputChars,
                             const VectorPtr& arg,
                             vector_size_t row,
                             std::optional<char> maskedChar,
                             const char* charType) -> std::optional<char> {
      if (inputChars == nullptr) {
        return maskedChar;
      }
      if (arg->isNullAt(row)) {
        return std::nullopt;
      }
      std::optional<StringView> inputCharStr =
          inputChars->valueAt<StringView>(row);
      if (inputCharStr.has_value()) {
        VELOX_USER_CHECK(
            inputCharStr.value().size() == 1,
            std::string("Length of ") + charType + " should be 1");
      }
      return inputCharStr->data()[0];
    };

    // Fast path for the (flat, const, const, const, const) case.
    if (strings->isIdentityMapping() and
        (upperChars == nullptr || upperChars->isConstantMapping()) and
        (lowerChars == nullptr || lowerChars->isConstantMapping()) and
        (digitChars == nullptr || digitChars->isConstantMapping()) and
        (otherChars == nullptr || otherChars->isConstantMapping())) {
      const auto* rawStrings = strings->data<StringView>();
      const auto upperChar =
          getMaskedChar(upperChars, args[1], 0, kMaskedUpperCase_, "upperChar");
      const auto lowerChar =
          getMaskedChar(lowerChars, args[2], 0, kMaskedLowerCase_, "lowerChar");
      const auto digitChar =
          getMaskedChar(digitChars, args[3], 0, kMaskedDigit_, "digitChar");
      const auto otherChar =
          getMaskedChar(otherChars, args[4], 0, std::nullopt, "otherChar");
      rows.applyToSelected([&](vector_size_t row) {
        if (args[0]->isNullAt(row)) {
          flatResult->setNull(row, true);
          return;
        }
        auto proxy = exec::StringWriter<>(flatResult, row);
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
      rows.applyToSelected([&](vector_size_t row) {
        const auto upperChar = getMaskedChar(
            upperChars, args[1], row, kMaskedUpperCase_, "upperChar");
        const auto lowerChar = getMaskedChar(
            lowerChars, args[2], row, kMaskedLowerCase_, "lowerChar");
        const auto digitChar =
            getMaskedChar(digitChars, args[3], row, kMaskedDigit_, "digitChar");
        const auto otherChar =
            getMaskedChar(otherChars, args[4], row, std::nullopt, "otherChar");
        if (args[0]->isNullAt(row)) {
          flatResult->setNull(row, true);
          return;
        }
        auto proxy = exec::StringWriter<>(flatResult, row);
        applyInner(
            strings->valueAt<StringView>(row),
            upperChar,
            lowerChar,
            digitChar,
            otherChar,
            row,
            proxy);
        proxy.finalize();
      });
    }
  }

  void applyInner(
      StringView input,
      const std::optional<char> upperChar,
      const std::optional<char> lowerChar,
      const std::optional<char> digitChar,
      const std::optional<char> otherChar,
      vector_size_t row,
      exec::StringWriter<false>& result) const {
    const auto inputSize = input.size();
    auto inputBuffer = input.data();
    result.reserve(inputSize);
    auto outputBuffer = result.data();

    for (auto i = 0; i < inputSize; i++) {
      unsigned char input = inputBuffer[i];
      unsigned char masked = input;
      if (isupper(input) && upperChar.has_value()) {
        masked = upperChar.value();
      } else if (islower(input) && lowerChar.has_value()) {
        masked = lowerChar.value();
      } else if (isdigit(input) && digitChar.has_value()) {
        masked = digitChar.value();
      } else if (
          !isupper(input) && !islower(input) && !isdigit(input) &&
          otherChar.has_value()) {
        masked = otherChar.value();
      }
      outputBuffer[i] = masked;
    }
    result.resize(inputSize);
  }

 private:
  static constexpr char kMaskedUpperCase_ = 'X';
  static constexpr char kMaskedLowerCase_ = 'x';
  static constexpr char kMaskedDigit_ = 'n';
};

std::shared_ptr<exec::VectorFunction> createMask(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK_GE(inputArgs.size(), 1);
  return std::make_shared<MaskFunction>();
}

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  signatures.emplace_back(exec::FunctionSignatureBuilder()
                              .returnType("varchar")
                              .argumentType("varchar")
                              .build());
  signatures.emplace_back(exec::FunctionSignatureBuilder()
                              .returnType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .build());
  signatures.emplace_back(exec::FunctionSignatureBuilder()
                              .returnType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .build());
  signatures.emplace_back(exec::FunctionSignatureBuilder()
                              .returnType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .argumentType("varchar")
                              .build());
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

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION_WITH_METADATA(
    mask,
    signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    createMask);
} // namespace facebook::velox::functions::sparksql
