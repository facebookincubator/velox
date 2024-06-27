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
  static constexpr std::string_view maskedUpperCase{"X"};
  static constexpr std::string_view maskedLowerCase{"x"};
  static constexpr std::string_view maskedDigit{"n"};

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
    DecodedVector* upperChars = args.size() >= 2 ? decodedArgs.at(1) : nullptr;
    DecodedVector* lowerChars = args.size() >= 3 ? decodedArgs.at(2) : nullptr;
    DecodedVector* digitChars = args.size() >= 4 ? decodedArgs.at(3) : nullptr;
    DecodedVector* otherChars = args.size() >= 5 ? decodedArgs.at(4) : nullptr;
    BaseVector::ensureWritable(rows, VARCHAR(), context.pool(), result);
    auto* results = result->as<FlatVector<StringView>>();
    auto getMaskedChar =
        [&](DecodedVector* inputChars,
            const VectorPtr& arg,
            int index,
            std::string_view maskedChar) -> std::optional<StringView> {
      if (inputChars == nullptr) {
        return StringView{maskedChar};
      }
      if (arg->isNullAt(index)) {
        return std::nullopt;
      }
      return inputChars->valueAt<StringView>(index);
    };

    // Optimization for the (flat, const, const, const, const) case.
    if (strings->isIdentityMapping() and
        (upperChars == nullptr || upperChars->isConstantMapping()) and
        (lowerChars == nullptr || lowerChars->isConstantMapping()) and
        (digitChars == nullptr || digitChars->isConstantMapping()) and
        (otherChars == nullptr || otherChars->isConstantMapping())) {
      // TODO: enable the inplace if possible
      const auto* rawStrings = strings->data<StringView>();
      const auto upperChar =
          getMaskedChar(upperChars, args[1], 0, maskedUpperCase);
      const auto lowerChar =
          getMaskedChar(lowerChars, args[2], 0, maskedLowerCase);
      const auto digitChar = getMaskedChar(digitChars, args[3], 0, maskedDigit);
      const auto otherChar = (otherChars == nullptr || args[4]->isNullAt(0))
          ? std::nullopt
          : std::optional(otherChars->valueAt<StringView>(0));

      rows.applyToSelected([&](vector_size_t row) {
        if (args[0]->isNullAt(row)) {
          results->setNull(row, true);
          return;
        }
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
        if (args[0]->isNullAt(row)) {
          results->setNull(row, true);
          return;
        }
        auto proxy = exec::StringWriter<>(results, row);
        const auto upperChar =
            getMaskedChar(upperChars, args[1], row, maskedUpperCase);
        const auto lowerChar =
            getMaskedChar(lowerChars, args[2], row, maskedLowerCase);
        const auto digitChar =
            getMaskedChar(digitChars, args[3], row, maskedDigit);
        const auto otherChar =
            (otherChars == nullptr || args[4]->containsNullAt(row))
            ? std::nullopt
            : std::optional(otherChars->valueAt<StringView>(row));
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
      const std::optional<StringView> upperChar,
      const std::optional<StringView> lowerChar,
      const std::optional<StringView> digitChar,
      const std::optional<StringView> otherChar,
      vector_size_t row,
      facebook::velox::exec::StringWriter<false>& result) const {
    const auto inputSize = input.size();
    auto inputBuffer = input.data();
    result.reserve(inputSize);
    auto outputBuffer = result.data();

    auto hasMaskedUpperChar = false;
    auto hasMaskedLowerChar = false;
    auto hasMaskedDigitChar = false;
    auto hasMaskedOtherChar = false;
    auto maskedUpperChar = "";
    auto maskedLowerChar = "";
    auto maskedDigitChar = "";
    auto maskedOtherChar = "";

    auto processChar = [&](const std::optional<StringView>& charOpt,
                           const char*& maskedChar,
                           bool& hasMaskedChar,
                           const char* charType) {
      if (charOpt.has_value()) {
        VELOX_USER_CHECK(
            charOpt.value().size() == 1,
            std::string("Length of ") + charType + " should be 1");
        maskedChar = charOpt.value().data();
        hasMaskedChar = true;
      }
    };

    processChar(upperChar, maskedUpperChar, hasMaskedUpperChar, "upperChar");
    processChar(lowerChar, maskedLowerChar, hasMaskedLowerChar, "lowerChar");
    processChar(digitChar, maskedDigitChar, hasMaskedDigitChar, "digitChar");
    processChar(otherChar, maskedOtherChar, hasMaskedOtherChar, "otherChar");

    for (auto i = 0; i < inputSize; i++) {
      unsigned char p = inputBuffer[i];
      if (isupper(p)) {
        outputBuffer[i] = hasMaskedUpperChar ? maskedUpperChar[0] : p;
      } else if (islower(p)) {
        outputBuffer[i] = hasMaskedLowerChar ? maskedLowerChar[0] : p;
      } else if (isdigit(p)) {
        outputBuffer[i] = hasMaskedDigitChar ? maskedDigitChar[0] : p;
      } else {
        outputBuffer[i] = hasMaskedOtherChar ? maskedOtherChar[0] : p;
      }
    }
    result.resize(inputSize);
  }
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
