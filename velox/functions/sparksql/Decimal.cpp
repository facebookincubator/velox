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
#include "velox/functions/sparksql/Decimal.h"

#include "velox/expression/DecodedArgs.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

class CheckOverflowFunction final : public exec::VectorFunction {
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK_EQ(args.size(), 3);
    // This VectorPtr type is different with type in makeCheckOverflow, because
    // we cannot get input type by signature the input vector origins from
    // DecimalArithmetic, it is a computed type by arithmetic operation
    auto fromType = args[0]->type();
    auto toType = args[2]->type();
    context.ensureWritable(rows, toType, resultRef);
    if (toType->kind() == TypeKind::SHORT_DECIMAL) {
      if (fromType->kind() == TypeKind::SHORT_DECIMAL) {
        applyForVectorType<UnscaledShortDecimal, UnscaledShortDecimal>(
            rows, args, outputType, context, resultRef);
      } else {
        applyForVectorType<UnscaledLongDecimal, UnscaledShortDecimal>(
            rows, args, outputType, context, resultRef);
      }
    } else {
      if (fromType->kind() == TypeKind::SHORT_DECIMAL) {
        applyForVectorType<UnscaledShortDecimal, UnscaledLongDecimal>(
            rows, args, outputType, context, resultRef);
      } else {
        applyForVectorType<UnscaledLongDecimal, UnscaledLongDecimal>(
            rows, args, outputType, context, resultRef);
      }
    }
  }

 private:
  template <typename TInput, typename TOutput>
  void applyForVectorType(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const {
    auto fromType = args[0]->type();
    auto toType = args[2]->type();
    auto result =
        resultRef->asUnchecked<FlatVector<TOutput>>()->mutableRawValues();
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decimalValue = decodedArgs.at(0);
    VELOX_CHECK(decodedArgs.at(1)->isConstantMapping());
    auto nullOnOverflow = decodedArgs.at(1)->valueAt<bool>(0);

    const auto& fromPrecisionScale = getDecimalPrecisionScale(*fromType);
    const auto& toPrecisionScale = getDecimalPrecisionScale(*toType);
    rows.applyToSelected([&](int row) {
      auto rescaledValue = DecimalUtil::rescaleWithRoundUp<TInput, TOutput>(
          decimalValue->valueAt<TInput>(row),
          fromPrecisionScale.first,
          fromPrecisionScale.second,
          toPrecisionScale.first,
          toPrecisionScale.second,
          nullOnOverflow);
      if (rescaledValue.has_value()) {
        result[row] = rescaledValue.value();
      } else {
        resultRef->setNull(row, true);
      }
    });
  }
};

class MakeDecimalFunction final : public exec::VectorFunction {
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK_EQ(args.size(), 3);
    auto fromType = args[0]->type();
    auto toType = args[1]->type();
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto unscaledVec = decodedArgs.at(0);
    VELOX_CHECK(decodedArgs.at(1)->isConstantMapping());
    VELOX_CHECK(decodedArgs.at(2)->isConstantMapping());
    auto nullOnOverflow = decodedArgs.at(2)->valueAt<bool>(0);
    const auto& toPrecisionScale = getDecimalPrecisionScale(*toType);
    auto precision = toPrecisionScale.first;
    auto scale = toPrecisionScale.second;
    context.ensureWritable(
        rows,
        SHORT_DECIMAL(
            static_cast<uint8_t>(precision), static_cast<uint8_t>(scale)),
        resultRef);
    auto result = resultRef->asUnchecked<FlatVector<UnscaledShortDecimal>>()
                      ->mutableRawValues();
    rows.applyToSelected([&](int row) {
      auto unscaled = unscaledVec->valueAt<int64_t>(row);

      if (unscaled <= -static_cast<long>(DecimalUtil::kPowersOfTen[18]) ||
          unscaled >= static_cast<long>(DecimalUtil::kPowersOfTen[18])) {
        if (precision < 19) {
          resultRef->setNull(row, true);
        }
      } else if (
          unscaled <= -static_cast<long>(
                          DecimalUtil::kPowersOfTen[std::min(precision, 18)]) ||
          unscaled >= static_cast<long>(
                          DecimalUtil::kPowersOfTen[std::min(precision, 18)])) {
        resultRef->setNull(row, true);
      } else {
        result[row] = UnscaledShortDecimal(unscaled);
      }
    });
  }
};

template <typename TInput>
class RoundDecimalFunction final : public exec::VectorFunction {
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK_EQ(args.size(), 2);
    auto fromType = args[0]->type();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decimalValue = decodedArgs.at(0);
    VELOX_CHECK(decodedArgs.at(1)->isConstantMapping());
    auto scale = decodedArgs.at(1)->valueAt<int32_t>(0);

    const auto& fromPrecisionScale = getDecimalPrecisionScale(*fromType);
    const auto& fromPrecision = fromPrecisionScale.first;
    const auto& fromScale = fromPrecisionScale.second;
    auto toPrecision = fromPrecision;
    auto toScale = fromScale;

    // Calculate the result data type based on spark logic.
    const auto& integralLeastNumDigits = fromPrecision - fromScale + 1;
    if (scale < 0) {
      const auto& newPrecision =
          std::max(integralLeastNumDigits, -fromScale + 1);
      toPrecision = std::min(newPrecision, 38);
      toScale = 0;
    } else {
      toScale = std::min(fromScale, scale);
      toPrecision = std::min(integralLeastNumDigits + toScale, 38);
    }

    rows.applyToSelected([&](int row) {
      if (toPrecision > 18) {
        context.ensureWritable(
            rows,
            LONG_DECIMAL(
                static_cast<uint8_t>(toPrecision),
                static_cast<uint8_t>(toScale)),
            resultRef);
        auto rescaledValue =
            DecimalUtil::rescaleWithRoundUp<TInput, UnscaledLongDecimal>(
                decimalValue->valueAt<TInput>(row),
                fromPrecision,
                fromScale,
                toPrecision,
                toScale);
        auto result = resultRef->asUnchecked<FlatVector<UnscaledLongDecimal>>()
                          ->mutableRawValues();
        if (rescaledValue.has_value()) {
          result[row] = rescaledValue.value();
        } else {
          resultRef->setNull(row, true);
        }
      } else {
        context.ensureWritable(
            rows,
            SHORT_DECIMAL(
                static_cast<uint8_t>(toPrecision),
                static_cast<uint8_t>(toScale)),
            resultRef);
        auto rescaledValue =
            DecimalUtil::rescaleWithRoundUp<TInput, UnscaledShortDecimal>(
                decimalValue->valueAt<TInput>(row),
                fromPrecision,
                fromScale,
                toPrecision,
                toScale);
        auto result = resultRef->asUnchecked<FlatVector<UnscaledShortDecimal>>()
                          ->mutableRawValues();
        if (rescaledValue.has_value()) {
          result[row] = rescaledValue.value();
        } else {
          resultRef->setNull(row, true);
        }
      }
    });
  }
};

template <typename TInput>
class AbsFunction final : public exec::VectorFunction {
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK_EQ(args.size(), 1);
    auto inputType = args[0]->type();
    VELOX_CHECK(
        inputType->isShortDecimal() || inputType->isLongDecimal(),
        "ShortDecimal or LongDecimal type is required.");

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decimalVector = decodedArgs.at(0);
    if (inputType->isShortDecimal()) {
      auto decimalType = inputType->asShortDecimal();
      context.ensureWritable(
          rows,
          SHORT_DECIMAL(decimalType.precision(), decimalType.scale()),
          resultRef);
      auto result = resultRef->asUnchecked<FlatVector<UnscaledShortDecimal>>()
                        ->mutableRawValues();
      rows.applyToSelected([&](int row) {
        auto unscaled = std::abs(decimalVector->valueAt<int64_t>(row));
        if (UnscaledShortDecimal::valueInRange(unscaled)) {
          result[row] = UnscaledShortDecimal(unscaled);
        } else {
          // TODO: adjust the bahavior according to ANSI.
          resultRef->setNull(row, true);
        }
      });
    } else {
      auto decimalType = inputType->asLongDecimal();
      context.ensureWritable(
          rows,
          LONG_DECIMAL(decimalType.precision(), decimalType.scale()),
          resultRef);
      auto result = resultRef->asUnchecked<FlatVector<UnscaledLongDecimal>>()
                        ->mutableRawValues();
      rows.applyToSelected([&](int row) {
        auto unscaled = std::abs(decimalVector->valueAt<int128_t>(row));
        if (UnscaledLongDecimal::valueInRange(unscaled)) {
          result[row] = UnscaledLongDecimal(unscaled);
        } else {
          // TODO: adjust the bahavior according to ANSI.
          resultRef->setNull(row, true);
        }
      });
    }
  }
};

} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>>
checkOverflowSignatures() {
  return {exec::FunctionSignatureBuilder()
              .integerVariable("a_precision")
              .integerVariable("a_scale")
              .integerVariable("b_precision")
              .integerVariable("b_scale")
              .integerVariable("r_precision", "min(38, b_precision)")
              .integerVariable("r_scale", "min(38, b_scale)")
              .returnType("DECIMAL(r_precision, r_scale)")
              .argumentType("DECIMAL(a_precision, a_scale)")
              .argumentType("boolean")
              .argumentType("DECIMAL(b_precision, b_scale)")
              .build()};
}

std::vector<std::shared_ptr<exec::FunctionSignature>> makeDecimalSignatures() {
  return {exec::FunctionSignatureBuilder()
              .integerVariable("a_precision")
              .integerVariable("a_scale")
              .integerVariable("r_precision", "min(38, a_precision)")
              .integerVariable("r_scale", "min(38, a_scale)")
              .returnType("DECIMAL(r_precision, r_scale)")
              .argumentType("bigint")
              .argumentType("DECIMAL(a_precision, a_scale)")
              .argumentType("boolean")
              .build()};
}

std::vector<std::shared_ptr<exec::FunctionSignature>> roundDecimalSignatures() {
  return {exec::FunctionSignatureBuilder()
              .integerVariable("a_precision")
              .integerVariable("a_scale")
              .integerVariable("r_precision", "min(38, a_precision)")
              .integerVariable("r_scale", "min(38, a_scale)")
              .returnType("DECIMAL(r_precision, r_scale)")
              .argumentType("DECIMAL(a_precision, a_scale)")
              .argumentType("integer")
              .build()};
}

std::vector<std::shared_ptr<exec::FunctionSignature>> absSignatures() {
  return {exec::FunctionSignatureBuilder()
              .integerVariable("a_precision")
              .integerVariable("a_scale")
              .integerVariable("r_precision", "min(38, a_precision)")
              .integerVariable("r_scale", "min(38, a_scale)")
              .returnType("DECIMAL(r_precision, r_scale)")
              .argumentType("DECIMAL(a_precision, a_scale)")
              .build()};
}

std::shared_ptr<exec::VectorFunction> makeCheckOverflow(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 3);
  static const auto kCheckOverflowFunction =
      std::make_shared<CheckOverflowFunction>();
  return kCheckOverflowFunction;
}

std::shared_ptr<exec::VectorFunction> makeMakeDecimal(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 3);
  static const auto kMakeDecimalFunction =
      std::make_shared<MakeDecimalFunction>();
  return kMakeDecimalFunction;
}

std::shared_ptr<exec::VectorFunction> makeRoundDecimal(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  auto fromType = inputArgs[0].type;
  switch (fromType->kind()) {
    case TypeKind::SHORT_DECIMAL:
      return std::make_shared<RoundDecimalFunction<UnscaledShortDecimal>>();
    case TypeKind::LONG_DECIMAL:
      return std::make_shared<RoundDecimalFunction<UnscaledLongDecimal>>();
    default:
      VELOX_FAIL(
          "Not support this type {} in round_decimal", fromType->kindName())
  }
}

std::shared_ptr<exec::VectorFunction> makeAbs(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 1);
  auto type = inputArgs[0].type;
  switch (type->kind()) {
    case TypeKind::SHORT_DECIMAL:
      return std::make_shared<AbsFunction<UnscaledShortDecimal>>();
    case TypeKind::LONG_DECIMAL:
      return std::make_shared<AbsFunction<UnscaledLongDecimal>>();
    default:
      VELOX_FAIL("Not support this type {} in abs", type->kindName())
  }
}

} // namespace facebook::velox::functions::sparksql
