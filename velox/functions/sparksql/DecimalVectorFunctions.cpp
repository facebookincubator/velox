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

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/ArithmeticImpl.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {
namespace {

template <class T, bool nullOnOverflow>
class MakeDecimalFunction final : public exec::VectorFunction {
 public:
  explicit MakeDecimalFunction(uint8_t precision) : precision_(precision) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK_EQ(args.size(), 3);
    context.ensureWritable(rows, outputType, resultRef);
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto unscaledVec = decodedArgs.at(0);
    auto result = resultRef->asUnchecked<FlatVector<T>>()->mutableRawValues();
    if constexpr (std::is_same_v<T, int64_t>) {
      int128_t bound = DecimalUtil::kPowersOfTen[precision_];
      rows.applyToSelected([&](int row) {
        auto unscaled = unscaledVec->valueAt<int64_t>(row);
        if (unscaled <= -bound || unscaled >= bound) {
          // Requested precision is too low to represent this value.
          if constexpr (nullOnOverflow) {
            resultRef->setNull(row, true);
          } else {
            VELOX_USER_FAIL(
                "Unscaled value {} too large for precision {}",
                unscaled,
                static_cast<int32_t>(precision_));
          }
        } else {
          result[row] = unscaled;
        }
      });
    } else {
      rows.applyToSelected([&](int row) {
        int128_t unscaled = unscaledVec->valueAt<int64_t>(row);
        result[row] = unscaled;
      });
    }
  }

 private:
  uint8_t precision_;
};
} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>>
makeDecimalByUnscaledValueSignatures() {
  return {exec::FunctionSignatureBuilder()
              .integerVariable("a_precision")
              .integerVariable("a_scale")
              .integerVariable("r_precision", "a_precision")
              .integerVariable("r_scale", "a_scale")
              .returnType("DECIMAL(r_precision, r_scale)")
              .argumentType("bigint")
              .constantArgumentType("DECIMAL(a_precision, a_scale)")
              .constantArgumentType("boolean")
              .build()};
}

std::shared_ptr<exec::VectorFunction> makeMakeDecimalByUnscaledValue(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  VELOX_CHECK_EQ(inputArgs.size(), 3);
  auto type = inputArgs[1].type;
  auto nullOnOverflow =
      inputArgs[2].constantValue->as<ConstantVector<bool>>()->valueAt(0);
  if (type->isShortDecimal()) {
    if (nullOnOverflow) {
      return std::make_shared<MakeDecimalFunction<int64_t, true>>(
          type->asShortDecimal().precision());
    } else {
      return std::make_shared<MakeDecimalFunction<int64_t, false>>(
          type->asShortDecimal().precision());
    }
  } else {
    if (nullOnOverflow) {
      return std::make_shared<MakeDecimalFunction<int128_t, true>>(
          type->asLongDecimal().precision());
    } else {
      return std::make_shared<MakeDecimalFunction<int128_t, false>>(
          type->asLongDecimal().precision());
    }
  }
}
} // namespace facebook::velox::functions::sparksql
