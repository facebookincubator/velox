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

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/type/DecimalUtils.cpp"

namespace facebook::velox::functions {
namespace {

class DecimalAddFunction : public exec::VectorFunction {
 public:
  DecimalAddFunction(uint8_t aRescale, uint8_t bRescale)
      : aRescale_(aRescale), bRescale_(bRescale) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Each arg is Array of integers
      const TypePtr& outputType,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    auto num = std::min(args[0]->size(), args[1]->size());
    FlatVector<int64_t>* aFlatVector = args[0]->asFlatVector<int64_t>();
    FlatVector<int64_t>* bFlatVector = args[1]->asFlatVector<int64_t>();
    // Initialize flat results vector.
    BaseVector::ensureWritable(rows, outputType, context->pool(), result);
    BufferPtr resultValues =
        (*result)->as<FlatVector<int64_t>>()->mutableValues(rows.size());
    auto rawValues = resultValues->asMutable<int64_t>();

    auto processRow = [&](vector_size_t row) {
      std::vector<int64_t> rescaledValues;
      rescale(aFlatVector, bFlatVector, row, rescaledValues);
      rawValues[row] = rescaledValues[0] + rescaledValues[1];
    };
    rows.applyToSelected([&](vector_size_t row) { processRow(row); });
  }

 private:
  void rescale(
      FlatVector<int64_t>* aFlatVector,
      FlatVector<int64_t>* bFlatVector,
      const vector_size_t i,
      std::vector<int64_t>& rescaled) const {
    rescaled.push_back(
        aFlatVector->isNullAt(i)
            ? 0
            : aFlatVector->valueAt(i) * pow(10, aRescale_));
    rescaled.push_back(
        bFlatVector->isNullAt(i)
            ? 0
            : bFlatVector->valueAt(i) * pow(10, bRescale_));
  }
  const uint8_t aRescale_;
  const uint8_t bRescale_;
};

// Function callback to compute the return type precision and scale.
void computeReturnPrecisionScale(
    const std::unordered_map<std::string, std::int32_t>& variablesBinding,
    std::vector<int32_t>& output) {
  // min(38, max(a_precision - a_scale, b_precision - b_scale) + max(a_scale,
  // b_scale) + 1)
  const uint8_t a_precision = variablesBinding.find("a_precision")->second;
  const uint8_t b_precision = variablesBinding.find("b_precision")->second;
  const uint8_t a_scale = variablesBinding.find("a_scale")->second;
  const uint8_t b_scale = variablesBinding.find("b_scale")->second;
  output.push_back(std::min(
      38,
      std::max(a_precision - a_scale, b_precision - b_scale) +
          std::max(a_scale, b_scale) + 1));
  output.push_back(std::max(a_scale, b_scale));
}
std::vector<std::shared_ptr<exec::FunctionSignature>> decimalAddSignature() {
  return {
      exec::FunctionSignatureBuilder()
          .returnType(
              "SHORT_DECIMAL(r_precision,r_scale)", {"r_precision", "r_scale"})
          .argumentType(
              "SHORT_DECIMAL(a_precision,a_scale)", {"a_precision", "a_scale"})
          .argumentType(
              "SHORT_DECIMAL(b_precision,b_scale)", {"b_precision", "b_scale"})
          .computeLongVariablesBinding(&computeReturnPrecisionScale)
          .build()};
}

std::shared_ptr<exec::VectorFunction> createDecimalAddFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  ShortDecimalTypePtr aType =
      std::dynamic_pointer_cast<const ShortDecimalType>(inputArgs[0].type);
  ShortDecimalTypePtr bType =
      std::dynamic_pointer_cast<const ShortDecimalType>(inputArgs[1].type);
  uint8_t aRescale = computeRescaleFactor(aType->scale(), bType->scale());
  uint8_t bRescale = computeRescaleFactor(bType->scale(), aType->scale());
  return std::make_shared<DecimalAddFunction>(aRescale, bRescale);
}
}; // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_add_short_short,
    decimalAddSignature(),
    createDecimalAddFunction);

}; // namespace facebook::velox::functions