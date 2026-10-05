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
#include "velox/functions/sparksql/Size.h"

#include "velox/expression/VectorFunction.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatMapVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

class SizeFunction final : public exec::VectorFunction {
 public:
  explicit SizeFunction(bool legacySizeOfNull)
      : legacySizeOfNull_(legacySizeOfNull) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);
    const auto& input = args[0];

    context.ensureWritable(rows, outputType, result);
    auto* rawResult =
        result->asUnchecked<FlatVector<int32_t>>()->mutableRawValues();
    result->clearNulls(rows);

    if (input->encoding() == VectorEncoding::Simple::ARRAY ||
        input->encoding() == VectorEncoding::Simple::MAP) {
      const auto* rawSizes = input->asUnchecked<ArrayVectorBase>()->rawSizes();
      applySizes(rows, *input, rawSizes, nullptr, rawResult, *result);
      return;
    }

    exec::LocalDecodedVector decoded(context, *input, rows);
    if (decoded->isConstantMapping() && decoded->isNullAt(0)) {
      rows.applyToSelected([&](vector_size_t row) {
        if (legacySizeOfNull_) {
          rawResult[row] = -1;
        } else {
          result->setNull(row, true);
        }
      });
      return;
    }

    if (const auto* base = decoded->base()->as<ArrayVectorBase>()) {
      applySizes(
          rows, *decoded, base->rawSizes(), decoded.get(), rawResult, *result);
      return;
    }

    const auto* flatMap = decoded->base()->as<FlatMapVector>();
    VELOX_CHECK_NOT_NULL(
        flatMap, "Unsupported base vector encoding for Spark size");
    applyFlatMapSizes(rows, *decoded, *flatMap, rawResult, *result);
  }

 private:
  template <typename TInput>
  void applySizes(
      const SelectivityVector& rows,
      const TInput& input,
      const vector_size_t* rawSizes,
      const DecodedVector* decoded,
      int32_t* rawResult,
      BaseVector& result) const {
    if (!input.mayHaveNulls()) {
      if (decoded == nullptr) {
        rows.applyToSelected(
            [&](vector_size_t row) { rawResult[row] = rawSizes[row]; });
      } else {
        rows.applyToSelected([&](vector_size_t row) {
          rawResult[row] = rawSizes[decoded->index(row)];
        });
      }
      return;
    }

    rows.applyToSelected([&](vector_size_t row) {
      if (input.isNullAt(row)) {
        if (legacySizeOfNull_) {
          rawResult[row] = -1;
        } else {
          result.setNull(row, true);
        }
      } else {
        rawResult[row] =
            rawSizes[decoded == nullptr ? row : decoded->index(row)];
      }
    });
  }

  void applyFlatMapSizes(
      const SelectivityVector& rows,
      const DecodedVector& decoded,
      const FlatMapVector& flatMap,
      int32_t* rawResult,
      BaseVector& result) const {
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded.isNullAt(row)) {
        if (legacySizeOfNull_) {
          rawResult[row] = -1;
        } else {
          result.setNull(row, true);
        }
      } else {
        rawResult[row] = flatMap.sizeAt(decoded.index(row));
      }
    });
  }

  const bool legacySizeOfNull_;
};

std::vector<std::shared_ptr<exec::FunctionSignature>> sizeSignatures() {
  return {
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("integer")
          .argumentType("array(T)")
          .constantArgumentType("boolean")
          .build(),
      exec::FunctionSignatureBuilder()
          .typeVariable("K")
          .typeVariable("V")
          .returnType("integer")
          .argumentType("map(K,V)")
          .constantArgumentType("boolean")
          .build()};
}

std::shared_ptr<exec::VectorFunction> makeSize(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  try {
    VELOX_CHECK_EQ(inputArgs.size(), 2);
    const auto& legacySizeOfNull = inputArgs[1].constantValue;
    VELOX_USER_CHECK(
        legacySizeOfNull != nullptr && !legacySizeOfNull->isNullAt(0),
        "{} requires legacySizeOfNull to be a non-null constant boolean",
        name);
    return std::make_shared<SizeFunction>(
        legacySizeOfNull->asUnchecked<ConstantVector<bool>>()->valueAt(0));
  } catch (...) {
    return std::make_shared<exec::AlwaysFailingVectorFunction>(
        std::current_exception());
  }
}
} // namespace

void registerSize(const std::string& prefix) {
  exec::registerStatefulVectorFunction(
      prefix + "size",
      sizeSignatures(),
      makeSize,
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build());
}

} // namespace facebook::velox::functions::sparksql
