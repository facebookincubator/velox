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

#include "velox/experimental/cudf/expression/CommonFunctions.h"
#include "velox/experimental/cudf/expression/DateTruncFunction.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/prestosql/DateAddFunction.h"
#include "velox/experimental/cudf/expression/prestosql/DateDiffFunction.h"
#include "velox/experimental/cudf/expression/prestosql/DatePlusIntervalFunction.h"
#include "velox/experimental/cudf/expression/prestosql/ToUnixtimeFunction.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/slice.hpp>

#include <memory>
#include <optional>

namespace facebook::velox::cudf_velox {
namespace {

// Reads the scalar value of a constant expression, materialising the constant
// vector via pool when the ConstantTypedExpr does not already hold one.
template <typename T>
T constantScalarValue(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  const auto* constExpr = expr->asUnchecked<core::ConstantTypedExpr>();
  const auto vector = constExpr->hasValueVector()
      ? constExpr->valueVector()
      : constExpr->toConstantVector(pool);
  return vector->template as<SimpleVector<T>>()->valueAt(0);
}

void registerPrestoArrayAccessFunctions(const std::string& prefix) {
  // Presto element_at is 1-based, allows negative indices from the end, and
  // returns NULL for out-of-bounds indices.
  registerArrayAccessFunction(
      prefix + "element_at",
      ArrayAccessPolicy{
          .allowNegativeIndices = true,
          .nullOnNegativeIndices = false,
          .allowOutOfBound = true,
          .indexStartsAtOne = true,
      },
      arrayAccessSignatures({"integer", "bigint"}));

  // Presto subscript is 1-based and raises on negative or out-of-bounds
  // indices.
  registerArrayAccessFunction(
      prefix + "subscript",
      ArrayAccessPolicy{
          .allowNegativeIndices = false,
          .nullOnNegativeIndices = false,
          .allowOutOfBound = false,
          .indexStartsAtOne = true,
      },
      arrayAccessSignatures({"integer", "bigint"}));
}

class SubstrFunction : public CudfFunction {
 public:
  SubstrFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "substr expects at least 2 inputs");
    VELOX_CHECK_LE(expr->inputs().size(), 3, "substr expects at most 3 inputs");

    VELOX_CHECK(
        expr->inputs()[1]->isConstantKind(), "substr start must be a constant");
    auto startValue = constantScalarValue<int64_t>(expr->inputs()[1], pool);
    start_ = static_cast<cudf::size_type>(startValue);
    if (startValue >= 1) {
      // cuDF indexing starts at 0.
      // Presto indexing starts at 1.
      // Positive indices need to subtract 1.
      start_ = static_cast<cudf::size_type>(startValue - 1);
    }

    if (expr->inputs().size() > 2) {
      VELOX_CHECK(
          expr->inputs()[2]->isConstantKind(),
          "substr length must be a constant");
      auto lengthValue = constantScalarValue<int64_t>(expr->inputs()[2], pool);
      // cuDF uses indices [begin, end).
      // Presto uses length as the length of the substring.
      // We compute the end as start + length.
      end_ = start_ + static_cast<cudf::size_type>(lengthValue);
      hasEnd_ = true;
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      [[maybe_unused]] cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    const auto start = std::optional<cudf::size_type>{start_};
    const auto end =
        hasEnd_ ? std::optional<cudf::size_type>{end_} : std::nullopt;
    const auto step = std::optional<cudf::size_type>{1};
    return cudf::strings::slice_strings(inputCol, start, end, step, stream, mr);
  }

 private:
  cudf::size_type start_{0};
  cudf::size_type end_{0};
  bool hasEnd_{false};
};

} // namespace

void registerPrestoFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerPrestoArrayAccessFunctions(prefix);

  registerCudfFunctions(
      {prefix + "substr", prefix + "substring"},
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<SubstrFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("bigint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("bigint")
           .constantArgumentType("bigint")
           .build()});

  registerCudfFunction(
      prefix + "plus",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<prestosql::DatePlusIntervalFunction>(
            expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .argumentType("interval day to second")
           .build()});

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<prestosql::DateAddFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .constantArgumentType("varchar")
           .argumentType("bigint")
           .argumentType("date")
           .build()},
      true,
      prestosql::DateAddFunction::canEvaluate);

  registerCudfFunction(
      prefix + "date_trunc",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<DateTruncFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp")
           .constantArgumentType("varchar")
           .argumentType("timestamp")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .constantArgumentType("varchar")
           .argumentType("date")
           .build()},
      true,
      DateTruncFunction::canEvaluate);

  registerCudfFunction(
      prefix + "date_diff",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<prestosql::DateDiffFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("varchar")
           .argumentType("date")
           .argumentType("date")
           .build(),
       FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("varchar")
           .argumentType("timestamp")
           .argumentType("timestamp")
           .build()},
      true,
      prestosql::DateDiffFunction::canEvaluate);

  registerCudfFunction(
      prefix + "to_unixtime",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<prestosql::ToUnixtimeFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("timestamp")
           .build()},
      true,
      prestosql::ToUnixtimeFunction::canEvaluate);
}

} // namespace facebook::velox::cudf_velox
