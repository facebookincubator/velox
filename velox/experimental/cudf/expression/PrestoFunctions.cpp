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
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/prestosql/DatePlusIntervalFunction.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/BaseVector.h"

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/slice.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {
namespace {

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
  explicit SubstrFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "substr expects at least 2 inputs");
    VELOX_CHECK_LE(expr->inputs().size(), 3, "substr expects at most 3 inputs");

    auto startExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(startExpr, "substr start must be a constant");

    auto startValue =
        startExpr->value()->as<SimpleVector<int64_t>>()->valueAt(0);
    start_ = static_cast<cudf::size_type>(startValue);
    if (startValue >= 1) {
      // cuDF indexing starts at 0.
      // Presto indexing starts at 1.
      // Positive indices need to subtract 1.
      start_ = static_cast<cudf::size_type>(startValue - 1);
    }

    if (expr->inputs().size() > 2) {
      auto lengthExpr =
          std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
      VELOX_CHECK_NOT_NULL(lengthExpr, "substr length must be a constant");

      auto lengthValue =
          lengthExpr->value()->as<SimpleVector<int64_t>>()->valueAt(0);
      // cuDF uses indices [begin, end).
      // Presto uses length as the length of the substring.
      // We compute the end as start + length.
      end_ = start_ + static_cast<cudf::size_type>(lengthValue);
      hasEnd_ = true;
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    cudf::numeric_scalar<cudf::size_type> startScalar(start_, true, stream, mr);
    cudf::numeric_scalar<cudf::size_type> endScalar(
        hasEnd_ ? end_ : 0, hasEnd_, stream, mr);
    cudf::numeric_scalar<cudf::size_type> stepScalar(1, true, stream, mr);
    return cudf::strings::slice_strings(
        inputCol, startScalar, endScalar, stepScalar, stream, mr);
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
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SubstrFunction>(expr);
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
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<prestosql::DatePlusIntervalFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .argumentType("interval day to second")
           .build()});
}

} // namespace facebook::velox::cudf_velox
