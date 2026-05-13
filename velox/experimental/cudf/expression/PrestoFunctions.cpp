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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionSignature.h"

#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

namespace facebook::velox::cudf_velox {
namespace {

template <typename T>
bool hasIntegralZero(
    const cudf::column_view& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (col.is_empty()) {
    return false;
  }

  cudf::numeric_scalar<T> zero{0, true, stream, mr};
  auto equals = cudf::binary_operation(
      col,
      zero,
      cudf::binary_operator::EQUAL,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto anyAgg = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto anyScalar = cudf::reduce(
      equals->view(),
      *anyAgg,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto const& boolScalar =
      static_cast<cudf::numeric_scalar<bool> const&>(*anyScalar);
  return boolScalar.is_valid(stream) && boolScalar.value(stream);
}

class IntegralModFunction : public CudfFunction {
 public:
  explicit IntegralModFunction(const std::shared_ptr<velox::exec::Expr>& expr)
      : type_(cudf_velox::veloxToCudfDataType(expr->type())) {
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "mod expects exactly 2 inputs");
    if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[0])) {
      auto constValue = constExpr->value();
      left_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    } else if (
        auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[1])) {
      auto constValue = constExpr->value();
      right_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    }

    VELOX_CHECK(
        !(left_ != nullptr && right_ != nullptr),
        "mod on two literals is not supported");
    VELOX_CHECK(cudf::is_integral(type_), "mod only supports integral types");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    if (left_ == nullptr && right_ == nullptr) {
      auto rhs = asView(inputColumns[1]);
      switch (type_.id()) {
        case cudf::type_id::INT8:
          VELOX_USER_CHECK(
              !hasIntegralZero<int8_t>(rhs, stream, mr), "Cannot divide by 0");
          break;
        case cudf::type_id::INT16:
          VELOX_USER_CHECK(
              !hasIntegralZero<int16_t>(rhs, stream, mr), "Cannot divide by 0");
          break;
        case cudf::type_id::INT32:
          VELOX_USER_CHECK(
              !hasIntegralZero<int32_t>(rhs, stream, mr), "Cannot divide by 0");
          break;
        case cudf::type_id::INT64:
          VELOX_USER_CHECK(
              !hasIntegralZero<int64_t>(rhs, stream, mr), "Cannot divide by 0");
          break;
        default:
          VELOX_FAIL("Unsupported type for integral mod");
      }
      return integralCheckedModulus(type_, asView(inputColumns[0]), rhs, stream, mr);
    }
    if (left_ == nullptr) {
      switch (type_.id()) {
        case cudf::type_id::INT8:
          VELOX_USER_CHECK(
              static_cast<cudf::numeric_scalar<int8_t> const&>(*right_)
                      .value(stream) !=
                  0,
              "Cannot divide by 0");
          break;
        case cudf::type_id::INT16:
          VELOX_USER_CHECK(
              static_cast<cudf::numeric_scalar<int16_t> const&>(*right_)
                      .value(stream) !=
                  0,
              "Cannot divide by 0");
          break;
        case cudf::type_id::INT32:
          VELOX_USER_CHECK(
              static_cast<cudf::numeric_scalar<int32_t> const&>(*right_)
                      .value(stream) !=
                  0,
              "Cannot divide by 0");
          break;
        case cudf::type_id::INT64:
          VELOX_USER_CHECK(
              static_cast<cudf::numeric_scalar<int64_t> const&>(*right_)
                      .value(stream) !=
                  0,
              "Cannot divide by 0");
          break;
        default:
          VELOX_FAIL("Unsupported type for integral mod");
      }
      return integralCheckedModulus(type_, asView(inputColumns[0]), *right_, stream, mr);
    }

    auto rhs = asView(inputColumns[0]);
    switch (type_.id()) {
      case cudf::type_id::INT8:
        VELOX_USER_CHECK(
            !hasIntegralZero<int8_t>(rhs, stream, mr), "Cannot divide by 0");
        break;
      case cudf::type_id::INT16:
        VELOX_USER_CHECK(
            !hasIntegralZero<int16_t>(rhs, stream, mr), "Cannot divide by 0");
        break;
      case cudf::type_id::INT32:
        VELOX_USER_CHECK(
            !hasIntegralZero<int32_t>(rhs, stream, mr), "Cannot divide by 0");
        break;
      case cudf::type_id::INT64:
        VELOX_USER_CHECK(
            !hasIntegralZero<int64_t>(rhs, stream, mr), "Cannot divide by 0");
        break;
      default:
        VELOX_FAIL("Unsupported type for integral mod");
    }
    return integralCheckedModulus(type_, *left_, rhs, stream, mr);
  }

 private:
  const cudf::data_type type_;
  std::unique_ptr<cudf::scalar> left_;
  std::unique_ptr<cudf::scalar> right_;
};

} // namespace

void registerPrestoFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "mod",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<IntegralModFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("tinyint")
           .argumentType("tinyint")
           .argumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("smallint")
           .argumentType("smallint")
           .argumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .argumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .argumentType("bigint")
           .build()});
}

} // namespace facebook::velox::cudf_velox
