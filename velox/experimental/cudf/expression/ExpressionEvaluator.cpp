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
#include "velox/experimental/cudf/exec/Validation.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/hashing.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/replace.hpp>
#include <cudf/round.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {
namespace {

struct CudfExpressionEvaluatorEntry {
  int priority;
  CudfExpressionEvaluatorCanEvaluate canEvaluate;
  CudfExpressionEvaluatorCreate create;
};

static std::unordered_map<std::string, CudfExpressionEvaluatorEntry>&
getCudfExpressionEvaluatorRegistry() {
  static std::unordered_map<std::string, CudfExpressionEvaluatorEntry> registry;
  return registry;
}

static void ensureBuiltinExpressionEvaluatorsRegistered() {
  static bool registered = false;
  if (registered) {
    return;
  }

  // Default priority for function evaluator
  const int kFunctionPriority = 50;

  // Function evaluator
  registerCudfExpressionEvaluator(
      "function",
      kFunctionPriority,
      [](std::shared_ptr<velox::exec::Expr> expr) {
        return FunctionExpression::canEvaluate(std::move(expr));
      },
      [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
        return FunctionExpression::create(std::move(expr), row);
      },
      /*overwrite=*/false);

  registered = true;
}

} // namespace

bool registerCudfExpressionEvaluator(
    const std::string& name,
    int priority,
    CudfExpressionEvaluatorCanEvaluate canEvaluate,
    CudfExpressionEvaluatorCreate create,
    bool overwrite) {
  auto& registry = getCudfExpressionEvaluatorRegistry();
  if (!overwrite && registry.find(name) != registry.end()) {
    return false;
  }
  registry[name] = CudfExpressionEvaluatorEntry{
      priority, std::move(canEvaluate), std::move(create)};
  return true;
}

std::unordered_map<std::string, CudfFunctionSpec>& getCudfFunctionRegistry() {
  static std::unordered_map<std::string, CudfFunctionSpec> registry;
  return registry;
}

namespace {

static bool matchCallAgainstSignatures(
    const velox::exec::Expr& call,
    const std::vector<exec::FunctionSignaturePtr>& sigs) {
  const auto n = call.inputs().size();
  std::vector<TypePtr> argTypes;
  argTypes.reserve(n);
  for (const auto& in : call.inputs()) {
    argTypes.push_back(in->type());
  }
  for (const auto& sig : sigs) {
    exec::SignatureBinder binder(*sig, argTypes);
    if (!binder.tryBind()) {
      continue;
    }
    // binder does not confirm whether positional arguments are
    // constants(scalars) as expected. we have to check manually
    const auto& constArgs = sig->constantArguments();
    const size_t fixed = std::min(constArgs.size(), n);
    bool ok = true;
    for (size_t i = 0; i < fixed; ++i) {
      if (constArgs[i] && call.inputs()[i]->name() != "literal") {
        ok = false;
        break;
      }
    }
    if (!ok) {
      continue;
    }
    return true;
  }
  return false;
}

} // namespace

class SplitFunction : public CudfFunction {
 public:
  SplitFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto delimiterExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(delimiterExpr, "split delimiter must be a constant");
    delimiterScalar_ = std::make_unique<cudf::string_scalar>(
        delimiterExpr->value()->toString(0), true, stream, mr);

    auto limitExpr =
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[2]);
    VELOX_CHECK_NOT_NULL(limitExpr, "split limit must be a constant");
    maxSplitCount_ = std::stoll(limitExpr->value()->toString(0));

    // Presto specifies maxSplitCount as the maximum size of the returned array
    // while cuDF understands the parameter as how many splits can it perform.
    maxSplitCount_ -= 1;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::split_record(
        inputCol, *delimiterScalar_, maxSplitCount_, stream, mr);
  };

 private:
  std::unique_ptr<cudf::string_scalar> delimiterScalar_;
  cudf::size_type maxSplitCount_;
};

class CastFunction : public CudfFunction {
 public:
  CastFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "cast expects exactly 1 input");

    targetCudfType_ =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(expr->type()));
    auto sourceType = cudf::data_type(
        cudf_velox::veloxToCudfTypeId(expr->inputs()[0]->type()));
    VELOX_CHECK(
        cudf::is_supported_cast(sourceType, targetCudfType_),
        "Cast from {} to {} is not supported",
        expr->inputs()[0]->type()->toString(),
        expr->type()->toString());
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::cast(inputCol, targetCudfType_, stream, mr);
  }

 private:
  cudf::data_type targetCudfType_;
};

// Spark date_add function implementation.
// For the presto date_add, the first value is unit string,
// may need to get the function with prefix, if the prefix is "", it is Spark
// function.
class DateAddFunction : public CudfFunction {
 public:
  DateAddFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "date_add function expects exactly 2 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->isDate(),
        "First argument to date_add must be a date");
    VELOX_CHECK_NULL(
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[0]));
    // The date_add second argument could be int8_t, int16_t, int32_t.
    value_ = makeScalarFromConstantExpr(
        expr->inputs()[1], cudf::type_id::DURATION_DAYS);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::binary_operation(
        inputCol,
        *value_,
        cudf::binary_operator::ADD,
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        stream,
        mr);
  }

 private:
  std::unique_ptr<cudf::scalar> value_;
};

class CardinalityFunction : public CudfFunction {
 public:
  CardinalityFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    // Cardinality doesn't need any pre-computed scalars, just validates input
    // count
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "cardinality expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::lists::count_elements(inputCol, stream, mr);
  }
};

class RoundFunction : public CudfFunction {
 public:
  explicit RoundFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    const auto argSize = expr->inputs().size();
    VELOX_CHECK(argSize >= 1 && argSize <= 2, "round expects 1 or 2 inputs");
    VELOX_CHECK_NULL(
        std::dynamic_pointer_cast<exec::ConstantExpr>(expr->inputs()[0]),
        "round expects first column is not literal");
    if (argSize == 2) {
      auto scaleExpr =
          std::dynamic_pointer_cast<exec::ConstantExpr>(expr->inputs()[1]);
      VELOX_CHECK_NOT_NULL(scaleExpr, "round scale must be a constant");
      scale_ = scaleExpr->value()->as<SimpleVector<int32_t>>()->valueAt(0);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::round_decimal(
        asView(inputColumns[0]),
        scale_,
        cudf::rounding_method::HALF_UP,
        stream,
        mr);
    ;
  }

 private:
  int32_t scale_ = 0;
};

class BinaryFunction : public CudfFunction {
 public:
  static cudf::data_type makeOutputType(const TypePtr& type) {
    auto typeId = cudf_velox::veloxToCudfTypeId(type);
    if (type->isDecimal()) {
      // Velox scale is positive for fractional digits; cuDF expects negative.
      auto scale = getDecimalPrecisionScale(*type).second;
      return cudf::data_type(typeId, -static_cast<int32_t>(scale));
    }
    return cudf::data_type(typeId);
  }

  BinaryFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      cudf::binary_operator op)
      : op_(op), type_(makeOutputType(expr->type())) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "binary function expects exactly 2 inputs");
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
        "Not support both left and right are literals");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    if (left_ == nullptr && right_ == nullptr) {
      // Ensure decimal promotion is respected by casting inputs to the output
      // decimal type (e.g. DECIMAL64 -> DECIMAL128) before multiplication.
      auto lhsView = asView(inputColumns[0]);
      auto rhsView = asView(inputColumns[1]);
      std::unique_ptr<cudf::column> lhsCast;
      std::unique_ptr<cudf::column> rhsCast;
      if (type_.id() == cudf::type_id::DECIMAL128) {
        if (lhsView.type() != type_) {
          lhsCast = cudf::cast(lhsView, type_, stream, mr);
          lhsView = lhsCast->view();
        }
        if (rhsView.type() != type_) {
          rhsCast = cudf::cast(rhsView, type_, stream, mr);
          rhsView = rhsCast->view();
        }
      }
      return cudf::binary_operation(
          lhsView,
          rhsView,
          op_,
          type_,
          stream,
          mr);
    } else if (left_ == nullptr) {
      return cudf::binary_operation(
          asView(inputColumns[0]), *right_, op_, type_, stream, mr);
    }
    return cudf::binary_operation(
        *left_, asView(inputColumns[0]), op_, type_, stream, mr);
  }

 private:
  const cudf::binary_operator op_;
  const cudf::data_type type_;
  std::unique_ptr<cudf::scalar> left_;
  std::unique_ptr<cudf::scalar> right_;
};

class SwitchFunction : public CudfFunction {
 public:
  SwitchFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 3, "case when expects exactly 3 inputs");
    VELOX_CHECK_EQ(
        expr->inputs()[0]->type()->kind(),
        TypeKind::BOOLEAN,
        "The switch condition result type should be boolean");
    VELOX_CHECK_NULL(
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr),
        "The condition should not be constant");
    if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[1])) {
      auto constValue = constExpr->value();
      left_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    }
    if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[2])) {
      auto constValue = constExpr->value();
      right_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    if (left_ == nullptr && right_ == nullptr) {
      return cudf::copy_if_else(
          asView(inputColumns[1]),
          asView(inputColumns[2]),
          asView(inputColumns[0]),
          stream,
          mr);
    } else if (left_ == nullptr) {
      return cudf::copy_if_else(
          asView(inputColumns[1]),
          *right_,
          asView(inputColumns[0]),
          stream,
          mr);
    } else if (right_ == nullptr) {
      return cudf::copy_if_else(
          *left_, asView(inputColumns[1]), asView(inputColumns[0]), stream, mr);
    }
    // right != null and left != null
    return cudf::copy_if_else(
        *left_, *right_, asView(inputColumns[0]), stream, mr);
  }

 private:
  std::unique_ptr<cudf::scalar> left_;
  std::unique_ptr<cudf::scalar> right_;
};

class SubstrFunction : public CudfFunction {
 public:
  SubstrFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "substr expects at least 2 inputs");
    VELOX_CHECK_LE(expr->inputs().size(), 3, "substr expects at most 3 inputs");

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto startExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(startExpr, "substr start must be a constant");

    auto startValue =
        startExpr->value()->as<SimpleVector<int64_t>>()->valueAt(0);
    cudf::size_type adjustedStart = static_cast<cudf::size_type>(startValue);
    if (startValue >= 1) {
      // cuDF indexing starts at 0.
      // Presto indexing starts at 1.
      // Positive indices need to substract 1.
      adjustedStart = static_cast<cudf::size_type>(startValue - 1);
    }

    startScalar_ = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(
        adjustedStart, true, stream, mr);

    if (expr->inputs().size() > 2) {
      auto lengthExpr =
          std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
      VELOX_CHECK_NOT_NULL(lengthExpr, "substr length must be a constant");

      auto lengthValue =
          lengthExpr->value()->as<SimpleVector<int64_t>>()->valueAt(0);
      // cuDF uses indices [begin, end).
      // Presto uses length as the length of the substring.
      // We compute the end as start + length.
      cudf::size_type endPosition =
          adjustedStart + static_cast<cudf::size_type>(lengthValue);

      endScalar_ = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(
          endPosition, true, stream, mr);
    } else {
      endScalar_ = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(
          0, false, stream, mr);
    }

    stepScalar_ = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(
        1, true, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::slice_strings(
        inputCol, *startScalar_, *endScalar_, *stepScalar_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::numeric_scalar<cudf::size_type>> startScalar_;
  std::unique_ptr<cudf::numeric_scalar<cudf::size_type>> endScalar_;
  std::unique_ptr<cudf::numeric_scalar<cudf::size_type>> stepScalar_;
};

class CoalesceFunction : public CudfFunction {
 public:
  CoalesceFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    // Storing the first literal that appears in inputs because we don't need to
    // process after that. This is the last fallback.
    numColumnsBeforeLiteral_ = expr->inputs().size();
    for (size_t i = 0; i < expr->inputs().size(); ++i) {
      const auto& input = expr->inputs()[i];
      if (auto c =
              std::dynamic_pointer_cast<velox::exec::ConstantExpr>(input)) {
        if (!c->value()->isNullAt(0)) {
          literalScalar_ = makeScalarFromConstantExpr(c);
          numColumnsBeforeLiteral_ = i;
          break;
        }
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    // Coalesce is practically a cudf::replace_nulls over multiple columns.
    // Starting from first column, we keep calling replace nulls with
    // subsequent cols until we get an all valid col or run out of columns

    // If a literal comes before any column input, fill the result with it.
    if (literalScalar_ && numColumnsBeforeLiteral_ == 0) {
      if (inputColumns.empty()) {
        // We need at least one column to tell us the required output size
        VELOX_NYI("coalesce with only literal inputs is not supported");
      }
      auto size = asView(inputColumns[0]).size();
      return cudf::make_column_from_scalar(*literalScalar_, size, stream, mr);
    }

    VELOX_CHECK(
        !inputColumns.empty(),
        "coalesce requires at least one non-literal input");
    ColumnOrView result = asView(inputColumns[0]);
    size_t stop = std::min(numColumnsBeforeLiteral_, inputColumns.size());
    for (size_t i = 1; i < stop && asView(result).has_nulls(); ++i) {
      result = cudf::replace_nulls(
          asView(result), asView(inputColumns[i]), stream, mr);
    }

    if (literalScalar_ && asView(result).has_nulls()) {
      result = cudf::replace_nulls(asView(result), *literalScalar_, stream, mr);
    }

    return result;
  }

 private:
  size_t numColumnsBeforeLiteral_;
  std::unique_ptr<cudf::scalar> literalScalar_;
};

class HashFunction : public CudfFunction {
 public:
  HashFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_GE(expr->inputs().size(), 2, "hash expects at least 2 inputs");
    auto seedExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(seedExpr, "hash seed must be a constant");
    int32_t seedValue =
        seedExpr->value()->as<SimpleVector<int32_t>>()->valueAt(0);
    VELOX_CHECK_GE(seedValue, 0);
    seedValue_ = seedValue;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(!inputColumns.empty());
    auto inputTableView = convertToTableView(inputColumns);
    return cudf::hashing::murmurhash3_x86_32(
        inputTableView, seedValue_, stream, mr);
  }

 private:
  static cudf::table_view convertToTableView(
      std::vector<ColumnOrView>& inputColumns) {
    std::vector<cudf::column_view> columns;
    columns.reserve(inputColumns.size());

    for (auto& col : inputColumns) {
      columns.push_back(asView(col));
    }

    return cudf::table_view(columns);
  }

  uint32_t seedValue_;
};

class YearFunction : public CudfFunction {
 public:
  explicit YearFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "year expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::extract_datetime_component(
        inputCol, cudf::datetime::datetime_component::YEAR, stream, mr);
  }
};

class LengthFunction : public CudfFunction {
 public:
  explicit LengthFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "length expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::count_characters(inputCol, stream, mr);
  }
};

class LowerFunction : public CudfFunction {
 public:
  explicit LowerFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "lower expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::to_lower(inputCol, stream, mr);
  }
};

class UpperFunction : public CudfFunction {
 public:
  explicit UpperFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "upper expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::to_upper(inputCol, stream, mr);
  }
};

class LikeFunction : public CudfFunction {
 public:
  explicit LikeFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "like expects 2 inputs");

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(patternExpr, "like pattern must be a constant");
    pattern_ = patternExpr->value()->toString(0);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::like(
        inputCol, pattern_, std::string_view(""), stream, mr);
  }

 private:
  std::string pattern_;
};

class StartswithFunction : public CudfFunction {
 public:
  explicit StartswithFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "startswith expects 2 inputs");

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(patternExpr, "startswith pattern must be a constant");
    pattern_ = std::make_unique<cudf::string_scalar>(
        patternExpr->value()->toString(0), true, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::starts_with(inputCol, *pattern_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::string_scalar> pattern_;
};

class EndswithFunction : public CudfFunction {
 public:
  explicit EndswithFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "endswith expects 2 inputs");

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(patternExpr, "endswith pattern must be a constant");
    pattern_ = std::make_unique<cudf::string_scalar>(
        patternExpr->value()->toString(0), true, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::ends_with(inputCol, *pattern_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::string_scalar> pattern_;
};

class ContainsFunction : public CudfFunction {
 public:
  explicit ContainsFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "contains expects 2 inputs");

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(patternExpr, "contains pattern must be a constant");
    pattern_ = std::make_unique<cudf::string_scalar>(
        patternExpr->value()->toString(0), true, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::contains(inputCol, *pattern_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::string_scalar> pattern_;
};

bool registerCudfFunction(
    const std::string& name,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite) {
  auto& registry = getCudfFunctionRegistry();
  if (!overwrite && registry.find(name) != registry.end()) {
    return false;
  }
  registry[name] = CudfFunctionSpec{std::move(factory), signatures};
  return true;
}

void registerCudfFunctions(
    const std::vector<std::string>& aliases,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite) {
  for (const auto& name : aliases) {
    registerCudfFunction(name, factory, signatures, overwrite);
  }
}

std::shared_ptr<CudfFunction> createCudfFunction(
    const std::string& name,
    const std::shared_ptr<velox::exec::Expr>& expr) {
  auto& registry = getCudfFunctionRegistry();
  auto it = registry.find(name);
  if (it != registry.end()) {
    return it->second.factory(name, expr);
  }
  return nullptr;
}

bool registerBuiltinFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "split",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SplitFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("array(varchar)")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("array(varchar)")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           // cuDF expects cudf::size_type (int32) but we may get bigint from
           // presto. SplitFunction hacks around this by converting to string
           // and back
           .constantArgumentType("bigint")
           .build()});

  registerCudfFunction(
      prefix + "cardinality",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<CardinalityFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("array(any)")
           .build()});

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

  // Coalesce is special form and doesn't have a prefix in its name.
  registerCudfFunction(
      "coalesce",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<CoalesceFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("T")
           .argumentType("T")
           .variableArity("T")
           .build()});

  registerCudfFunction(
      prefix + "hash_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<HashFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("integer")
           .argumentType("any")
           .variableArity()
           .build()});

  registerCudfFunction(
      prefix + "round",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<RoundFunction>(expr);
      },
      {
        FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .build(),
        FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .constantArgumentType("integer")
           .build(),
        FunctionSignatureBuilder()
           .returnType("tinyint")
           .argumentType("tinyint")
           .build(),
        FunctionSignatureBuilder()
           .returnType("tinyint")
           .argumentType("tinyint")
           .constantArgumentType("integer")
           .build(),
        FunctionSignatureBuilder()
           .returnType("smallint")
           .argumentType("smallint")
           .build(),
        FunctionSignatureBuilder()
           .returnType("smallint")
           .argumentType("smallint")
           .constantArgumentType("integer")
           .build(),
        FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .build(),
        FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .constantArgumentType("integer")
           .build(),
        FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .build(),
        FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .constantArgumentType("integer")
           .build()});

  registerCudfFunction(
      prefix + "year",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<YearFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("timestamp")
           .build(),
       FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("date")
           .build()});

  registerCudfFunction(
      prefix + "length",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LengthFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "lower",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LowerFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "upper",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<UpperFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "like",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LikeFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  // Our cudf binary ops can take all numeric types but instead of listing them
  // all, we're testing if input types can be casted to double. Coersion will
  // pass because all numerics can be casted to double.
  // TODO (dm): This could break for decimal

  auto registerBinaryOp = [&](const std::vector<std::string>& aliases, cudf::binary_operator op) {
    registerCudfFunctions(
        aliases,
        [op](const std::string&,
             const std::shared_ptr<velox::exec::Expr>& expr) {
          return std::make_shared<BinaryFunction>(expr, op);
        },
        {FunctionSignatureBuilder()
             .returnType("double")
             .argumentType("double")
             .argumentType("double")
             .build(),
         FunctionSignatureBuilder()
             .integerVariable("p")
             .integerVariable("s")
             .returnType("decimal(p,s)")
             .argumentType("decimal(p,s)")
             .argumentType("decimal(p,s)")
             .build()});
  };
  
  registerBinaryOp({prefix + "plus", prefix + "add"}, cudf::binary_operator::ADD);
  registerBinaryOp({prefix + "minus", prefix + "subtract"}, cudf::binary_operator::SUB);
  registerBinaryOp({prefix + "multiply"}, cudf::binary_operator::MUL);
  registerBinaryOp({prefix + "divide"}, cudf::binary_operator::DIV);
  registerBinaryOp({prefix + "mod"}, cudf::binary_operator::MOD);

  auto registerComparisonOp = [&](const std::vector<std::string>& aliases, cudf::binary_operator op) {
    registerCudfFunctions(
        aliases,
        [op](const std::string&,
             const std::shared_ptr<velox::exec::Expr>& expr) {
          return std::make_shared<BinaryFunction>(expr, op);
        },
        {FunctionSignatureBuilder()
             .returnType("boolean")
             .argumentType("double")
             .argumentType("double")
             .build(),
         FunctionSignatureBuilder()
             .integerVariable("p")
             .integerVariable("s")
             .returnType("boolean")
             .argumentType("decimal(p,s)")
             .argumentType("decimal(p,s)")
             .build()});
  };

  registerComparisonOp({prefix + "equal", prefix + "eq"}, cudf::binary_operator::EQUAL);
  registerComparisonOp({prefix + "notequal", prefix + "neq"}, cudf::binary_operator::NOT_EQUAL);
  registerComparisonOp({prefix + "greaterthanorequal", prefix + "gte"}, cudf::binary_operator::GREATER_EQUAL);
  registerComparisonOp({prefix + "lessthanorequal", prefix + "lte"}, cudf::binary_operator::LESS_EQUAL);
  registerComparisonOp({prefix + "greaterthan", prefix + "gt"}, cudf::binary_operator::GREATER);
  registerComparisonOp({prefix + "lessthan", prefix + "lt"}, cudf::binary_operator::LESS);

  // No prefix because switch and if are special form
  registerCudfFunctions(
      {"switch", "if"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SwitchFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("T")
           .argumentType("boolean")
           .argumentType("T")
           .argumentType("T")
           .build()});

  registerCudfFunctions(
      // No signatures required for cast and try_cast. They are special forms.
      {"try_cast", "cast"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<CastFunction>(expr);
      },
      {
          // Cast needs special handling dynamically using cudf.
      });

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DateAddFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("integer")
           .build()});

  return true;
}

std::shared_ptr<FunctionExpression> FunctionExpression::create(
    const std::shared_ptr<velox::exec::Expr>& expr,
    const RowTypePtr& inputRowSchema) {
  auto node = std::make_shared<FunctionExpression>();
  node->expr_ = expr;
  node->inputRowSchema_ = inputRowSchema;

  auto name = expr->name();
  node->function_ = createCudfFunction(name, expr);

  if (node->function_) {
    for (const auto& input : expr->inputs()) {
      if (input->name() != "literal") {
        node->subexpressions_.push_back(
            createCudfExpression(input, inputRowSchema));
      }
    }
  }

  return node;
}

ColumnOrView FunctionExpression::eval(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  using velox::exec::FieldReference;

  if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr_)) {
    auto name = fieldExpr->name();
    auto columnIndex = inputRowSchema_->getChildIdx(name);
    return inputTableColumns[columnIndex]->view();
  }

  if (function_) {
    std::vector<ColumnOrView> inputColumns;
    inputColumns.reserve(subexpressions_.size());

    for (const auto& subexpr : subexpressions_) {
      inputColumns.push_back(subexpr->eval(inputTableColumns, stream, mr));
    }

    auto result = function_->eval(inputColumns, stream, mr);
    if (finalize) {
      auto requestedType =
          cudf::data_type(cudf_velox::veloxToCudfTypeId(expr_->type()));
      if (expr_->type()->isDecimal()) {
        auto scale = getDecimalPrecisionScale(*expr_->type()).second;
        requestedType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(expr_->type()),
            -static_cast<int32_t>(scale));
      }
      auto resultView = asView(result);
      if (resultView.type() != requestedType) {
        return cudf::cast(resultView, requestedType, stream, mr);
      }
    }
    return result;
  }

  VELOX_FAIL(
      "Unsupported expression for recursive evaluation: " + expr_->name());
}

void FunctionExpression::close() {
  function_.reset();
  subexpressions_.clear();
}

bool FunctionExpression::canEvaluate(std::shared_ptr<velox::exec::Expr> expr) {
  using velox::exec::FieldReference;

  if (std::dynamic_pointer_cast<FieldReference>(expr)) {
    return true;
  }

  const auto& opName = expr->name();
  if (opName == "cast" || opName == "try_cast") {
    const auto& srcType =
        expr->inputs().empty() ? nullptr : expr->inputs()[0]->type();
    const auto& dstType = expr->type();
    if (srcType == nullptr || dstType == nullptr) {
      return false;
    }
    auto src = cudf::data_type(cudf_velox::veloxToCudfTypeId(srcType));
    auto dst = cudf::data_type(cudf_velox::veloxToCudfTypeId(dstType));
    return cudf::is_supported_cast(src, dst);
  }

  auto& registry = getCudfFunctionRegistry();
  auto it = registry.find(expr->name());
  if (it == registry.end()) {
    return false;
  }
  const auto& spec = it->second;
  return matchCallAgainstSignatures(*expr, spec.signatures);
}

bool canBeEvaluatedByCudf(std::shared_ptr<velox::exec::Expr> expr, bool deep) {
  ensureBuiltinExpressionEvaluatorsRegistered();
  const auto& registry = getCudfExpressionEvaluatorRegistry();

  bool supported = false;
  for (const auto& [name, entry] : registry) {
    if (entry.canEvaluate && entry.canEvaluate(expr)) {
      supported = true;
      break;
    }
  }
  if (!supported) {
    LOG_FALLBACK(expr->toString());
    return false;
  }

  if (deep) {
    for (const auto& input : expr->inputs()) {
      if (input->name() != "literal" && !canBeEvaluatedByCudf(input, true)) {
        return false;
      }
    }
  }

  return true;
}

std::shared_ptr<CudfExpression> createCudfExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema,
    std::optional<std::string> except) {
  ensureBuiltinExpressionEvaluatorsRegistered();
  const auto& registry = getCudfExpressionEvaluatorRegistry();

  const CudfExpressionEvaluatorEntry* best = nullptr;
  for (const auto& [name, entry] : registry) {
    if (except && name == *except) {
      continue;
    }
    if (entry.canEvaluate && entry.canEvaluate(expr)) {
      if (best == nullptr || entry.priority > best->priority) {
        best = &entry;
      }
    }
  }

  if (best != nullptr) {
    return best->create(expr, inputRowSchema);
  }

  return FunctionExpression::create(expr, inputRowSchema);
}

} // namespace facebook::velox::cudf_velox
