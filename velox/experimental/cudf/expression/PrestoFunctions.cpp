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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/CommonFunctions.h"
#include "velox/experimental/cudf/expression/DateTruncFunction.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/NullMask.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/prestosql/ArrayJoinFunction.h"
#include "velox/experimental/cudf/expression/prestosql/DateAddFunction.h"
#include "velox/experimental/cudf/expression/prestosql/DatePlusIntervalFunction.h"
#include "velox/experimental/cudf/expression/prestosql/UnixtimeFunctions.h"
#include "velox/experimental/cudf/expression/prestosql/VectorDistanceFunctions.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>

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
      cuda::stream_ref stream,
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

// Presto strpos(string, substring): 1-based character position of the first
// occurrence of substring in string, or 0 when it is not present. Backed by
// cudf::strings::find, which returns the 0-based character position (-1 when
// not found); adding one reproduces Presto semantics for both the found and
// not-found cases. Positions are character (code-point) based on both sides,
// matching Presto. A constant substring uses the string_scalar overload (no
// target column is materialised); a column substring uses the column overload.
class StrposFunction : public CudfFunction {
 public:
  StrposFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "strpos expects exactly 2 inputs");

    if (expr->inputs()[0]->isConstantKind()) {
      stringIsConstant_ = true;
      auto stringValue = toConstantVector(expr->inputs()[0], pool);
      stringIsNull_ = stringValue->isNullAt(0);
      if (!stringIsNull_) {
        string_ = stringValue->toString(0);
      }
    }

    if (expr->inputs()[1]->isConstantKind()) {
      targetIsConstant_ = true;
      auto targetValue = toConstantVector(expr->inputs()[1], pool);
      targetIsNull_ = targetValue->isNullAt(0);
      if (!targetIsNull_) {
        target_ = targetValue->toString(0);
      }
    }

    VELOX_CHECK(
        !(stringIsConstant_ && targetIsConstant_),
        "strpos with a constant string and substring is not supported by the "
        "cuDF evaluator because there is no input column to derive the output "
        "row count from");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(
        !inputColumns.empty(),
        "strpos requires at least one non-literal input column");
    // inputColumns holds only non-literal children, so the first entry
    // determines the output row count even when the string argument is
    // constant.
    const auto outputRowCount = asView(inputColumns[0]).size();
    const auto tempMr = get_temp_mr();
    const auto makeAllNullPositions = [&]() {
      const auto nullPosition =
          cudf::numeric_scalar<int64_t>(0, false, stream, tempMr);
      return cudf::make_column_from_scalar(
          nullPosition, outputRowCount, stream, mr);
    };

    // strpos(NULL, x) and strpos(x, NULL) are NULL for every row;
    // cudf::strings::find has no null-operand overloads, so produce an all-null
    // bigint column directly.
    if ((stringIsConstant_ && stringIsNull_) ||
        (targetIsConstant_ && targetIsNull_)) {
      return makeAllNullPositions();
    }

    size_t nextInput = 0;
    std::unique_ptr<cudf::column> stringHolder;
    cudf::column_view stringColumn;
    if (stringIsConstant_) {
      // A null constant string returned the all-null result above.
      cudf::string_scalar stringScalar(string_, true, stream, tempMr);
      stringHolder = cudf::make_column_from_scalar(
          stringScalar, outputRowCount, stream, tempMr);
      stringColumn = stringHolder->view();
    } else {
      stringColumn = asView(inputColumns[nextInput++]);
    }
    const cudf::strings_column_view stringsView(stringColumn);

    // 0-based character position of the first match, -1 when not found. Nulls
    // in either operand propagate to a null result, matching Velox.
    std::unique_ptr<cudf::column> found;
    if (targetIsConstant_) {
      cudf::string_scalar targetScalar(target_, true, stream, tempMr);
      found =
          cudf::strings::find(stringsView, targetScalar, 0, -1, stream, tempMr);
    } else {
      const cudf::strings_column_view targetView(
          asView(inputColumns[nextInput]));
      // TODO: Remove the next two guards once the pinned cuDF includes the fix
      // for https://github.com/NVIDIA/cudf/issues/25231.
      // With a substring column, cudf::strings::find returns rows that are
      // marked valid and hold uninitialized memory when every string is null.
      if (stringColumn.null_count() == stringColumn.size()) {
        return makeAllNullPositions();
      }
      // It also reports every substring as not found when the substring column
      // holds no characters, that is, when every non-null substring is empty.
      // Presto finds an empty substring at position 1. The check reads the
      // host-side data pointer, so it needs no device synchronization.
      if (targetView.chars_begin(stream) == nullptr) {
        const auto firstPosition =
            cudf::numeric_scalar<int64_t>(1, true, stream, tempMr);
        auto positions = cudf::make_column_from_scalar(
            firstPosition, outputRowCount, stream, mr);
        mergeNullSourceNullsIntoResult(
            *positions, {stringColumn, targetView.parent()}, stream, mr);
        return positions;
      }
      found = cudf::strings::find(stringsView, targetView, 0, stream, tempMr);
    }

    // Shift to Presto's 1-based position (not-found -1 becomes 0) and widen to
    // bigint in a single kernel. Adding a valid scalar preserves nulls, so null
    // rows stay null.
    const auto one = cudf::numeric_scalar<int32_t>(1, true, stream, tempMr);
    return cudf::binary_operation(
        found->view(),
        one,
        cudf::binary_operator::ADD,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);
  }

 private:
  // Set when the string argument is a constant; the value and its nullness are
  // read once at construction.
  bool stringIsConstant_{false};
  bool stringIsNull_{false};
  // Set when the substring argument is a constant; the value and its nullness
  // are read once at construction.
  bool targetIsConstant_{false};
  bool targetIsNull_{false};
  // Constant string / substring values, valid only when the corresponding
  // *IsConstant_ flag is set and the value is not null.
  std::string string_;
  std::string target_;
};

} // namespace

void registerPrestoFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  // Presto special form functions don't have a prefix.
  registerCudfFunction(
      "is_null",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*,
         const core::QueryConfig&) {
        return std::make_shared<IsNullFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("boolean")
           .argumentType("T")
           .build()});

  // CastFunction's cast from VARCHAR and cast from TIMESTAMP to VARCHAR follow
  // Presto's rules.
  setPrestoCastEnabled(true);
  registerPrestoArrayAccessFunctions(prefix);
  prestosql::registerVectorDistanceFunctions(prefix);

  registerCudfFunction(
      prefix + "array_join",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
        return std::make_shared<prestosql::ArrayJoinFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("array(varchar)")
           .constantArgumentType("varchar")
           .build()},
      true,
      [](const core::TypedExprPtr& expr, const core::QueryConfig&) {
        return prestosql::ArrayJoinFunction::canEvaluate(expr);
      });

  registerCudfFunctions(
      {prefix + "substr", prefix + "substring"},
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
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
      prefix + "strpos",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
        return std::make_shared<StrposFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "plus",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
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
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
        return std::make_shared<prestosql::DateAddFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .constantArgumentType("varchar")
           .argumentType("bigint")
           .argumentType("date")
           .build()},
      true,
      [](const core::TypedExprPtr& expr, const core::QueryConfig&) {
        return prestosql::DateAddFunction::canEvaluate(expr);
      });

  registerCudfFunction(
      prefix + "date_trunc",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
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
      [](const core::TypedExprPtr& expr, const core::QueryConfig&) {
        return DateTruncFunction::canEvaluate(expr);
      });

  registerCudfFunction(
      prefix + "from_unixtime",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
        return std::make_shared<prestosql::FromUnixtimeFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp")
           .argumentType("double")
           .build()},
      true,
      [](const core::TypedExprPtr& expr, const core::QueryConfig&) {
        return prestosql::FromUnixtimeFunction::canEvaluate(expr);
      });

  registerCudfFunction(
      prefix + "to_unixtime",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
        return std::make_shared<prestosql::ToUnixtimeFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("timestamp")
           .build()},
      true,
      [](const core::TypedExprPtr& expr, const core::QueryConfig&) {
        return prestosql::ToUnixtimeFunction::canEvaluate(expr);
      });
}

} // namespace facebook::velox::cudf_velox
