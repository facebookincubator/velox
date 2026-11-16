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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/CommonFunctions.h"
#include "velox/experimental/cudf/expression/DateTruncFunction.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/prestosql/ArrayJoinFunction.h"
#include "velox/experimental/cudf/expression/prestosql/DateAddFunction.h"
#include "velox/experimental/cudf/expression/prestosql/DatePlusIntervalFunction.h"
#include "velox/experimental/cudf/expression/prestosql/UnixtimeFunctions.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/copying.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>

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

// Returns the T (MapVector or ArrayVector) that holds the value of the non-null
// constant 'expr' with the row of that value, or nullptr if 'expr' holds none.
template <typename T>
std::pair<const T*, vector_size_t> constantCollection(
    const core::TypedExprPtr& expr) {
  if (!expr->isConstantKind()) {
    return {nullptr, 0};
  }
  const auto* constant = expr->asUnchecked<core::ConstantTypedExpr>();
  if (!constant->hasValueVector() || constant->isNull()) {
    return {nullptr, 0};
  }
  const auto& vector = constant->valueVector();
  return {vector->wrappedVector()->as<T>(), vector->wrappedIndex(0)};
}

// Restrict lookup keys to supported primitive types without floating-point
// or custom comparison semantics.
bool isSupportedConstantLookupKeyType(const TypePtr& type) {
  return type->isPrimitiveType() && !type->isReal() && !type->isDouble() &&
      !type->providesCustomComparison() && isTypeSupportedByCudf(type);
}

// Copies 'columns' of a constant collection to the GPU. toCudfTable()
// synchronizes 'stream' before it returns.
std::unique_ptr<cudf::table> copyConstantLookupColumnsToGpu(
    std::vector<VectorPtr> columns,
    memory::MemoryPool* pool,
    cuda::stream_ref stream) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (const auto& column : columns) {
    names.push_back(fmt::format("c{}", names.size()));
    types.push_back(column->type());
  }
  const auto size = columns[0]->size();
  auto rows = std::make_shared<RowVector>(
      pool,
      ROW(std::move(names), std::move(types)),
      nullptr,
      size,
      std::move(columns));
  return with_arrow::toCudfTable(rows, pool, stream, get_temp_mr());
}

// element_at(<constant map>, key) looks the key up in a hash table of the
// map's keys and gathers the matching values, so a missing or NULL key gives
// NULL. Presto's RewriteCaseToMap rule produces this form from simple CASE
// expressions. Spark's element_at raises on a missing key under ANSI mode, so
// only the Presto registration has it.
class ConstantMapElementAtFunction : public CudfFunction {
 public:
  static bool canEvaluate(const core::TypedExprPtr& expr) {
    const auto [map, row] = constantCollection<MapVector>(expr->inputs()[0]);
    const auto& mapType = expr->inputs()[0]->type()->asMap();
    // Under a unit coarser than nanoseconds, distinct TIMESTAMP keys can
    // convert to equal cuDF keys, and the hash table needs distinct keys.
    const bool keysStayDistinct = !mapType.keyType()->isTimestamp() ||
        CudfConfig::getInstance().timestampUnit ==
            cudf::type_id::TIMESTAMP_NANOSECONDS;
    return map != nullptr && map->sizeAt(row) > 0 &&
        !expr->inputs()[1]->isConstantKind() &&
        isSupportedConstantLookupKeyType(mapType.keyType()) &&
        keysStayDistinct && isTypeSupportedByCudf(mapType.valueType());
  }

  ConstantMapElementAtFunction(
      const core::TypedExprPtr& expr,
      memory::MemoryPool* pool) {
    VELOX_CHECK(
        canEvaluate(expr), "Unsupported element_at: {}", expr->toString());
    const auto [map, row] = constantCollection<MapVector>(expr->inputs()[0]);
    const auto offset = map->offsetAt(row);
    const auto size = map->sizeAt(row);
    auto stream = getDefaultStreamForCurrentThread();
    entries_ = copyConstantLookupColumnsToGpu(
        {map->mapKeys()->slice(offset, size),
         map->mapValues()->slice(offset, size)},
        pool,
        stream);
    // Built once and only read afterwards, on whatever stream eval() gets;
    // the sync below completes the build first. A map's keys are distinct.
    keyIndex_ = std::make_unique<cudf::distinct_hash_join>(
        entries_->view().select({0}),
        cudf::null_equality::UNEQUAL,
        0.5,
        stream,
        get_temp_mr());
    stream.sync();
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    // One index per row into the keys, JoinNoMatch for a missing or NULL key,
    // which the gather turns into NULL.
    const auto indices = keyIndex_->left_join(
        cudf::table_view{{asView(inputColumns[0])}}, stream, get_temp_mr());
    auto values = cudf::gather(
        entries_->view().select({1}),
        cudf::column_view{cudf::device_span<const cudf::size_type>{*indices}},
        cudf::out_of_bounds_policy::NULLIFY,
        stream,
        cudf::memory_resources{mr, get_temp_mr()});
    return std::move(values->release()[0]);
  }

 private:
  // The keys and values of the map; keyIndex_ refers to the keys.
  std::unique_ptr<cudf::table> entries_;
  std::unique_ptr<cudf::distinct_hash_join> keyIndex_;
};

// contains(<constant array>, x) looks x up among the array's elements.
class ConstantArrayContainsFunction : public CudfFunction {
 public:
  static bool canEvaluate(const core::TypedExprPtr& expr) {
    const auto [array, row] =
        constantCollection<ArrayVector>(expr->inputs()[0]);
    if (array == nullptr || array->sizeAt(row) == 0 ||
        expr->inputs()[1]->isConstantKind() ||
        !isSupportedConstantLookupKeyType(
            array->type()->asArray().elementType())) {
      return false;
    }
    // Presto gives NULL, not false, when no element matches and one is NULL.
    const auto offset = array->offsetAt(row);
    for (auto i = offset; i < offset + array->sizeAt(row); ++i) {
      if (array->elements()->isNullAt(i)) {
        return false;
      }
    }
    return true;
  }

  ConstantArrayContainsFunction(
      const core::TypedExprPtr& expr,
      memory::MemoryPool* pool) {
    VELOX_CHECK(
        canEvaluate(expr), "Unsupported contains: {}", expr->toString());
    const auto [array, row] =
        constantCollection<ArrayVector>(expr->inputs()[0]);
    elements_ = copyConstantLookupColumnsToGpu(
        {array->elements()->slice(array->offsetAt(row), array->sizeAt(row))},
        pool,
        getDefaultStreamForCurrentThread());
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    // The result takes the probe's null mask, so a NULL probe gives NULL.
    return cudf::contains(
        elements_->get_column(0).view(), asView(inputColumns[0]), stream, mr);
  }

 private:
  std::unique_ptr<cudf::table> elements_;
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

  registerCudfFunction(
      prefix + "element_at",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<ConstantMapElementAtFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .typeVariable("K")
           .typeVariable("V")
           .returnType("V")
           .argumentType("map(K,V)")
           .argumentType("K")
           .build()},
      true,
      ConstantMapElementAtFunction::canEvaluate);

  registerCudfFunction(
      prefix + "contains",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<ConstantArrayContainsFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("boolean")
           .argumentType("array(T)")
           .argumentType("T")
           .build()},
      true,
      ConstantArrayContainsFunction::canEvaluate);

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
