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
#include "velox/experimental/cudf/expression/ConstantLookupFunctions.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/copying.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/span.hpp>

namespace facebook::velox::cudf_velox {

namespace {

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

// Returns true if `type` can be a constant lookup key: a cuDF-supported
// primitive type other than REAL or DOUBLE.
bool isSupportedConstantLookupKeyType(const TypePtr& type) {
  return type->isPrimitiveType() && !type->isReal() && !type->isDouble() &&
      isTypeSupportedByCudf(type);
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

// Returns 'probe' as 'type'. Child expressions are not finalized, so a probe
// can arrive narrower than the constant, e.g. INT32 from length() for BIGINT.
ColumnOrView
castProbe(ColumnOrView& probe, cudf::data_type type, cuda::stream_ref stream) {
  const auto view = asView(probe);
  if (view.type() == type) {
    return view;
  }
  return cudf::cast(view, type, stream, get_temp_mr());
}

// element_at(<constant map>, key) looks the key up in a hash table of the
// map's keys and gathers the matching values, so a missing or NULL key gives
// NULL. Presto's RewriteCaseToMap rule produces this form from simple CASE
// expressions.
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
    auto probe =
        castProbe(inputColumns[0], entries_->get_column(0).type(), stream);
    const auto indices = keyIndex_->left_join(
        cudf::table_view{{asView(probe)}}, stream, get_temp_mr());
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
    auto probe =
        castProbe(inputColumns[0], elements_->get_column(0).type(), stream);
    return cudf::contains(
        elements_->get_column(0).view(), asView(probe), stream, mr);
  }

 private:
  // Elements of the constant array, none of them NULL.
  std::unique_ptr<cudf::table> elements_;
};

} // namespace

void registerConstantMapElementAtFunction(const std::string& name) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      name,
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
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
      [](const core::TypedExprPtr& expr, const core::QueryConfig&) {
        return ConstantMapElementAtFunction::canEvaluate(expr);
      });
}

void registerConstantArrayContainsFunction(const std::string& name) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      name,
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool,
         const core::QueryConfig&) {
        return std::make_shared<ConstantArrayContainsFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("boolean")
           .argumentType("array(T)")
           .argumentType("T")
           .build()},
      true,
      [](const core::TypedExprPtr& expr, const core::QueryConfig&) {
        return ConstantArrayContainsFunction::canEvaluate(expr);
      });
}

} // namespace facebook::velox::cudf_velox
