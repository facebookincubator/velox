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
#include "velox/experimental/cudf/expression/prestosql/VectorDistanceFunctions.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

#include <type_traits>

namespace facebook::velox::cudf_velox::prestosql {
namespace {

// Returns the elements of a constant array argument, or nullopt when the
// argument is not constant, is null, or holds a null element.
template <typename T>
std::optional<std::vector<T>> constantArray(const core::TypedExprPtr& expr) {
  constexpr auto kind = std::is_same_v<T, double> ? TypeKind::DOUBLE
                                                  : TypeKind::REAL;
  if (!expr->isConstantKind()) {
    return std::nullopt;
  }
  const auto* constant = expr->asUnchecked<core::ConstantTypedExpr>();
  std::vector<T> values;
  if (constant->hasValueVector()) {
    const auto& vector = constant->valueVector();
    if (vector->isNullAt(0)) {
      return std::nullopt;
    }
    const auto* arrays = vector->wrappedVector()->as<ArrayVector>();
    if (arrays == nullptr) {
      return std::nullopt;
    }
    const auto row = vector->wrappedIndex(0);
    const auto offset = arrays->offsetAt(row);
    const auto size = arrays->sizeAt(row);
    const auto* elements = arrays->elements()->as<SimpleVector<T>>();
    if (elements == nullptr) {
      return std::nullopt;
    }
    values.reserve(size);
    for (vector_size_t i = 0; i < size; ++i) {
      if (elements->isNullAt(offset + i)) {
        return std::nullopt;
      }
      values.push_back(elements->valueAt(offset + i));
    }
    return values;
  }
  const auto& value = constant->value();
  if (value.isNull()) {
    return std::nullopt;
  }
  for (const auto& element : value.array()) {
    if (element.isNull()) {
      return std::nullopt;
    }
    values.push_back(element.value<kind>());
  }
  return values;
}

const char* sizeMismatchMessage(VectorDistanceKind kind) {
  return kind == VectorDistanceKind::kDotProduct
      ? "Both array arguments must have identical sizes"
      : "Both arrays need to have identical size";
}

} // namespace

bool VectorDistanceFunction::canEvaluate(const core::TypedExprPtr& expr) {
  const auto& inputs = expr->inputs();
  if (inputs.size() != 2 ||
      (inputs[0]->isConstantKind() && inputs[1]->isConstantKind())) {
    return false;
  }
  const bool asDouble = expr->type()->kind() == TypeKind::DOUBLE;
  for (const auto& input : inputs) {
    if (!input->isConstantKind()) {
      continue;
    }
    const bool accepted = asDouble ? constantArray<double>(input).has_value()
                                   : constantArray<float>(input).has_value();
    if (!accepted) {
      return false;
    }
  }
  return true;
}

VectorDistanceFunction::VectorDistanceFunction(
    VectorDistanceKind kind,
    const core::TypedExprPtr& expr,
    memory::MemoryPool* /*pool*/)
    : kind_{kind},
      sizeMismatchMessage_{sizeMismatchMessage(kind)},
      doublePrecision_{expr->type()->kind() == TypeKind::DOUBLE},
      leftFloat_{
          doublePrecision_ ? std::nullopt
                           : constantArray<float>(expr->inputs()[0])},
      rightFloat_{
          doublePrecision_ ? std::nullopt
                           : constantArray<float>(expr->inputs()[1])},
      leftDouble_{
          doublePrecision_ ? constantArray<double>(expr->inputs()[0])
                           : std::nullopt},
      rightDouble_{
          doublePrecision_ ? constantArray<double>(expr->inputs()[1])
                           : std::nullopt} {
  VELOX_CHECK(
      canEvaluate(expr),
      "Vector distance expression cannot be evaluated on the GPU");
}

ColumnOrView VectorDistanceFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  // Constant arguments have no input column.
  size_t nextColumn{0};
  if (doublePrecision_) {
    const auto operand =
        [&](const std::optional<std::vector<double>>& constant) {
          if (constant.has_value()) {
            return VectorDistanceOperand<double>{.constant = *constant};
          }
          return VectorDistanceOperand<double>{
              .column = asView(inputColumns.at(nextColumn++))};
        };
    auto result = vectorDistance(
        kind_, operand(leftDouble_), operand(rightDouble_), stream, mr);
    VELOX_USER_CHECK(!result.sizeMismatch, sizeMismatchMessage_);
    return std::move(result.column);
  }
  const auto operand = [&](const std::optional<std::vector<float>>& constant) {
    if (constant.has_value()) {
      return VectorDistanceOperand<float>{.constant = *constant};
    }
    return VectorDistanceOperand<float>{
        .column = asView(inputColumns.at(nextColumn++))};
  };
  auto result = vectorDistance(
      kind_, operand(leftFloat_), operand(rightFloat_), stream, mr);
  VELOX_USER_CHECK(!result.sizeMismatch, sizeMismatchMessage_);
  return std::move(result.column);
}

void registerVectorDistanceFunctions(const std::string& prefix) {
  const std::vector<exec::FunctionSignaturePtr> signatures{
      exec::FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("array(real)")
          .argumentType("array(real)")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("array(double)")
          .argumentType("array(double)")
          .build(),
  };
  const std::pair<const char*, VectorDistanceKind> functions[]{
      {"cosine_similarity", VectorDistanceKind::kCosineSimilarity},
      {"l2_squared", VectorDistanceKind::kL2Squared},
      {"dot_product", VectorDistanceKind::kDotProduct},
  };
  for (const auto& [name, kind] : functions) {
    registerCudfFunction(
        prefix + name,
        [kind](
            const std::string&,
            const core::TypedExprPtr& expr,
            memory::MemoryPool* pool) {
          return std::make_shared<VectorDistanceFunction>(kind, expr, pool);
        },
        signatures,
        /*overwrite=*/true,
        VectorDistanceFunction::canEvaluate);
  }
}

} // namespace facebook::velox::cudf_velox::prestosql
