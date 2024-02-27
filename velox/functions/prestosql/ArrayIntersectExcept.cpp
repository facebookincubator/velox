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
#include "velox/functions/lib/ArrayIntersectExceptFunction.h"

namespace facebook::velox::functions {
namespace {

template <typename T>
class ArraysOverlapFunction : public exec::VectorFunction {
 public:
  ArraysOverlapFunction() = default;

  ArraysOverlapFunction(SetWithNull<T> constantSet, bool isLeftConstant)
      : constantSet_(std::move(constantSet)), isLeftConstant_(isLeftConstant) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();
    if (constantSet_.has_value() && isLeftConstant_) {
      std::swap(left, right);
    }
    exec::LocalDecodedVector arrayDecoder(context, *left, rows);
    exec::LocalDecodedVector elementsDecoder(context);
    auto decodedLeftElements =
        decodeArrayElements(arrayDecoder, elementsDecoder, rows);
    auto decodedLeftArray = arrayDecoder.get();
    auto baseLeftArray = decodedLeftArray->base()->as<ArrayVector>();
    context.ensureWritable(rows, BOOLEAN(), result);
    auto resultBoolVector = result->template asFlatVector<bool>();
    auto processRow = [&](auto row, const SetWithNull<T>& rightSet) {
      auto idx = decodedLeftArray->index(row);
      auto offset = baseLeftArray->offsetAt(idx);
      auto size = baseLeftArray->sizeAt(idx);
      bool hasNull = rightSet.hasNull;
      for (auto i = offset; i < (offset + size); ++i) {
        // For each element in the current row search for it in the rightSet.
        if (decodedLeftElements->isNullAt(i)) {
          // Arrays overlap function skips null values.
          hasNull = true;
          continue;
        }
        if (rightSet.set.count(decodedLeftElements->valueAt<T>(i)) > 0) {
          // Found an overlapping element. Add to result set.
          resultBoolVector->set(row, true);
          return;
        }
      }
      if (hasNull) {
        // If encountered a NULL, insert NULL in the result.
        resultBoolVector->setNull(row, true);
      } else {
        // If there is no overlap and no nulls, then insert false.
        resultBoolVector->set(row, false);
      }
    };

    if (constantSet_.has_value()) {
      rows.applyToSelected(
          [&](vector_size_t row) { processRow(row, *constantSet_); });
    }
    // General case when no arrays are constant and both sets need to be
    // computed for each row.
    else {
      exec::LocalDecodedVector rightDecoder(context, *right, rows);
      exec::LocalDecodedVector rightElementsDecoder(context);
      auto decodedRightElements =
          decodeArrayElements(rightDecoder, rightElementsDecoder, rows);
      SetWithNull<T> rightSet;
      auto baseRightArray = rightDecoder.get()->base()->as<ArrayVector>();
      rows.applyToSelected([&](vector_size_t row) {
        auto idx = rightDecoder.get()->index(row);
        generateSet<T, false>(
            baseRightArray, decodedRightElements, idx, rightSet);
        processRow(row, rightSet);
      });
    }
  }

 private:
  // If one of the arrays is constant, this member will store a pointer to the
  // set generated from its elements, which is calculated only once, before
  // instantiating this object.
  std::optional<SetWithNull<T>> constantSet_;

  // If there's a `constantSet`, whether it refers to left or right-hand side.
  const bool isLeftConstant_{false};
}; // class ArraysOverlapFunction

std::shared_ptr<exec::VectorFunction> createArrayIntersect(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  validateMatchingArrayTypes(inputArgs, name, 2);
  auto elementType = inputArgs.front().type->childAt(0);

  return VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      createTypedArraysIntersectExcept,
      /* isIntersect */ true,
      elementType->kind(),
      inputArgs,
      false);
}

std::shared_ptr<exec::VectorFunction> createArrayExcept(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  validateMatchingArrayTypes(inputArgs, name, 2);
  auto elementType = inputArgs.front().type->childAt(0);

  return VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      createTypedArraysIntersectExcept,
      /* isIntersect */ false,
      elementType->kind(),
      inputArgs,
      false);
}

template <TypeKind kind>
const std::shared_ptr<exec::VectorFunction> createTypedArraysOverlap(
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  auto left = inputArgs[0].constantValue.get();
  auto right = inputArgs[1].constantValue.get();
  using T = typename TypeTraits<kind>::NativeType;
  if (left == nullptr && right == nullptr) {
    return std::make_shared<ArraysOverlapFunction<T>>();
  }
  auto isLeftConstant = (left != nullptr);
  auto baseVector = isLeftConstant ? left : right;
  auto constantSet = validateConstantVectorAndGenerateSet<T, false>(baseVector);
  return std::make_shared<ArraysOverlapFunction<T>>(
      std::move(constantSet), isLeftConstant);
}

std::shared_ptr<exec::VectorFunction> createArraysOverlapFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  validateMatchingArrayTypes(inputArgs, name, 2);
  auto elementType = inputArgs.front().type->childAt(0);

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      createTypedArraysOverlap, elementType->kind(), inputArgs);
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_arrays_overlap,
    signatures("boolean"),
    createArraysOverlapFunction);

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_array_intersect,
    signatures("array({})"),
    createArrayIntersect);

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_array_except,
    signatures("array({})"),
    createArrayExcept);
} // namespace facebook::velox::functions
