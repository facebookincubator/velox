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
#include "velox/functions/lib/LambdaFunctionUtil.h"

namespace facebook::velox::functions {
namespace {

template <typename T>
class ArraysOverlapFunction : public exec::VectorFunction {
 public:
  ArraysOverlapFunction() {}

  explicit ArraysOverlapFunction(
      SetWithNull<T> constantSet,
      bool isLeftConstant)
      : constantSet_(std::move(constantSet)), isLeftConstant_(isLeftConstant) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    // if one of them is a ConstantVector.
    // Iterate through the other ArrayVector->elementsFlatVector
    // and do constantSet_.find()
    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();
    if (constantSet_.has_value() && isLeftConstant_) {
      std::swap(left, right);
    }

    exec::LocalDecodedVector decoder(context, *left, rows);
    auto decodedLeftArray = decoder.get();
    auto baseLeftArray = decoder->base()->as<ArrayVector>();

    // Decode and acquire array elements vector.
    //    auto leftElementsVector = baseLeftArray->elements();
    //    auto leftElementsRows = toElementRows(
    //        leftElementsVector->size(),
    //        rows,
    //        baseLeftArray,
    //        decodedLeftArray->indices());
    //    exec::LocalDecodedVector leftElementsHolder(
    //        context, *leftElementsVector, leftElementsRows);
    //    auto decodedLeftElements = leftElementsHolder.get();
    auto decodedLeftElements =
        getDecodedElementsFromArrayVector(context, *left, rows);
    auto leftElementsCount =
        countElements<ArrayVector>(rows, *decodedLeftArray);
    vector_size_t rowCount = left->size();

    if (constantSet_.has_value()) {
      rows.applyToSelected(
          [&](vector_size_t row) { processRow(row, *constantSet_, result); });
    }
    // General case when no arrays are constant and both sets need to be
    // computed for each row.
    else {
      auto decodedRightElements =
          getDecodedElementsFromArrayVector(context, *right, rows);
      SetWithNull<T> rightSet;

      rows.applyToSelected([&](vector_size_t row) {
        auto idx = decodedRightArray->index(row);
        generateSet<T>(baseRightArray, decodedRightElements, idx, rightSet);
        processRow(row, rightSet, result);
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

template <TypeKind kind>
const std::shared_ptr<exec::VectorFunction> createTyped(
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  BaseVector* left = inputArgs[0].constantValue.get();
  BaseVector* right = inputArgs[1].constantValue.get();
  using T = typename TypeTraits<kind>::NativeType;
  // Constant optimization is not supported for constant lhs for array_except
  const bool isLeftConstant = (left != nullptr);
  if (left == nullptr && right == nullptr) {
    return std::make_shared<ArraysOverlapFunction<T>>();
  }
  BaseVector* baseVector = (left != nullptr) ? left : right;
  auto constantVector = baseVector->as<ConstantVector<velox::ComplexType>>();
  VELOX_CHECK_NOT_NULL(constantVector, "wrong constant type found");
  auto arrayVecPtr = constantVector->valueVector()->as<ArrayVector>();
  VELOX_CHECK_NOT_NULL(arrayVecPtr, "wrong array literal type");
  auto elementsAsFlatVector = arrayVecPtr->elements()->as<FlatVector<T>>();
  VELOX_CHECK_NOT_NULL(
      elementsAsFlatVector, "constant value must be encoded as flat");

  auto idx = constantVector->index();
  SetWithNull<T> constantSet;
  generateSet<T>(arrayVecPtr, elementsAsFlatVector, idx, constantSet);
  return std::make_shared<ArraysOverlapFunction<T>>(
      std::move(constantSet), isLeftConstant);
}

std::shared_ptr<exec::VectorFunction> createArraysOverlapFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  validateType(inputArgs, name, 2);
  auto elementType = inputArgs.front().type->childAt(0);

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      createTyped, elementType->kind(), inputArgs);
}

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  // array(T), array(T) -> array(T)
  return {exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType("array(T)")
              .argumentType("array(T)")
              .argumentType("array(T)")
              .build()};
}

} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_arrays_overlap,
    signatures(),
    createArraysOverlapFunction);
} // namespace facebook::velox::functions