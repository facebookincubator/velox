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
    exec::LocalDecodedVector elementsDecoder(context);
    auto decodedLeftElements =
        getDecodedElementsFromArrayVector(decoder, elementsDecoder, rows);

    FlatVector<bool>* resBoolVector = (*result)->asFlatVector<bool>();
    auto processRow = [&](const vector_size_t rowId,
                          const SetWithNull<T>& rightSet) {
      auto decodedLeftArray = decoder.get();
      auto baseLeftArray = decodedLeftArray->base()->as<ArrayVector>();
      auto idx = decodedLeftArray->index(rowId);
      auto offset = baseLeftArray->offsetAt(idx);
      auto size = baseLeftArray->sizeAt(idx);

      for (auto i = offset; i < (offset + size); ++i) {
        // For each element in the current row search for it in the rightSet.
        if (decodedLeftElements->isNullAt(i)) {
          continue;
        }
        if (rightSet.set.count(decodedLeftElements->valueAt<T>(i)) > 0) {
          // Found and overlapping element. Add to result set.
          resBoolVector->set(rowId, true);
          return;
        }
      }
      // If none of them is found in the rightSet, set false for current row
      // indicating there are no overlapping elements with rightSet.
      resBoolVector->set(rowId, false);
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
      auto decodedRightElements = getDecodedElementsFromArrayVector(
          rightDecoder, rightElementsDecoder, rows);
      SetWithNull<T> rightSet;

      rows.applyToSelected([&](vector_size_t row) {
        auto idx = rightDecoder.get()->index(row);
        auto baseRightArray = rightDecoder.get()->base()->as<ArrayVector>();
        generateSet<T>(baseRightArray, decodedRightElements, idx, rightSet);
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

template <TypeKind kind>
const std::shared_ptr<exec::VectorFunction> createTyped(
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  BaseVector* left = inputArgs[0].constantValue.get();
  BaseVector* right = inputArgs[1].constantValue.get();
  using T = typename TypeTraits<kind>::NativeType;
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