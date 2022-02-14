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

// See documentation at https://prestodb.io/docs/current/functions/array.html
template <bool isIntersect, typename T>
class ArrayIntersectExceptFunction : public exec::VectorFunction {
 public:
  /// This class is used for both array_intersect and array_except functions
  /// (behavior controlled at compile time by the isIntersect template
  /// variable). Both these functions take two ArrayVectors as inputs (left and
  /// right) and leverage two sets to calculate the intersection (or except):
  ///
  /// - rightSet: a set that contains all (distinct) elements from the
  ///   right-hand side array.
  /// - outputSet: a set that contains the elements that were already added to
  ///   the output (to prevent duplicates).
  ///
  /// Along with each set, we maintain a `hasNull` flag that indicates whether
  /// null is present in the arrays, to prevent the use of optional types or
  /// special values.
  ///
  /// Zero element copy:
  ///
  /// In order to prevent copies of array elements, the function reuses the
  /// internal elements() vector from the left-hand side ArrayVector.
  ///
  /// First a new vector is created containing the indices of the elements
  /// which will be present in the output, and wrapped into a DictionaryVector.
  /// Next the `lengths` and `offsets` vectors that control where output arrays
  /// start and end are wrapped into the output ArrayVector.
  ///
  /// Constant optimization:
  ///
  /// If any of the values passed to array_intersect() or rhs for array_except()
  /// are constant (array literals) we create a set before instantiating the
  /// object and pass as a constructor parameter (constantSet).

  ArrayIntersectExceptFunction() {}

  explicit ArrayIntersectExceptFunction(
      SetWithNull<T> constantSet,
      bool isLeftConstant)
      : constantSet_(std::move(constantSet)), isLeftConstant_(isLeftConstant) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    memory::MemoryPool* pool = context->pool();
    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    // For array_intersect, if there's a constant input, then require it is on
    // the right side. For array_except, the constant optimization only applies
    // if the constant is on the rhs, so this swap is not applicable.
    if constexpr (isIntersect) {
      if (constantSet_.has_value() && isLeftConstant_) {
        std::swap(left, right);
      }
    }

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray = decodedLeftArray->base()->as<ArrayVector>();

    // Decode and acquire array elements vector.
    exec::LocalDecodedVector leftElementsDecoder(context);
    auto decodedLeftElements = getDecodedElementsFromArrayVector(
        leftHolder, leftElementsDecoder, rows);

    auto leftElementsCount =
        countElements<ArrayVector>(rows, *decodedLeftArray);
    vector_size_t rowCount = left->size();

    // Allocate new vectors for indices, nulls, length and offsets.
    BufferPtr newIndices = allocateIndices(leftElementsCount, pool);
    BufferPtr newElementNulls =
        AlignedBuffer::allocate<bool>(leftElementsCount, pool, bits::kNotNull);
    BufferPtr newLengths = allocateSizes(rowCount, pool);
    BufferPtr newOffsets = allocateOffsets(rowCount, pool);

    // Pointers and cursors to the raw data.
    auto rawNewIndices = newIndices->asMutable<vector_size_t>();
    auto rawNewElementNulls = newElementNulls->asMutable<uint64_t>();
    auto rawNewLengths = newLengths->asMutable<vector_size_t>();
    auto rawNewOffsets = newOffsets->asMutable<vector_size_t>();

    vector_size_t indicesCursor = 0;

    // Lambda that process each row. This is detached from the code so we can
    // apply it differently based on whether the right-hand side set is constant
    // or not.
    auto processRow = [&](vector_size_t row,
                          const SetWithNull<T>& rightSet,
                          SetWithNull<T>& outputSet) {
      auto idx = decodedLeftArray->index(row);
      auto size = baseLeftArray->sizeAt(idx);
      auto offset = baseLeftArray->offsetAt(idx);

      outputSet.reset();
      *rawNewOffsets = indicesCursor;

      // Scans the array elements on the left-hand side.
      for (vector_size_t i = offset; i < (offset + size); ++i) {
        if (decodedLeftElements->isNullAt(i)) {
          // For a NULL value not added to the output row yet, insert in
          // array_intersect if it was found on the rhs (and not found in the
          // case of array_except).
          if (!outputSet.hasNull) {
            bool setNull = false;
            if constexpr (isIntersect) {
              setNull = rightSet.hasNull;
            } else {
              setNull = !rightSet.hasNull;
            }
            if (setNull) {
              bits::setNull(rawNewElementNulls, indicesCursor++, true);
              outputSet.hasNull = true;
            }
          }
        } else {
          auto val = decodedLeftElements->valueAt<T>(i);
          // For array_intersect, add the element if it is found (not found
          // for array_except) in the right-hand side, and wasn't added already
          // (check outputSet).
          bool addValue = false;
          if constexpr (isIntersect) {
            addValue = rightSet.set.count(val) > 0;
          } else {
            addValue = rightSet.set.count(val) == 0;
          }
          if (addValue) {
            auto it = outputSet.set.insert(val);
            if (it.second) {
              rawNewIndices[indicesCursor++] = i;
            }
          }
        }
      }
      *rawNewLengths = indicesCursor - *rawNewOffsets;
      ++rawNewLengths;
      ++rawNewOffsets;
    };

    SetWithNull<T> outputSet;

    // Optimized case when the right-hand side array is constant.
    if (constantSet_.has_value()) {
      rows.applyToSelected([&](vector_size_t row) {
        processRow(row, *constantSet_, outputSet);
      });
    }
    // General case when no arrays are constant and both sets need to be
    // computed for each row.
    else {
      exec::LocalDecodedVector rightHolder(context, *right, rows);
      // Decode and acquire array elements vector.
      exec::LocalDecodedVector rightElementsHolder(context);
      auto decodedRightElements = getDecodedElementsFromArrayVector(
          rightHolder, rightElementsHolder, rows);
      SetWithNull<T> rightSet;

      rows.applyToSelected([&](vector_size_t row) {
        auto idx = rightHolder.get()->index(row);
        generateSet<T>(
            rightHolder.get()->base()->as<ArrayVector>(),
            decodedRightElements,
            idx,
            rightSet);
        processRow(row, rightSet, outputSet);
      });
    }

    auto newElements = BaseVector::wrapInDictionary(
        newElementNulls, newIndices, indicesCursor, baseLeftArray->elements());
    auto resultArray = std::make_shared<ArrayVector>(
        pool,
        ARRAY(CppToType<T>::create()),
        BufferPtr(nullptr),
        rowCount,
        newOffsets,
        newLengths,
        newElements,
        0);
    context->moveOrCopyResult(resultArray, rows, result);
  }

  // If one of the arrays is constant, this member will store a pointer to the
  // set generated from its elements, which is calculated only once, before
  // instantiating this object.
  std::optional<SetWithNull<T>> constantSet_;

  // If there's a `constantSet`, whether it refers to left or right-hand side.
  const bool isLeftConstant_{false};
};


template <bool isIntersect, TypeKind kind>
std::shared_ptr<exec::VectorFunction> createTyped(
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  BaseVector* left = inputArgs[0].constantValue.get();
  BaseVector* right = inputArgs[1].constantValue.get();
  using T = typename TypeTraits<kind>::NativeType;
  // No constant values.
  if ((left == nullptr) && (right == nullptr)) {
    return std::make_shared<ArrayIntersectExceptFunction<isIntersect, T>>();
  }

  // Constant optimization is not supported for constant lhs for array_except
  const bool isLeftConstant = (left != nullptr);
  if (isLeftConstant) {
    if constexpr (!isIntersect) {
      return std::make_shared<ArrayIntersectExceptFunction<isIntersect, T>>();
    }
  }

  // From now on either left or right is constant; generate a set based on its
  // elements.
  BaseVector* constantVector = isLeftConstant ? left : right;

  auto constantArray = constantVector->as<ConstantVector<velox::ComplexType>>();
  VELOX_CHECK_NOT_NULL(constantArray, "wrong constant type found");

  auto constantArrayVector = constantArray->valueVector()->as<ArrayVector>();
  VELOX_CHECK_NOT_NULL(constantArrayVector, "wrong array literal type");

  auto flatVectorElements =
      constantArrayVector->elements()->as<FlatVector<T>>();
  VELOX_CHECK_NOT_NULL(
      flatVectorElements, "constant value must be encoded as flat");

  auto idx = constantArray->index();
  SetWithNull<T> constantSet;
  generateSet<T>(constantArrayVector, flatVectorElements, idx, constantSet);
  return std::make_shared<ArrayIntersectExceptFunction<isIntersect, T>>(
      std::move(constantSet), isLeftConstant);
}

std::shared_ptr<exec::VectorFunction> createArrayIntersect(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  validateType(inputArgs, name, 2);
  auto elementType = inputArgs.front().type->childAt(0);

  return VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      createTyped,
      /* isIntersect */ true,
      elementType->kind(),
      inputArgs);
}

std::shared_ptr<exec::VectorFunction> createArrayExcept(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  validateType(inputArgs, name, 2);
  auto elementType = inputArgs.front().type->childAt(0);

  return VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      createTyped, /* isIntersect */ false, elementType->kind(), inputArgs);
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
    udf_array_intersect,
    signatures(),
    createArrayIntersect);

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_array_except,
    signatures(),
    createArrayExcept);

} // namespace facebook::velox::functions
