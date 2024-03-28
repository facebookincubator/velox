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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

namespace facebook::velox::functions {

template <typename T>
struct SetWithNull {
  SetWithNull(vector_size_t initialSetSize = kInitialSetSize) {
    set.reserve(initialSetSize);
  }

  void reset() {
    set.clear();
    hasNull = false;
  }

  folly::F14FastSet<T> set;
  bool hasNull{false};
  bool hasNaN{false};
  static constexpr vector_size_t kInitialSetSize{128};
};

// Generates a set based on the elements of an ArrayVector. Note that we take
// rightSet as a parameter (instead of returning a new one) to reuse the
// allocated memory.
template <typename T, bool recordNaN, typename TVector>
void generateSet(
    const ArrayVector* arrayVector,
    const TVector* arrayElements,
    vector_size_t idx,
    SetWithNull<T>& rightSet) {
  auto size = arrayVector->sizeAt(idx);
  auto offset = arrayVector->offsetAt(idx);
  rightSet.reset();

  for (vector_size_t i = offset; i < (offset + size); ++i) {
    if (arrayElements->isNullAt(i)) {
      rightSet.hasNull = true;
    } else {
      // Function can be called with either FlatVector or DecodedVector, but
      // their APIs are slightly different.
      T value;
      if constexpr (std::is_same_v<TVector, DecodedVector>) {
        value = arrayElements->template valueAt<T>(i);
      } else {
        value = arrayElements->valueAt(i);
      }
      if constexpr (
          recordNaN &&
          (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        if (std::isnan(value)) {
          rightSet.hasNaN = true;
        }
      }
      rightSet.set.insert(value);
    }
  }
}

DecodedVector* decodeArrayElements(
    exec::LocalDecodedVector& arrayDecoder,
    exec::LocalDecodedVector& elementsDecoder,
    const SelectivityVector& rows);

/// See documentation at https://prestodb.io/docs/current/functions/array.html
/// @tparam isIntersect: if true, the function is array_intersect, otherwise
/// array_except.
/// @tparam equalNaN: if true, NaN is considered equal to NaN, otherwise it is
/// not.
/// @tparam T: the type of the array elements.
template <bool isIntersect, bool equalNaN, typename T>
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
  /// If the rhs values passed to either array_intersect() or array_except()
  /// are constant (array literals) we create a set before instantiating the
  /// object and pass as a constructor parameter (constantSet).

  ArrayIntersectExceptFunction() = default;

  explicit ArrayIntersectExceptFunction(SetWithNull<T> constantSet)
      : constantSet_(std::move(constantSet)) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    memory::MemoryPool* pool = context.pool();
    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray = decodedLeftArray->base()->as<ArrayVector>();

    // Decode and acquire array elements vector.
    exec::LocalDecodedVector leftElementsDecoder(context);
    auto decodedLeftElements =
        decodeArrayElements(leftHolder, leftElementsDecoder, rows);

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
      bool hasNaN = false;

      outputSet.reset();
      rawNewOffsets[row] = indicesCursor;

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
          constexpr bool isFloating =
              std::is_same_v<T, float> || std::is_same_v<T, double>;
          if constexpr (isIntersect) {
            bool addNaN = false;
            if constexpr (equalNaN && isFloating) {
              // For a NaN value not added to the output row yet, insert in
              // array_intersect if it was found on the rhs (and not found in
              // the case of array_except).
              addNaN = rightSet.hasNaN && std::isnan(val);
            }
            addValue = rightSet.set.count(val) > 0 || addNaN;
          } else {
            bool addNaN = true;
            if constexpr (equalNaN && isFloating) {
              addNaN = !rightSet.hasNaN || !std::isnan(val);
            }
            addValue = rightSet.set.count(val) == 0 && addNaN;
          }
          if (addValue) {
            auto it = outputSet.set.insert(val);
            if (it.second) {
              rawNewIndices[indicesCursor++] = i;
            }
          }
        }
      }
      rawNewLengths[row] = indicesCursor - rawNewOffsets[row];
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
      auto decodedRightElements =
          decodeArrayElements(rightHolder, rightElementsHolder, rows);
      SetWithNull<T> rightSet;
      auto rightArrayVector = rightHolder.get()->base()->as<ArrayVector>();
      rows.applyToSelected([&](vector_size_t row) {
        auto idx = rightHolder.get()->index(row);
        generateSet<T, equalNaN>(
            rightArrayVector, decodedRightElements, idx, rightSet);
        processRow(row, rightSet, outputSet);
      });
    }

    auto newElements = BaseVector::wrapInDictionary(
        newElementNulls, newIndices, indicesCursor, baseLeftArray->elements());
    auto resultArray = std::make_shared<ArrayVector>(
        pool,
        outputType,
        nullptr,
        rowCount,
        newOffsets,
        newLengths,
        newElements);
    context.moveOrCopyResult(resultArray, rows, result);
  }

  // If one of the arrays is constant, this member will store a pointer to the
  // set generated from its elements, which is calculated only once, before
  // instantiating this object.
  std::optional<SetWithNull<T>> constantSet_;
}; // class ArrayIntersectExcept

template <typename T, bool recordNaN>
SetWithNull<T> validateConstantVectorAndGenerateSet(
    const BaseVector* baseVector) {
  auto constantVector = baseVector->as<ConstantVector<velox::ComplexType>>();
  auto constantArray = constantVector->as<ConstantVector<velox::ComplexType>>();
  VELOX_CHECK_NOT_NULL(constantArray, "wrong constant type found");
  VELOX_CHECK_NOT_NULL(constantVector, "wrong constant type found");
  auto arrayVecPtr = constantVector->valueVector()->as<ArrayVector>();
  VELOX_CHECK_NOT_NULL(arrayVecPtr, "wrong array literal type");

  auto idx = constantArray->index();
  auto elementBegin = arrayVecPtr->offsetAt(idx);
  auto elementEnd = elementBegin + arrayVecPtr->sizeAt(idx);

  SelectivityVector rows{elementEnd, false};
  rows.setValidRange(elementBegin, elementEnd, true);
  rows.updateBounds();

  DecodedVector decodedElements{*arrayVecPtr->elements(), rows};

  SetWithNull<T> constantSet;
  generateSet<T, recordNaN>(arrayVecPtr, &decodedElements, idx, constantSet);
  return constantSet;
}

template <bool isIntersect, TypeKind kind>
std::shared_ptr<exec::VectorFunction> createTypedArraysIntersectExcept(
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    bool equalNaN) {
  using T = typename TypeTraits<kind>::NativeType;

  VELOX_CHECK_EQ(inputArgs.size(), 2);
  BaseVector* rhs = inputArgs[1].constantValue.get();

  // We don't optimize the case where lhs is a constant expression for
  // array_intersect() because that would make this function non-deterministic.
  // For example, a constant lhs would mean the constantSet is created based on
  // lhs; the same data encoded as a regular column could result in the set
  // being created based on rhs. Running this function with different sets could
  // results in arrays in different orders.
  //
  // If rhs is a constant value:
  if (rhs != nullptr) {
    if (equalNaN) {
      return std::make_shared<
          ArrayIntersectExceptFunction<isIntersect, true, T>>(
          validateConstantVectorAndGenerateSet<T, true>(rhs));
    } else {
      return std::make_shared<
          ArrayIntersectExceptFunction<isIntersect, false, T>>(
          validateConstantVectorAndGenerateSet<T, false>(rhs));
    }
  } else {
    if (equalNaN) {
      return std::make_shared<
          ArrayIntersectExceptFunction<isIntersect, true, T>>();
    } else {
      return std::make_shared<
          ArrayIntersectExceptFunction<isIntersect, false, T>>();
    }
  }
}

void validateMatchingArrayTypes(
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const std::string& name,
    vector_size_t expectedArgCount);

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures(
    const std::string& returnTypeTemplate);

} // namespace facebook::velox::functions
