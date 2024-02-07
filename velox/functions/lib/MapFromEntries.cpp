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
#include <memory>

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/CheckDuplicateKeys.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::functions {
namespace {
static const char* kNullKeyErrorMessage = "map key cannot be null";
static const char* kErrorMessageEntryNotNull = "map entry cannot be null";

class MapFromEntriesFunction : public exec::VectorFunction {
 public:
  MapFromEntriesFunction(bool throwOnNull) : throwOnNull_(throwOnNull) {}
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);
    auto& arg = args[0];
    VectorPtr localResult;
    // Input can be constant or flat.
    if (arg->isConstantEncoding()) {
      auto* constantArray = arg->as<ConstantVector<ComplexType>>();
      const auto& flatArray = constantArray->valueVector();
      const auto flatIndex = constantArray->index();

      exec::LocalSelectivityVector singleRow(context, flatIndex + 1);
      singleRow->clearAll();
      singleRow->setValid(flatIndex, true);
      singleRow->updateBounds();

      localResult = applyFlat(
          *singleRow.get(), flatArray->as<ArrayVector>(), outputType, context);
      localResult =
          BaseVector::wrapInConstant(rows.size(), flatIndex, localResult);
    } else {
      localResult =
          applyFlat(rows, arg->as<ArrayVector>(), outputType, context);
    }

    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        // unknown -> map(unknown, unknown)
        exec::FunctionSignatureBuilder()
            .returnType("map(unknown, unknown)")
            .argumentType("unknown")
            .build(),
        // array(unknown) -> map(unknown, unknown)
        exec::FunctionSignatureBuilder()
            .returnType("map(unknown, unknown)")
            .argumentType("array(unknown)")
            .build(),
        // array(row(K,V)) -> map(K,V)
        exec::FunctionSignatureBuilder()
            .typeVariable("K")
            .typeVariable("V")
            .returnType("map(K,V)")
            .argumentType("array(row(K,V))")
            .build()};
  }

 private:
  VectorPtr applyFlat(
      const SelectivityVector& rows,
      const ArrayVector* inputArray,
      const TypePtr& outputType,
      exec::EvalCtx& context) const {
    auto& inputValueVector = inputArray->elements();
    exec::LocalDecodedVector decodedRowVector(context);
    decodedRowVector.get()->decode(*inputValueVector);
    if (inputValueVector->typeKind() == TypeKind::UNKNOWN) {
      // For Presto, if the input is array(unknown), all rows should have
      // errors.
      if (throwOnNull_) {
        try {
          VELOX_USER_FAIL(kErrorMessageEntryNotNull);
        } catch (...) {
          context.setErrors(rows, std::current_exception());
        }
      }

      auto sizes = allocateSizes(rows.end(), context.pool());
      auto offsets = allocateSizes(rows.end(), context.pool());

      // Output in this case is map(unknown, unknown), but all elements are
      // nulls, all offsets and sizes are 0.
      return std::make_shared<MapVector>(
          context.pool(),
          outputType,
          inputArray->nulls(),
          rows.end(),
          sizes,
          offsets,
          BaseVector::create(UNKNOWN(), 0, context.pool()),
          BaseVector::create(UNKNOWN(), 0, context.pool()));
    }

    exec::LocalSelectivityVector remainingRows(context, rows);
    auto rowVector = decodedRowVector->base()->as<RowVector>();
    auto keyVector = rowVector->childAt(0);

    BufferPtr sizes = allocateSizes(rows.end(), context.pool());
    vector_size_t* mutableSizes = sizes->asMutable<vector_size_t>();
    rows.applyToSelected([&](vector_size_t row) {
      mutableSizes[row] = inputArray->rawSizes()[row];
    });

    auto resetSize = [&](vector_size_t row) { mutableSizes[row] = 0; };
    auto nulls = allocateNulls(decodedRowVector->size(), context.pool());
    auto* mutableNulls = nulls->asMutable<uint64_t>();

    if (decodedRowVector->mayHaveNulls() || keyVector->mayHaveNulls() ||
        keyVector->mayHaveNullsRecursive()) {
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        const auto size = inputArray->sizeAt(row);
        const auto offset = inputArray->offsetAt(row);

        for (auto i = 0; i < size; ++i) {
          // Check nulls in the top level row vector.
          const bool isMapEntryNull = decodedRowVector->isNullAt(offset + i);
          if (isMapEntryNull) {
            // The map vector needs to be valid because its consumed by
            // checkDuplicateKeys before try sets invalid rows to null.
            resetSize(row);
            if (throwOnNull_) {
              VELOX_USER_FAIL(kErrorMessageEntryNotNull);
            }
            bits::setNull(mutableNulls, row);
            break;
          }

          // Check null keys.
          auto keyIndex = decodedRowVector->index(offset + i);
          if (keyVector->isNullAt(keyIndex)) {
            resetSize(row);
            VELOX_USER_FAIL(kNullKeyErrorMessage);
          }
        }
      });
    }

    context.deselectErrors(*remainingRows.get());

    VectorPtr wrappedKeys;
    VectorPtr wrappedValues;
    if (decodedRowVector->isIdentityMapping()) {
      wrappedKeys = rowVector->childAt(0);
      wrappedValues = rowVector->childAt(1);
    } else if (decodedRowVector->isConstantMapping()) {
      if (decodedRowVector->isNullAt(0)) {
        // If top level row is null, child might not be addressable at index 0
        // so we do not try to read it.
        wrappedKeys = BaseVector::createNullConstant(
            rowVector->childAt(0)->type(),
            decodedRowVector->size(),
            context.pool());
        wrappedValues = BaseVector::createNullConstant(
            rowVector->childAt(1)->type(),
            decodedRowVector->size(),
            context.pool());
      } else {
        wrappedKeys = BaseVector::wrapInConstant(
            decodedRowVector->size(),
            decodedRowVector->index(0),
            rowVector->childAt(0));
        wrappedValues = BaseVector::wrapInConstant(
            decodedRowVector->size(),
            decodedRowVector->index(0),
            rowVector->childAt(1));
      }
    } else {
      // Dictionary.
      auto indices = allocateIndices(decodedRowVector->size(), context.pool());
      memcpy(
          indices->asMutable<vector_size_t>(),
          decodedRowVector->indices(),
          BaseVector::byteSize<vector_size_t>(decodedRowVector->size()));
      // Any null in the top row(X, Y) should be marked as null since its
      // not guranteed to be addressable at X or Y.
      for (auto i = 0; i < decodedRowVector->size(); i++) {
        if (decodedRowVector->isNullAt(i)) {
          bits::setNull(mutableNulls, i);
        }
      }
      wrappedKeys = BaseVector::wrapInDictionary(
          nulls, indices, decodedRowVector->size(), rowVector->childAt(0));
      wrappedValues = BaseVector::wrapInDictionary(
          nulls, indices, decodedRowVector->size(), rowVector->childAt(1));
    }

    // For Presto, need construct map vector based on input nulls for possible
    // outer expression like try(). For Spark, use the updated nulls unless it's
    // empty.
    if (throwOnNull_ || decodedRowVector->size() == 0) {
      nulls = inputArray->nulls();
    }
    auto mapVector = std::make_shared<MapVector>(
        context.pool(),
        outputType,
        nulls,
        rows.end(),
        inputArray->offsets(),
        sizes,
        wrappedKeys,
        wrappedValues);

    checkDuplicateKeys(mapVector, *remainingRows, context);
    return mapVector;
  }

  // If true, throws exception when input is NULL or contains NULL entry
  // (Presto's behavior). Otherwise, returns NULL (Spark's behavior).
  const bool throwOnNull_;
};
} // namespace

void registerMapFromEntriesFunction(const std::string& name, bool throwOnNull) {
  exec::registerVectorFunction(
      name,
      MapFromEntriesFunction::signatures(),
      std::make_unique<MapFromEntriesFunction>(throwOnNull));
}
} // namespace facebook::velox::functions
