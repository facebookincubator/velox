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
#include <map>
#include <memory>

#include "velox/core/QueryConfig.h"
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
  MapFromEntriesFunction(bool throwOnNull, bool throwOnDuplicateKeys)
      : throwOnNull_(throwOnNull),
        throwOnDuplicateKeys_(throwOnDuplicateKeys) {}
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
      // For Presto, if the input is array(unknown), rows with non-empty
      // arrays should have errors since all entries would be null. Empty
      // arrays produce empty maps without error.
      if (throwOnNull_) {
        try {
          VELOX_USER_FAIL(kErrorMessageEntryNotNull);
        } catch (...) {
          auto exceptionPtr = std::current_exception();
          rows.applyToSelected([&](vector_size_t row) {
            if (inputArray->sizeAt(row) > 0) {
              context.setError(row, exceptionPtr);
            }
          });
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
    auto nulls = allocateNulls(rows.end(), context.pool());
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
            if (throwOnNull_) {
              // Set the sizes to 0 so that the final map vector generated is
              // valid in case we are inside a try. The map vector needs to be
              // valid because its consumed by checkDuplicateKeys before try
              // sets invalid rows to null.
              resetSize(row);
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

    // For Presto, need construct map vector based on input nulls for possible
    // outer expression like try(). For Spark, use the updated nulls unless it's
    // empty.
    if (throwOnNull_ || decodedRowVector->size() == 0) {
      nulls = inputArray->nulls();
    }

    if (!throwOnDuplicateKeys_) {
      return applyLastWin(
          *remainingRows.get(),
          inputArray,
          *decodedRowVector.get(),
          rowVector,
          nulls,
          outputType,
          context);
    }

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
      auto rowVectorNulls =
          allocateNulls(decodedRowVector->size(), context.pool());
      auto* rowVectorMutableNulls = rowVectorNulls->asMutable<uint64_t>();
      memcpy(
          indices->asMutable<vector_size_t>(),
          decodedRowVector->indices(),
          BaseVector::byteSize<vector_size_t>(decodedRowVector->size()));
      // Any null in the top row(X, Y) should be marked as null since its
      // not guranteed to be addressable at X or Y.
      for (auto i = 0; i < decodedRowVector->size(); i++) {
        if (decodedRowVector->isNullAt(i)) {
          bits::setNull(rowVectorMutableNulls, i);
        }
      }
      wrappedKeys = BaseVector::wrapInDictionary(
          rowVectorNulls,
          indices,
          decodedRowVector->size(),
          rowVector->childAt(0));
      wrappedValues = BaseVector::wrapInDictionary(
          rowVectorNulls,
          indices,
          decodedRowVector->size(),
          rowVector->childAt(1));
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

  // Builds the result for the LAST_WIN policy. Entries are kept in input order.
  // A repeated key keeps the position of its first occurrence and takes the
  // value of its last occurrence, which is how Spark's ArrayBasedMapBuilder
  // deduplicates. Rows that are null or already failed are left empty.
  VectorPtr applyLastWin(
      const SelectivityVector& rows,
      const ArrayVector* inputArray,
      const DecodedVector& decodedRowVector,
      const RowVector* rowVector,
      const BufferPtr& nulls,
      const TypePtr& outputType,
      exec::EvalCtx& context) const {
    const uint64_t* rawNulls = nulls ? nulls->as<uint64_t>() : nullptr;
    auto isNullRow = [&](vector_size_t row) {
      return rawNulls != nullptr && bits::isBitNull(rawNulls, row);
    };

    vector_size_t maxEntries = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (!isNullRow(row)) {
        maxEntries += inputArray->sizeAt(row);
      }
    });

    BufferPtr offsets = allocateOffsets(rows.end(), context.pool());
    BufferPtr sizes = allocateSizes(rows.end(), context.pool());
    BufferPtr keyIndices = allocateIndices(maxEntries, context.pool());
    BufferPtr valueIndices = allocateIndices(maxEntries, context.pool());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    auto* rawKeyIndices = keyIndices->asMutable<vector_size_t>();
    auto* rawValueIndices = valueIndices->asMutable<vector_size_t>();

    // Entry slot owned by each distinct key of the current row, keyed by the
    // index of the key's first occurrence in the key child vector. The
    // comparator uses the same null handling as BaseVector::equalValueAt, so
    // equality here matches checkDuplicateKeys.
    const auto* keys = rowVector->childAt(0).get();
    auto compareKeys = [keys](vector_size_t lhs, vector_size_t rhs) {
      return keys->compare(keys, lhs, rhs) < 0;
    };
    std::map<vector_size_t, vector_size_t, decltype(compareKeys)> slotByKey(
        compareKeys);

    vector_size_t numEntries = 0;
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      rawOffsets[row] = numEntries;
      if (isNullRow(row)) {
        return;
      }

      const auto offset = inputArray->offsetAt(row);
      const auto size = inputArray->sizeAt(row);
      slotByKey.clear();
      vector_size_t numDistinctKeys = 0;
      for (vector_size_t i = 0; i < size; ++i) {
        // Null entries were handled above: the row is null or has failed, so
        // every entry here is addressable in the children.
        const auto entryIndex = decodedRowVector.index(offset + i);
        const auto [it, inserted] =
            slotByKey.emplace(entryIndex, numDistinctKeys);
        if (!inserted) {
          rawValueIndices[numEntries + it->second] = entryIndex;
          continue;
        }
        rawKeyIndices[numEntries + numDistinctKeys] = entryIndex;
        rawValueIndices[numEntries + numDistinctKeys] = entryIndex;
        ++numDistinctKeys;
      }
      rawSizes[row] = numDistinctKeys;
      numEntries += numDistinctKeys;
    });

    return std::make_shared<MapVector>(
        context.pool(),
        outputType,
        nulls,
        rows.end(),
        offsets,
        sizes,
        BaseVector::wrapInDictionary(
            nullptr, keyIndices, numEntries, rowVector->childAt(0)),
        BaseVector::wrapInDictionary(
            nullptr, valueIndices, numEntries, rowVector->childAt(1)));
  }

  // If true, throws exception when input is NULL or contains NULL entry
  // (Presto's behavior). Otherwise, returns NULL (Spark's behavior).
  const bool throwOnNull_;

  // If true, throws exception when a key is repeated. Otherwise, the last
  // value wins (Spark's LAST_WIN policy).
  const bool throwOnDuplicateKeys_;
};
} // namespace

void registerMapFromEntriesFunction(
    const std::string& name,
    bool throwOnNull,
    bool honorDuplicateKeysConfig,
    std::string_view defaultOwner) {
  exec::registerStatefulVectorFunction(
      name,
      MapFromEntriesFunction::signatures(),
      [throwOnNull, honorDuplicateKeysConfig](
          const std::string& /*name*/,
          const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
          const core::QueryConfig& config) {
        const bool throwOnDuplicateKeys = !honorDuplicateKeysConfig ||
            config.throwExceptionOnDuplicateMapKeys();
        return std::make_shared<MapFromEntriesFunction>(
            throwOnNull, throwOnDuplicateKeys);
      },
      {},
      /*overwrite=*/true,
      defaultOwner);
}
} // namespace facebook::velox::functions
