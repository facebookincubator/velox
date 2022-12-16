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
#include <iostream>

#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/expression/VectorReaders.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

namespace facebook::velox::functions {
namespace {
// See documentation at https://prestodb.io/docs/current/functions/map.html
class MapFromEntriesFunction : public exec::VectorFunction {
 public:
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

      SelectivityVector singleRow(flatIndex + 1, false);
      singleRow.setValid(flatIndex, true);
      singleRow.updateBounds();

      localResult = applyFlat(singleRow, flatArray.get(), outputType, context);
      localResult =
          BaseVector::wrapInConstant(rows.size(), flatIndex, localResult);
    } else {
      exec::DecodedArgs decodedArgs(rows, args, context);
      auto decodedArg = decodedArgs.at(0);
      VELOX_CHECK_EQ(decodedArg->base()->typeKind(), TypeKind::ARRAY);
      auto inputArray = decodedArg->base();
      localResult = applyFlat(rows, inputArray, outputType, context);
    }

    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {// array(unknown) -> map(unknown, unknown)
            exec::FunctionSignatureBuilder()
                .returnType("map(unknown, unknown)")
                .argumentType("array(unknown)")
                .build(),
            // array(row(K,V)) -> map(K,V)
            exec::FunctionSignatureBuilder()
                .knownTypeVariable("K")
                .typeVariable("V")
                .returnType("map(K,V)")
                .argumentType("array(row(K,V))")
                .build()};
  }

 private:
  VectorPtr applyFlat(
      const SelectivityVector& rows,
      const BaseVector* inputVector,
      const TypePtr& outputType,
      exec::EvalCtx& context) const {
    auto inputArray = inputVector->as<ArrayVector>();
    VELOX_CHECK(inputArray);

    VELOX_CHECK_EQ(inputArray->elements()->typeKind(), TypeKind::ROW);
    auto rowVector = inputArray->elements()->as<RowVector>();
    VELOX_CHECK(rowVector);
    auto rowKeyVector = rowVector->childAt(0);

    // validate all map entries and map keys are not null
    static const char* kNullEntry = "map entry at {} cannot be null";
    static const char* kNullKey = "map key at {} cannot be null";
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      VELOX_USER_CHECK(!rowVector->isNullAt(row), fmt::format(kNullEntry, row));
      VELOX_USER_CHECK(
          !rowKeyVector->isNullAt(row), fmt::format(kNullKey, row));
    });

    // To avoid creating new buffers, we try to reuse the input's buffers
    // as many as possible
    auto rowValueVector = rowVector->childAt(1);
    auto mapVector = std::make_shared<MapVector>(
        context.pool(),
        outputType,
        inputArray->nulls(),
        rows.size(),
        inputArray->offsets(),
        inputArray->sizes(),
        rowKeyVector,
        rowValueVector);

    checkForDuplicateKeys(mapVector, rows, context);
    return mapVector;
  }

  static void checkForDuplicateKeys(
      const MapVectorPtr& mapVector,
      const SelectivityVector& rows,
      exec::EvalCtx& context) {
    MapVector::canonicalize(mapVector);

    static const char* kDuplicateKey = "Duplicate keys ({}) are not allowed";
    auto offsets = mapVector->rawOffsets();
    auto sizes = mapVector->rawSizes();
    auto mapKeys = mapVector->mapKeys();
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      auto offset = offsets[row];
      auto size = sizes[row];
      for (vector_size_t i = 1; i < size; i++) {
        if (mapKeys->equalValueAt(mapKeys.get(), offset + i, offset + i - 1)) {
          auto duplicateKey = mapKeys->wrappedVector()->toString(
              mapKeys->wrappedIndex(offset + i));
          VELOX_USER_FAIL(kDuplicateKey, duplicateKey);
        }
      }
    });
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_map_from_entries,
    MapFromEntriesFunction::signatures(),
    std::make_unique<MapFromEntriesFunction>());
} // namespace facebook::velox::functions
