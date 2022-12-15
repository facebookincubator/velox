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
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/expression/VectorReaders.h"
#include "velox/expression/VectorWriters.h"

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
    MapVectorPtr mapVector;

    auto inputArray = arg->as<ArrayVector>();
    auto rowVector = inputArray->elements();
    auto rowKeys = rowVector->as<RowVector>()->childAt(0);
    auto rowValues = rowVector->as<RowVector>()->childAt(1);

    DecodedVector rowVectorDecoded;
    rowVectorDecoded.decode(*rowVector, rows);
    exec::VectorReader<Any> mapEntryReader(&rowVectorDecoded);

    DecodedVector rowKeyVectorDecoded;
    rowKeyVectorDecoded.decode(*rowKeys, rows);
    exec::VectorReader<Any> mapKeyReader(&rowKeyVectorDecoded);

    auto offsets = inputArray->rawOffsets();
    auto sizes = inputArray->rawSizes();

    rows.applyToSelected([&](vector_size_t row) {
      VELOX_USER_CHECK(
          mapEntryReader.isSet(row),
          fmt::format("map entry at {} cannot be null", row));
      VELOX_USER_CHECK(
          mapKeyReader.isSet(row),
          fmt::format("map key at {} cannot be null", row));

      // check for duplicate keys
      auto offset = offsets[row];
      auto size = sizes[row];
      for (vector_size_t i = 1; i < size; i++) {
        if (rowKeys->equalValueAt(rowKeys.get(), offset + i, offset + i - 1)) {
          auto duplicateKey = rowKeys->wrappedVector()->toString(
              rowKeys->wrappedIndex(offset + i));
          VELOX_USER_FAIL(
              fmt::format("Duplicate keys ({}) are not allowed", duplicateKey));
        }
      }
    });

    // To avoid creating new buffers, we try to reuse the input's buffers
    // as many as possible
    mapVector = std::make_shared<MapVector>(
        context.pool(),
        outputType,
        inputArray->nulls(),
        rows.size(),
        inputArray->offsets(),
        inputArray->sizes(),
        rowKeys,
        rowValues);

    context.moveOrCopyResult(mapVector, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // array(row(K,V)) -> map(K,V)
    return {exec::FunctionSignatureBuilder()
                .knownTypeVariable("K")
                .typeVariable("V")
                .returnType("map(K,V)")
                .argumentType("array(row(K,V))")
                .build()};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_map_from_entries,
    MapFromEntriesFunction::signatures(),
    std::make_unique<MapFromEntriesFunction>());
} // namespace facebook::velox::functions
