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
#include "velox/common/base/Exceptions.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/window/NthValueBase.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::window::prestosql {

namespace {
class NthValue : public functions::window::NthValueBase {
 public:
  NthValue(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : NthValueBase(args, resultType, pool) {
    VELOX_USER_CHECK(args[1].type->isBigint(), "Offset must be Bigint");
    if (args[1].constantValue) {
      if (args[1].constantValue->isNullAt(0)) {
        initializeConstantOffset(std::nullopt);
      } else {
        initializeConstantOffset(std::make_optional(
            args[1]
                .constantValue->template as<ConstantVector<int64_t>>()
                ->valueAt(0)));
      }
    } else {
      initializeOffsetVector(
          args[1].index.value(),
          BaseVector::create<FlatVector<int64_t>>(BIGINT(), 0, pool));
    }
  }

 protected:
  void setRowNumbers(
      vector_size_t numRows,
      const SelectivityVector& validRows,
      const vector_size_t* frameStarts,
      const vector_size_t* frameEnds,
      const vector_size_t partitionOffset,
      const exec::WindowPartition* partition,
      const column_index_t offsetIndex,
      const VectorPtr& offsets,
      std::vector<vector_size_t>& rowNumbers) override {
    offsets->resize(numRows);
    partition->extractColumn(offsetIndex, partitionOffset, numRows, 0, offsets);
    auto flatOffsets = offsets->asFlatVector<int64_t>();
    validRows.applyToSelected([&](auto i) {
      if (flatOffsets->isNullAt(i)) {
        rowNumbers[i] = kNullRow;
      } else {
        vector_size_t offset = flatOffsets->valueAt(i);
        VELOX_USER_CHECK_GE(offset, 1, "Offset must be at least 1");
        setRowNumber(i, frameStarts, frameEnds, offset);
      }
    });

    setRowNumbersForEmptyFrames(validRows);
  }
};
} // namespace

void registerNthValue(const std::string& name) {
  // nth_value(T, bigint) -> T.
  std::vector<exec::FunctionSignaturePtr> signatures{
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .argumentType("T")
          .argumentType("bigint")
          .build(),
  };

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      [name](
          const std::vector<exec::WindowFunctionArg>& args,
          const TypePtr& resultType,
          velox::memory::MemoryPool* pool,
          HashStringAllocator*
          /*stringAllocator*/) -> std::unique_ptr<exec::WindowFunction> {
        return std::make_unique<NthValue>(args, resultType, pool);
      });
}
} // namespace facebook::velox::window::prestosql
