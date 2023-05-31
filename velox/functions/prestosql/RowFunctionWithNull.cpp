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

namespace facebook::velox::functions {
namespace {

class RowFunctionWithNull : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto argsCopy = args;

    BufferPtr nulls = AlignedBuffer::allocate<char>(
        bits::nbytes(rows.size()), context.pool(), 1);
    auto* nullsPtr = nulls->asMutable<uint64_t>();
    auto cntNull = 0;
    rows.applyToSelected([&](vector_size_t i) {
      bits::clearNull(nullsPtr, i);
      if (!bits::isBitNull(nullsPtr, i)) {
        for (size_t c = 0; c < argsCopy.size(); c++) {
          auto arg = argsCopy[c].get();
          if (arg->mayHaveNulls() && arg->isNullAt(i)) {
            // If any argument of the struct is null, set the struct as null.
            bits::setNull(nullsPtr, i, true);
            cntNull++;
            break;
          }
        }
      }
    });

    RowVectorPtr localResult = std::make_shared<RowVector>(
        context.pool(),
        outputType,
        nulls,
        rows.size(),
        std::move(argsCopy),
        cntNull /*nullCount*/);
    context.moveOrCopyResult(localResult, rows, result);
  }

  bool isDefaultNullBehavior() const override {
    return false;
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_concat_row_with_null,
    std::vector<std::shared_ptr<exec::FunctionSignature>>{},
    std::make_unique<RowFunctionWithNull>());

} // namespace facebook::velox::functions
