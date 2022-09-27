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
#include "velox/exec/WindowFunction.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::window {

namespace {

class NtileFunction : public exec::WindowFunction {
 public:
  explicit NtileFunction(
      const std::vector<TypePtr>& argTypes,
      const std::vector<column_index_t>& argIndices)
      : WindowFunction(BIGINT(), nullptr) {
    if (!argTypes.size() || !argTypes[0]->isBigint() || argIndices[0] <= 0) {
      VELOX_FAIL("Ntile function requires a positive integer argument");
    }
    numBuckets_ = argIndices[0];
  }

  void resetPartition(const exec::WindowPartition* partition) {
    partitionOffset_ = 0;
    numPartitionRows_ = partition->numRows();
    rowsPerBucket_ = numPartitionRows_ / numBuckets_;
    remainderRows_ = numPartitionRows_ % numBuckets_;
  }

  void apply(
      const BufferPtr& peerGroupStarts,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& /*frameStarts*/,
      const BufferPtr& /*frameEnds*/,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    int numRows = peerGroupStarts->size() / sizeof(vector_size_t);
    auto* rawValues = result->asFlatVector<int64_t>()->mutableRawValues();

    for (auto i = 0; i < numRows; i++) {
      vector_size_t currentRow = i + partitionOffset_;
      if (numPartitionRows_ < numBuckets_) {
        rawValues[resultOffset + i] = i + 1;
      } else {
        if (currentRow < (rowsPerBucket_ + 1) * remainderRows_) {
          rawValues[resultOffset + i] = currentRow / (rowsPerBucket_ + 1) + 1;
        } else {
          rawValues[resultOffset + i] =
              (currentRow - remainderRows_) / rowsPerBucket_ + 1;
        }
      }
    }

    partitionOffset_ += numRows;
  }

 private:
  int64_t numBuckets_ = 1;
  int64_t rowsPerBucket_ = 1;
  int64_t remainderRows_ = 0;
  int64_t numPartitionRows_ = 0;
  int64_t partitionOffset_ = 0;
};

} // namespace

void registerNtile(const std::string& name) {
  std::vector<exec::FunctionSignaturePtr> signatures{
      exec::FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
  };

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      [name](
          const std::vector<TypePtr>& argTypes,
          const std::vector<column_index_t>& argIndices,
          const TypePtr& /*resultType*/,
          velox::memory::MemoryPool* /*pool*/)
          -> std::unique_ptr<exec::WindowFunction> {
        auto typeKind = argTypes[0]->kind();
        return std::make_unique<NtileFunction>(argTypes, argIndices);
      });
}
} // namespace facebook::velox::window
