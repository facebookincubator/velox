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

namespace facebook::velox::window::prestosql {

namespace {

class CumeDistFunction : public exec::WindowFunction {
 public:
  explicit CumeDistFunction() : WindowFunction(DOUBLE(), nullptr, nullptr) {}

  void resetPartition(const exec::WindowPartition* partition) override {
    runningTotal_ = 0;
    cumeDist_ = 0;
    currentPeerGroupStart_ = -1;
    numPartitionRows_ = partition->numRows();
  }

  void apply(
      const BufferPtr& peerGroupStarts,
      const BufferPtr& peerGroupEnds,
      const BufferPtr& /*frameStarts*/,
      const BufferPtr& /*frameEnds*/,
      const SelectivityVector& validRows,
      vector_size_t resultOffset,
      const VectorPtr& result) override {
    int numRows = peerGroupStarts->size() / sizeof(vector_size_t);
    auto* rawValues = result->asFlatVector<double>()->mutableRawValues();
    auto* peerGroupStartsVector = peerGroupStarts->as<vector_size_t>();
    auto* peerGroupEndsVector = peerGroupEnds->as<vector_size_t>();

    for (int i = 0; i < numRows; i++) {
      auto peerStart = peerGroupStartsVector[i];
      if (peerStart != currentPeerGroupStart_) {
        currentPeerGroupStart_ = peerStart;
        runningTotal_ += peerGroupEndsVector[i] - peerStart + 1;
        cumeDist_ = double(runningTotal_) / numPartitionRows_;
      }
      rawValues[resultOffset + i] = cumeDist_;
    }

    // Set NULL values for rows with empty frames.
    setNullEmptyFramesResults(validRows, resultOffset, result);
  }

 private:
  int64_t runningTotal_ = 0;
  double cumeDist_ = 0;
  int64_t currentPeerGroupStart_ = -1;
  vector_size_t numPartitionRows_ = 1;
};

} // namespace

void registerCumeDist(const std::string& name) {
  std::vector<exec::FunctionSignaturePtr> signatures{
      exec::FunctionSignatureBuilder().returnType("double").build(),
  };

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      [name](
          const std::vector<exec::WindowFunctionArg>& /*args*/,
          const TypePtr& /*resultType*/,
          velox::memory::MemoryPool* /*pool*/,
          HashStringAllocator* /*stringAllocator*/)
          -> std::unique_ptr<exec::WindowFunction> {
        return std::make_unique<CumeDistFunction>();
      });
}
} // namespace facebook::velox::window::prestosql
