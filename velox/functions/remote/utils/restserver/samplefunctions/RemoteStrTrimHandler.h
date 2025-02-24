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

#pragma once

#include "velox/functions/remote/utils/restserver/RemoteFunctionRestHandler.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions {

class RemoteStrTrimHandler : public RemoteFunctionRestHandler {
 public:
  void handleRequest(
      std::unique_ptr<folly::IOBuf> inputBuffer,
      VectorSerde* serde,
      memory::MemoryPool* pool,
      std::function<void(folly::IOBuf&&)> sendResponse) override {
    auto argType = type::fbhive::HiveTypeParser().parse({"varchar"});
    auto outType = type::fbhive::HiveTypeParser().parse("varchar");

    auto inputVector =
        IOBufToRowVector(*inputBuffer, ROW({argType}), *pool, serde);
    VELOX_CHECK_EQ(
        inputVector->childrenSize(),
        1,
        "Expected exactly 1 column for function 'remote_strlen'.");

    vector_size_t numRows = inputVector->size();
    auto resultVector = BaseVector::create(outType, numRows, pool);

    auto inputFlat = inputVector->childAt(0)->asFlatVector<StringView>();
    auto outFlat = resultVector->asFlatVector<StringView>();

    for (vector_size_t i = 0; i < numRows; ++i) {
      if (inputFlat->isNullAt(i)) {
        outFlat->setNull(i, true);
      } else {
        std::string result = inputFlat->valueAt(i).str();
        result.erase(
            std::remove_if(result.begin(), result.end(), ::isspace),
            result.end());
        outFlat->set(i, result.data());
      }
    }

    auto outputRowVector = std::make_shared<RowVector>(
        pool,
        ROW({outType}),
        BufferPtr(),
        numRows,
        std::vector<VectorPtr>{resultVector});

    auto payload = rowVectorToIOBuf(
        outputRowVector, outputRowVector->size(), *pool, serde);

    sendResponse(std::move(payload));
  }
};

} // namespace facebook::velox::functions
