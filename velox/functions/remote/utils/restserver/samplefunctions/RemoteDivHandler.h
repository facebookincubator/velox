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

class RemoteDivHandler : public RemoteFunctionRestHandler {
 public:
  void handleRequest(
      std::unique_ptr<folly::IOBuf> inputBuffer,
      VectorSerde* serde,
      memory::MemoryPool* pool,
      std::function<void(folly::IOBuf&&)> sendResponse) override {
    auto numType = type::fbhive::HiveTypeParser().parse("double");
    auto denType = type::fbhive::HiveTypeParser().parse("double");
    auto outType = type::fbhive::HiveTypeParser().parse("double");

    auto rowType = ROW({numType, denType});
    auto inputVector = IOBufToRowVector(*inputBuffer, rowType, *pool, serde);

    VELOX_CHECK_EQ(
        inputVector->childrenSize(),
        2,
        "Expected exactly 2 columns for function 'remote_divide'.");

    const auto numRows = inputVector->size();

    auto resultVector = BaseVector::create(outType, numRows, pool);

    auto inputFlat0 = inputVector->childAt(0)->asFlatVector<double>();
    auto inputFlat1 = inputVector->childAt(1)->asFlatVector<double>();
    auto outFlat = resultVector->asFlatVector<double>();

    for (vector_size_t i = 0; i < numRows; ++i) {
      if (inputFlat0->isNullAt(i) || inputFlat1->isNullAt(i)) {
        // If either operand is null, result is null
        outFlat->setNull(i, true);
      } else {
        double numerator = inputFlat0->valueAt(i);
        double denominator = inputFlat1->valueAt(i);

        if (denominator == 0.0) {
          outFlat->setNull(i, true);
        } else {
          outFlat->set(i, numerator / denominator);
        }
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
