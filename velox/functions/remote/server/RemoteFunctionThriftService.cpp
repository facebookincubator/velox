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

#include "velox/functions/remote/server/RemoteFunctionThriftService.h"

#include "velox/functions/remote/if/GetSerde.h"

namespace facebook::velox::functions {

void RemoteFunctionServiceHandler::handleErrors(
    apache::thrift::field_ref<remote::RemoteFunctionPage&> result,
    exec::EvalErrors* evalErrors,
    const std::unique_ptr<VectorSerde>& serde) const {
  const std::int64_t numRows = folly::copy(result->rowCount().value());
  BufferPtr dataBuffer =
      AlignedBuffer::allocate<StringView>(numRows, pool_.get());

  auto flatVector = std::make_shared<FlatVector<StringView>>(
      pool_.get(),
      VARCHAR(),
      nullptr, // null vectors
      numRows,
      std::move(dataBuffer),
      std::vector<BufferPtr>{});

  for (vector_size_t i = 0; i < numRows; ++i) {
    if (evalErrors->hasErrorAt(i)) {
      auto exceptionPtr = *evalErrors->errorAt(i);
      try {
        std::rethrow_exception(*exceptionPtr);
      } catch (const std::exception& ex) {
        flatVector->set(i, ex.what());
      }
    } else {
      flatVector->set(i, StringView());
      flatVector->setNull(i, true);
    }
  }
  auto errorRowVector = std::make_shared<RowVector>(
      pool_.get(),
      ROW({VARCHAR()}),
      BufferPtr(),
      numRows,
      std::vector<VectorPtr>{flatVector});
  result->errorPayload_ref() =
      rowVectorToIOBuf(errorRowVector, *pool_, serde.get());
}

void RemoteFunctionServiceHandler::invokeFunction(
    remote::RemoteFunctionResponse& response,
    std::unique_ptr<remote::RemoteFunctionRequest> request) {
  const auto& functionHandle = request->remoteFunctionHandle().value();
  const auto& inputs = request->inputs().value();

  auto serdeFormat = folly::copy(inputs.pageFormat().value());
  auto serde = getSerde(serdeFormat);

  auto outputRowVector = invokeFunctionInternal(
      inputs.get_payload(),
      functionHandle.argumentTypes().value(),
      functionHandle.returnType().value(),
      functionHandle.name().value(),
      request->get_throwOnError(),
      serde.get());

  auto result = response.result_ref();
  result->rowCount_ref() = outputRowVector->size();
  result->pageFormat_ref() = serdeFormat;
  result->payload_ref() = rowVectorToIOBuf(
      outputRowVector, outputRowVector->size(), *pool_, serde.get());

  auto evalErrors = getEvalErrors_();
  if (evalErrors && evalErrors->hasError()) {
    handleErrors(result, evalErrors, serde);
  }
}

} // namespace facebook::velox::functions
