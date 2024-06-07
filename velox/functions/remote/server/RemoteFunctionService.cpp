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

#include "velox/functions/remote/server/RemoteFunctionService.h"
#include "velox/common/base/Exceptions.h"
#include "velox/functions/remote/if/GetSerde.h"

namespace facebook::velox::functions {

void RemoteFunctionServiceHandler::handleErrors(
    apache::thrift::field_ref<remote::RemoteFunctionPage&> result,
    exec::EvalErrors* evalErrors,
    const std::unique_ptr<VectorSerde>& serde) const {
  const std::int64_t numRows = result->get_rowCount();
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
  const auto& functionHandle = request->get_remoteFunctionHandle();
  const auto& inputs = request->get_inputs();

  auto serdeFormat = inputs.get_pageFormat();
  auto serde = getSerde(serdeFormat);

  auto outputRowVector = invokeFunctionInternal(
      inputs.get_payload(),
      functionHandle.get_argumentTypes(),
      functionHandle.get_returnType(),
      functionHandle.get_name(),
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
