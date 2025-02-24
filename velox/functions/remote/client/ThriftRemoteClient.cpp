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

#include "velox/functions/remote/client/ThriftRemoteClient.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"

namespace facebook::velox::functions {

ThriftRemoteClient::ThriftRemoteClient(
    const folly::SocketAddress& address,
    const std::string& functionName,
    RowTypePtr remoteInputType,
    std::vector<std::string> serializedInputTypes,
    const RemoteVectorFunctionMetadata& metadata)
    : RemoteClient(
          functionName,
          std::move(remoteInputType),
          std::move(serializedInputTypes),
          metadata),
      thriftClient_(getThriftClient(address, &eventBase_)) {}

void ThriftRemoteClient::applyRemote(
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) const {
  // Create type and row vector for serialization.
  auto remoteRowVector = std::make_shared<RowVector>(
      context.pool(),
      remoteInputType_,
      BufferPtr{},
      rows.end(),
      std::move(args));

  // Send to remote server.
  remote::RemoteFunctionResponse remoteResponse;
  remote::RemoteFunctionRequest request;
  request.throwOnError_ref() = context.throwOnError();

  auto functionHandle = request.remoteFunctionHandle_ref();
  functionHandle->name_ref() = functionName_;
  functionHandle->returnType_ref() =
      type::fbhive::HiveTypeSerializer::serialize(outputType);
  functionHandle->argumentTypes_ref() = serializedInputTypes_;

  auto requestInputs = request.inputs_ref();
  requestInputs->rowCount_ref() = remoteRowVector->size();
  requestInputs->pageFormat_ref() = serdeFormat_;

  // TODO: serialize only active rows.
  requestInputs->payload_ref() = rowVectorToIOBuf(
      remoteRowVector, rows.end(), *context.pool(), serde_.get());

  try {
    thriftClient_->sync_invokeFunction(remoteResponse, request);
  } catch (const std::exception& e) {
    VELOX_FAIL(
        "Error while executing remote function '{}' at '{}': {}",
        functionName_,
        std::get<folly::SocketAddress>(metadata_.location).describe(),
        e.what());
  }

  auto outputRowVector = IOBufToRowVector(
      remoteResponse.result().value().payload().value(),
      ROW({outputType}),
      *context.pool(),
      serde_.get());
  result = outputRowVector->childAt(0);

  if (auto errorPayload = remoteResponse.result().value().errorPayload()) {
    auto errorsRowVector = IOBufToRowVector(
        *errorPayload, ROW({VARCHAR()}), *context.pool(), serde_.get());
    auto errorsVector = errorsRowVector->childAt(0)->asFlatVector<StringView>();
    VELOX_CHECK(errorsVector, "Should be convertible to flat vector");

    SelectivityVector selectedRows(errorsRowVector->size());
    selectedRows.applyToSelected([&](vector_size_t i) {
      if (errorsVector->isNullAt(i)) {
        return;
      }
      try {
        throw std::runtime_error(errorsVector->valueAt(i));
      } catch (const std::exception&) {
        context.setError(i, std::current_exception());
      }
    });
  }
}

} // namespace facebook::velox::functions
