/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/functions/remote/client/RemoteVectorFunction.h"

#include "velox/expression/VectorFunction.h"
#include "velox/functions/remote/if/GetSerde.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"

namespace facebook::velox::functions {
namespace {

std::string serializeType(const TypePtr& type) {
  // Use hive type serializer.
  return type::fbhive::HiveTypeSerializer::serialize(type);
}

} // namespace

RemoteVectorFunction::RemoteVectorFunction(
    const std::string& functionName,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const RemoteVectorFunctionMetadata& metadata)
    : functionName_(functionName),
      serdeFormat_(metadata.serdeFormat),
      serde_(getSerde(serdeFormat_)),
      serdeOptions_(
          metadata.preserveEncoding
              ? getSerdeOptions(serdeFormat_, metadata.preserveEncoding)
              : nullptr),
      preserveEncoding_(metadata.preserveEncoding) {
  std::vector<TypePtr> types;
  types.reserve(inputArgs.size());
  serializedInputTypes_.reserve(inputArgs.size());

  for (const auto& arg : inputArgs) {
    types.emplace_back(arg.type);
    serializedInputTypes_.emplace_back(serializeType(arg.type));
  }
  remoteInputType_ = ROW(std::move(types));
}

void RemoteVectorFunction::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) const {
  try {
    applyRemote(rows, args, outputType, context, result);
  } catch (const VeloxRuntimeError&) {
    throw;
  } catch (const std::exception&) {
    context.setErrors(rows, std::current_exception());
  }
}

void RemoteVectorFunction::applyRemote(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
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

  // Create the thrift payload.
  remote::RemoteFunctionRequest request;
  request.throwOnError() = context.throwOnError();

  auto functionHandle = request.remoteFunctionHandle();
  functionHandle->name() = functionName_;
  functionHandle->returnType() = serializeType(outputType);
  functionHandle->argumentTypes() = serializedInputTypes_;

  auto requestInputs = request.inputs();
  requestInputs->rowCount() = remoteRowVector->size();
  requestInputs->pageFormat() = serdeFormat_;

  // TODO: serialize only active rows.
  if (preserveEncoding_) {
    requestInputs->payload_ref() = rowVectorToIOBufBatch(
        remoteRowVector,
        rows.end(),
        *context.pool(),
        serde_.get(),
        serdeOptions_.get());
  } else {
    requestInputs->payload_ref() = rowVectorToIOBuf(
        remoteRowVector, rows.end(), *context.pool(), serde_.get());
  }

  std::unique_ptr<remote::RemoteFunctionResponse> remoteResponse;

  // Invoke function that communicates with the remote host.
  try {
    remoteResponse = invokeRemoteFunction(request);
  } catch (const std::exception& e) {
    VELOX_FAIL(
        "Error while executing remote function '{}' at '{}': {}",
        functionName_,
        remoteLocationToString(),
        e.what());
  }

  const auto& remoteResult = remoteResponse->result().value();
  auto outputRowVector = IOBufToRowVector(
      remoteResult.payload().value(),
      ROW({outputType}),
      *context.pool(),
      serde_.get());
  result = outputRowVector->childAt(0);

  if (auto errorPayload = remoteResult.errorPayload()) {
    auto errorsRowVector = IOBufToRowVector(
        *errorPayload, ROW({VARCHAR()}), *context.pool(), serde_.get());
    auto errorsVector = errorsRowVector->childAt(0)->asFlatVector<StringView>();
    VELOX_CHECK(
        errorsVector,
        "Remote function error payload should be convertible to flat vector.");

    SelectivityVector selectedRows(errorsRowVector->size());
    selectedRows.applyToSelected([&](vector_size_t i) {
      if (errorsVector->isNullAt(i)) {
        return;
      }
      try {
        throw std::runtime_error(std::string(errorsVector->valueAt(i)));
      } catch (const std::exception&) {
        context.setError(i, std::current_exception());
      }
    });
  }
}

} // namespace facebook::velox::functions
