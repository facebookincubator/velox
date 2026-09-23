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

#include "velox/functions/remote/client/RemoteVectorFunction.h"

#include <folly/coro/BlockingWait.h>

#include "velox/expression/VectorFunction.h"
#include "velox/functions/remote/if/GetSerde.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"

namespace facebook::velox::functions {
namespace {

std::string serializeType(const TypePtr& type) {
  // Use hive type serializer.
  return type::fbhive::HiveTypeSerializer::serialize(type);
}

// Maps compacted position -> original row number, for the selected rows only.
BufferPtr compactedToOriginalIndices(
    const SelectivityVector& rows,
    memory::MemoryPool* pool) {
  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(rows.countSelected(), pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  vector_size_t compactedPosition = 0;
  rows.applyToSelected(
      [&](vector_size_t row) { rawIndices[compactedPosition++] = row; });
  return indices;
}

// Maps original row number -> compacted position, inverting
// 'compactedToOriginal' so the remote result can be scattered back onto the
// rows the caller selected. Unselected positions are never read, so they are
// left pointing at the first compacted row.
BufferPtr originalToCompactedIndices(
    const BufferPtr& compactedToOriginal,
    vector_size_t numCompacted,
    vector_size_t size,
    memory::MemoryPool* pool) {
  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(size, pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  std::fill(rawIndices, rawIndices + size, 0);

  const auto* rawCompactedToOriginal = compactedToOriginal->as<vector_size_t>();
  for (vector_size_t compactedPosition = 0; compactedPosition < numCompacted;
       ++compactedPosition) {
    rawIndices[rawCompactedToOriginal[compactedPosition]] = compactedPosition;
  }
  return indices;
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
  const vector_size_t numSelected = rows.countSelected();
  const bool shouldCompact =
      !preserveEncoding_ && numSelected > 0 && numSelected < rows.end();

  BufferPtr compactedToOriginal;
  if (shouldCompact) {
    compactedToOriginal = compactedToOriginalIndices(rows, context.pool());
    for (auto& arg : args) {
      arg = BaseVector::wrapInDictionary(
          BufferPtr{}, compactedToOriginal, numSelected, arg);
    }
  }
  // Create type and row vector for serialization.
  auto remoteRowVector = std::make_shared<RowVector>(
      context.pool(),
      remoteInputType_,
      BufferPtr{},
      shouldCompact ? numSelected : rows.end(),
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

  if (preserveEncoding_) {
    requestInputs->payload_ref() = rowVectorToIOBufBatch(
        remoteRowVector, *context.pool(), serde_.get(), serdeOptions_.get());
  } else {
    requestInputs->payload_ref() =
        rowVectorToIOBuf(remoteRowVector, *context.pool(), serde_.get());
  }

  std::unique_ptr<remote::RemoteFunctionResponse> remoteResponse;

  // Invoke function that communicates with the remote host.
  try {
    remoteResponse = folly::coro::blockingWait(invokeRemoteFunction(request));
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
  if (shouldCompact) {
    result = BaseVector::wrapInDictionary(
        BufferPtr{},
        originalToCompactedIndices(
            compactedToOriginal, numSelected, rows.end(), context.pool()),
        rows.end(),
        result);
  }

  if (auto errorPayload = remoteResult.errorPayload()) {
    auto errorsRowVector = IOBufToRowVector(
        *errorPayload, ROW({VARCHAR()}), *context.pool(), serde_.get());
    auto errorsVector = errorsRowVector->childAt(0)->asFlatVector<StringView>();
    VELOX_CHECK(
        errorsVector,
        "Remote function error payload should be convertible to flat vector.");

    // Error rows are positions in the payload that was sent, so they need the
    // same mapping back to original rows that the result does.
    const auto* rawCompactedToOriginal =
        shouldCompact ? compactedToOriginal->as<vector_size_t>() : nullptr;

    SelectivityVector selectedRows(errorsRowVector->size());
    selectedRows.applyToSelected([&](vector_size_t i) {
      if (errorsVector->isNullAt(i)) {
        return;
      }
      try {
        VELOX_USER_FAIL("{}", errorsVector->valueAt(i));
      } catch (const std::exception&) {
        context.setError(
            rawCompactedToOriginal != nullptr ? rawCompactedToOriginal[i] : i,
            std::current_exception());
      }
    });
  }
}

} // namespace facebook::velox::functions
