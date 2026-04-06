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

#include "velox/common/base/BitUtil.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/remote/if/GetSerde.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"

namespace facebook::velox::functions {
namespace {

std::string serializeType(const TypePtr& type) {
  // Use hive type serializer.
  return type::fbhive::HiveTypeSerializer::serialize(type);
}

/// Convert a SelectivityVector into contiguous IndexRange runs of selected
/// rows.
std::vector<IndexRange> toIndexRanges(const SelectivityVector& rows) {
  std::vector<IndexRange> ranges;
  vector_size_t rangeStart{-1};
  vector_size_t rangeSize{0};

  bits::forEachSetBit(
      rows.allBits(), rows.begin(), rows.end(), [&](vector_size_t row) {
        if (rangeStart == -1) {
          rangeStart = row;
          rangeSize = 1;
        } else if (row == rangeStart + rangeSize) {
          ++rangeSize;
        } else {
          ranges.push_back({rangeStart, rangeSize});
          rangeStart = row;
          rangeSize = 1;
        }
      });

  if (rangeStart != -1) {
    ranges.push_back({rangeStart, rangeSize});
  }
  return ranges;
}

/// Count total rows across all ranges.
vector_size_t countRows(const std::vector<IndexRange>& ranges) {
  vector_size_t count{0};
  for (const auto& range : ranges) {
    count += range.size;
  }
  return count;
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
  requestInputs->pageFormat() = serdeFormat_;

  // Serialize only active rows to reduce network and server overhead.
  const bool allSelected = rows.isAllSelected();
  std::vector<IndexRange> activeRanges;

  if (allSelected) {
    activeRanges.push_back({0, rows.end()});
  } else {
    activeRanges = toIndexRanges(rows);
  }

  const auto numActiveRows =
      allSelected ? rows.end() : countRows(activeRanges);
  requestInputs->rowCount() = numActiveRows;

  auto rangesView = folly::Range<const IndexRange*>(
      activeRanges.data(), activeRanges.size());
  if (preserveEncoding_) {
    auto serializer =
        serde_->createBatchSerializer(context.pool(), serdeOptions_.get());
    IOBufOutputStream stream(*context.pool());
    Scratch scratch;
    serializer->serialize(remoteRowVector, rangesView, scratch, &stream);
    requestInputs->payload_ref() = std::move(*stream.getIOBuf());
  } else {
    auto streamGroup =
        std::make_unique<VectorStreamGroup>(context.pool(), serde_.get());
    streamGroup->createStreamTree(
        asRowType(remoteRowVector->type()), numActiveRows);
    Scratch scratch;
    streamGroup->append(remoteRowVector, rangesView, scratch);
    IOBufOutputStream stream(*context.pool());
    streamGroup->flush(&stream);
    requestInputs->payload_ref() = std::move(*stream.getIOBuf());
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
  auto compactedResult = outputRowVector->childAt(0);

  // Scatter compacted result back to original row positions if needed.
  if (allSelected) {
    result = compactedResult;
  } else {
    result = BaseVector::create(outputType, rows.end(), context.pool());
    vector_size_t compactIdx{0};
    rows.applyToSelected([&](vector_size_t origRow) {
      result->copy(compactedResult.get(), origRow, compactIdx, 1);
      ++compactIdx;
    });
  }

  if (auto errorPayload = remoteResult.errorPayload()) {
    auto errorsRowVector = IOBufToRowVector(
        *errorPayload, ROW({VARCHAR()}), *context.pool(), serde_.get());
    auto errorsVector = errorsRowVector->childAt(0)->asFlatVector<StringView>();
    VELOX_CHECK(
        errorsVector,
        "Remote function error payload should be convertible to flat vector.");

    // Map compacted error indices back to original row positions.
    if (allSelected) {
      SelectivityVector selectedRows(errorsRowVector->size());
      selectedRows.applyToSelected([&](vector_size_t i) {
        if (errorsVector->isNullAt(i)) {
          return;
        }
        try {
          VELOX_USER_FAIL("{}", errorsVector->valueAt(i));
        } catch (const std::exception&) {
          context.setError(i, std::current_exception());
        }
      });
    } else {
      vector_size_t compactIdx{0};
      rows.applyToSelected([&](vector_size_t origRow) {
        if (compactIdx < errorsRowVector->size() &&
            !errorsVector->isNullAt(compactIdx)) {
          try {
            VELOX_USER_FAIL("{}", errorsVector->valueAt(compactIdx));
          } catch (const std::exception&) {
            context.setError(origRow, std::current_exception());
          }
        }
        ++compactIdx;
      });
    }
  }
}

} // namespace facebook::velox::functions
