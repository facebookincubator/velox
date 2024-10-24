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

#include "velox/functions/remote/client/Remote.h"

#include <folly/io/async/EventBase.h>
#include "velox/common/memory/ByteStream.h"
// #include "velox/common/memory/StreamArena.h"
#include "velox/exec/ExchangeQueue.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/remote/client/RestClient.h"
#include "velox/functions/remote/client/ThriftClient.h"
#include "velox/functions/remote/if/GetSerde.h"
#include "velox/functions/remote/if/gen-cpp2/RemoteFunctionServiceAsyncClient.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"
// #include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorStream.h"

using namespace folly;
using namespace proxygen;
namespace facebook::velox::functions {
namespace {

std::string serializeType(const TypePtr& type) {
  // Use hive type serializer.
  return type::fbhive::HiveTypeSerializer::serialize(type);
}

class RemoteFunction : public exec::VectorFunction {
 public:
  RemoteFunction(
      const std::string& functionName,
      const std::vector<exec::VectorFunctionArg>& inputArgs,
      const RemoteVectorFunctionMetadata& metadata)
      : functionName_(functionName), metadata_(metadata) {
    if (metadata.location.type() == typeid(SocketAddress)) {
      location_ = boost::get<SocketAddress>(metadata.location);
      thriftClient_ = getThriftClient(location_, &eventBase_);
    } else if (metadata.location.type() == typeid(URL)) {
      url_ = boost::get<URL>(metadata.location);
    }

    std::vector<TypePtr> types;
    types.reserve(inputArgs.size());
    serializedInputTypes_.reserve(inputArgs.size());

    for (const auto& arg : inputArgs) {
      types.emplace_back(arg.type);
      serializedInputTypes_.emplace_back(serializeType(arg.type));
    }
    remoteInputType_ = ROW(std::move(types));
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    try {
      if ((metadata_.location.type() == typeid(SocketAddress))) {
        applyRemote(rows, args, outputType, context, result);
      } else if (metadata_.location.type() == typeid(URL)) {
        applyRestRemote(rows, args, outputType, context, result);
      }
    } catch (const VeloxRuntimeError&) {
      throw;
    } catch (const std::exception&) {
      context.setErrors(rows, std::current_exception());
    }
  }

 private:
  const std::string urlEncode(const std::string& value) const {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (char c : value) {
      // Keep alphanumeric characters and some reserved characters unchanged
      if (isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' ||
          c == '.' || c == '~') {
        escaped << c;
      } else {
        // Convert non-alphanumeric characters to %hex representation
        escaped << '%' << std::setw(2) << int(static_cast<unsigned char>(c));
      }
    }

    return escaped.str();
  }

  void applyRestRemote(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const {
    try {
      // Prepare the full URL
      std::string functionId =
          metadata_.functionId.value_or("default_function_id");
      std::string encodedFunctionId = urlEncode(functionId);

      std::string fullUrl = fmt::format(
          "{}/v1/functions/{}/{}/{}/{}",
          url_.getUrl(),
          metadata_.schema.value_or("default_schema"),
          functionName_,
          encodedFunctionId,
          metadata_.version.value_or("default_version"));

      // Prepare headers
      std::unordered_map<std::string, std::string> headers;
      headers["Content-Type"] = "application/octet-stream";
      headers["Accept"] = "application/octet-stream";

      // Create the RowVector from input arguments
      auto remoteRowVector = std::make_shared<RowVector>(
          context.pool(), remoteInputType_, BufferPtr{}, rows.end(), args);

      // Create PrestoVectorSerde instance
      serializer::presto::PrestoVectorSerde serde;

      // Create options for serialization if needed
      serializer::presto::PrestoVectorSerde::PrestoOptions options;

      // Use OStreamOutputStream for serialization
      std::ostringstream out;
      serializer::presto::PrestoOutputStreamListener listener;
      OStreamOutputStream output(&out, &listener);

      // Obtain a BatchVectorSerializer
      auto batchSerializer =
          serde.createBatchSerializer(context.pool(), &options);

      // Serialize the vector
      batchSerializer->serialize(remoteRowVector, &output);

      // Get the serialized data as a string
      std::string serializedData = out.str();

      // Convert the serialized data into an IOBuf
      auto payloadIOBuf = IOBuf::copyBuffer(
          serializedData.data(), serializedData.size());

      // Create a SerializedPage from the IOBuf
      exec::SerializedPage requestPage(std::move(payloadIOBuf));

      // Invoke the REST function with the SerializedPage
      RestClient restClient(fullUrl, headers);

      // Send the SerializedPage and receive the response as a SerializedPage
      auto [statusCode, responsePage] = restClient.invoke_function(requestPage);

      // Handle HTTP response status
      if (statusCode != 200) {
        VELOX_FAIL(
            "Error while executing remote function '{}': HTTP status code {}",
            functionName_,
            statusCode);
      }

      // Deserialize the response SerializedPage back into a RowVector
      auto inputByteRanges =
          byteRangesFromIOBuf(responsePage->getIOBuf().get());
      BufferInputStream inputStream(std::move(inputByteRanges));

      // Prepare the output RowVectorPtr
      RowVectorPtr outputRowVector;

      // Deserialize using PrestoVectorSerde
      serde.deserialize(
          &inputStream,
          context.pool(),
          remoteInputType_,
          &outputRowVector,
          nullptr);

      // Extract the result column
      result = outputRowVector->childAt(0);

    } catch (const std::exception& e) {
      // Log and throw an error if the remote call fails
      VELOX_FAIL(
          "Error while executing remote function '{}': {}",
          functionName_,
          e.what());
    }
  }

  void applyRemote(
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

    // Send to remote server.
    remote::RemoteFunctionResponse remoteResponse;
    remote::RemoteFunctionRequest request;
    request.throwOnError_ref() = context.throwOnError();

    auto functionHandle = request.remoteFunctionHandle_ref();
    functionHandle->name_ref() = functionName_;
    functionHandle->returnType_ref() = serializeType(outputType);
    functionHandle->argumentTypes_ref() = serializedInputTypes_;

    auto requestInputs = request.inputs_ref();
    requestInputs->rowCount_ref() = remoteRowVector->size();
    requestInputs->pageFormat_ref() = metadata_.serdeFormat;

    // TODO: serialize only active rows.
    requestInputs->payload_ref() = rowVectorToIOBuf(
        remoteRowVector,
        rows.end(),
        *context.pool(),
        getSerde(metadata_.serdeFormat).get());

    try {
      thriftClient_->sync_invokeFunction(remoteResponse, request);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Error while executing remote function '{}' at '{}': {}",
          functionName_,
          location_.describe(),
          e.what());
    }

    auto outputRowVector = IOBufToRowVector(
        remoteResponse.get_result().get_payload(),
        ROW({outputType}),
        *context.pool(),
        getSerde(metadata_.serdeFormat).get());
    result = outputRowVector->childAt(0);

    if (auto errorPayload = remoteResponse.get_result().errorPayload()) {
      auto errorsRowVector = IOBufToRowVector(
          *errorPayload,
          ROW({VARCHAR()}),
          *context.pool(),
          getSerde(metadata_.serdeFormat).get());
      auto errorsVector =
          errorsRowVector->childAt(0)->asFlatVector<StringView>();
      VELOX_CHECK(errorsVector, "Should be convertible to flat vector");

      SelectivityVector selectedRows(errorsRowVector->size());
      selectedRows.applyToSelected([&](vector_size_t i) {
        if (errorsVector->isNullAt(i)) {
          return;
        }
        try {
          throw std::runtime_error(errorsVector->valueAt(i));
        } catch (const std::exception& ex) {
          context.setError(i, std::current_exception());
        }
      });
    }
  }

  const std::string functionName_;

  EventBase eventBase_;
  std::unique_ptr<RemoteFunctionClient> thriftClient_;
  SocketAddress location_;

  proxygen::URL url_;

  // Structures we construct once to cache:
  RowTypePtr remoteInputType_;
  std::vector<std::string> serializedInputTypes_;

  const RemoteVectorFunctionMetadata metadata_;
};

std::shared_ptr<exec::VectorFunction> createRemoteFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/,
    const RemoteVectorFunctionMetadata& metadata) {
  return std::make_unique<RemoteFunction>(name, inputArgs, metadata);
}

} // namespace

void registerRemoteFunction(
    const std::string& name,
    std::vector<exec::FunctionSignaturePtr> signatures,
    const RemoteVectorFunctionMetadata& metadata,
    bool overwrite) {
  exec::registerStatefulVectorFunction(
      name,
      signatures,
      std::bind(
          createRemoteFunction,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3,
          metadata),
      metadata,
      overwrite);
}

} // namespace facebook::velox::functions
