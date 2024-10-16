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
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/remote/client/RestClient.h"
#include "velox/functions/remote/client/ThriftClient.h"
#include "velox/functions/remote/if/GetSerde.h"
#include "velox/functions/remote/if/gen-cpp2/RemoteFunctionServiceAsyncClient.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::functions {
namespace {

std::string serializeType(const TypePtr& type) {
  // Use hive type serializer.
  return type::fbhive::HiveTypeSerializer::serialize(type);
}

std::string iobufToString(const folly::IOBuf& buf) {
  std::string result;
  result.reserve(buf.computeChainDataLength());

  for (auto range : buf) {
    result.append(reinterpret_cast<const char*>(range.data()), range.size());
  }

  return result;
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
          if (isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.' || c == '~') {
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
      std::string responseBody;

      // Create a RowVector for the remote function call
      auto remoteRowVector = std::make_shared<RowVector>(
          context.pool(),
          remoteInputType_,
          BufferPtr{},
          rows.end(),
          std::move(args));

      // Build the JSON request with function and input details
      folly::dynamic remoteFunctionHandle = folly::dynamic::object;
      remoteFunctionHandle["functionName"] = functionName_;
      remoteFunctionHandle["returnType"] = serializeType(outputType);
      remoteFunctionHandle["argumentTypes"] = folly::dynamic::array;
      for (const auto& value : serializedInputTypes_) {
        remoteFunctionHandle["argumentTypes"].push_back(value);
      }

      folly::dynamic inputs = folly::dynamic::object;
      inputs["pageFormat"] = static_cast<int>(metadata_.serdeFormat);
      inputs["payload"] = iobufToString(rowVectorToIOBuf(
          remoteRowVector,
          rows.end(),
          *context.pool(),
          getSerde(metadata_.serdeFormat).get()));
      inputs["rowCount"] = remoteRowVector->size();

      // Create the final JSON object to be sent
      folly::dynamic jsonObject = folly::dynamic::object;
      jsonObject["remoteFunctionHandle"] = remoteFunctionHandle;
      jsonObject["inputs"] = inputs;
      jsonObject["throwOnError"] = context.throwOnError();

      std::string functionid = metadata_.functionId.value_or("default_function_id");
      std::string encodedFunctionId = urlEncode(functionid);

      // Construct the full URL for the REST request
      std::string fullUrl = fmt::format(
          "{}/v1/functions/{}/{}/{}/{}",
          url_.getUrl(),
          metadata_.schema.value_or("default_schema"),
          functionName_,
          encodedFunctionId,
          metadata_.version.value_or("default_version"));

      // Invoke the remote function using RestClient
      RestClient restClient_(fullUrl);
      restClient_.invoke_function(folly::toJson(jsonObject), responseBody);
      LOG(INFO) << responseBody;

      // Parse the JSON response
      auto responseJsonObj = parseJson(responseBody);
      if (responseJsonObj.count("err") > 0) {
        VELOX_NYI(responseJsonObj["err"].asString());
      }

      // Deserialize the result payload
      auto payloadIObuf = folly::IOBuf::copyBuffer(
          responseJsonObj["result"]["payload"].asString());

      auto outputRowVector = IOBufToRowVector(
          *payloadIObuf,
          ROW({outputType}),
          *context.pool(),
          getSerde(metadata_.serdeFormat).get());
      result = outputRowVector->childAt(0);

    } catch (const std::exception& e) {
      // Log and throw an error if the remote call fails
      VELOX_FAIL(
          "Error while executing remote function '{}' at '{}': {}",
          functionName_,
          url_.getUrl(),
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
          *errorPayload, ROW({VARCHAR()}), *context.pool(), getSerde(metadata_.serdeFormat).get());
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

  folly::EventBase eventBase_;
  std::unique_ptr<RemoteFunctionClient> thriftClient_;
  folly::SocketAddress location_;

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
