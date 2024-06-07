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

#include "velox/functions/remote/server/RemoteFunctionRestService.h"
#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/ResponseBuilder.h>
#include "velox/expression/Expr.h"
#include "velox/functions/remote/if/GetSerde.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::functions {
namespace {
std::string iobufToString(const folly::IOBuf& buf) {
  std::string result;
  result.reserve(buf.computeChainDataLength());

  for (auto range : buf) {
    result.append(reinterpret_cast<const char*>(range.data()), range.size());
  }

  return result;
}

TypePtr deserializeType(const std::string& input) {
  // Use hive type parser/serializer.
  return type::fbhive::HiveTypeParser().parse(input);
}

RowTypePtr deserializeArgTypes(const std::vector<std::string>& argTypes) {
  const size_t argCount = argTypes.size();

  std::vector<TypePtr> argumentTypes;
  std::vector<std::string> typeNames;
  argumentTypes.reserve(argCount);
  typeNames.reserve(argCount);

  for (size_t i = 0; i < argCount; ++i) {
    argumentTypes.emplace_back(deserializeType(argTypes[i]));
    typeNames.emplace_back(fmt::format("c{}", i));
  }
  return ROW(std::move(typeNames), std::move(argumentTypes));
}

std::string getFunctionName(
    const std::string& prefix,
    const std::string& functionName) {
  return prefix.empty() ? functionName
                        : fmt::format("{}.{}", prefix, functionName);
}
} // namespace

std::vector<core::TypedExprPtr> getExpressions(
    const RowTypePtr& inputType,
    const TypePtr& returnType,
    const std::string& functionName) {
  std::vector<core::TypedExprPtr> inputs;
  for (size_t i = 0; i < inputType->size(); ++i) {
    inputs.push_back(std::make_shared<core::FieldAccessTypedExpr>(
        inputType->childAt(i), inputType->nameOf(i)));
  }

  return {std::make_shared<core::CallTypedExpr>(
      returnType, std::move(inputs), functionName)};
}

// RestRequestHandler
void RestRequestHandler::onRequest(
    std::unique_ptr<HTTPMessage> headers) noexcept {}

void RestRequestHandler::onEOM() noexcept {
  try {
    auto jsonObj = folly::parseJson(body_);

    auto payload = jsonObj["inputs"]["payload"];
    auto rowCount = jsonObj["inputs"]["rowCount"];
    auto remoteFunctionHandle = jsonObj["remoteFunctionHandle"];

    LOG(INFO) << "Got a request for '" << remoteFunctionHandle["functionName"]
              << "': " << rowCount << " input rows.";

    if (!jsonObj["throwOnError"].asBool()) {
      VELOX_NYI("throwOnError not implemented yet on remote server.");
    }

    // A remote function service should handle the function execution by its
    // own. We use Velox eval framework here for quick prototype.
    // Start of Function execution
    std::vector<std::string> argumentTypes;
    for (const auto& element : remoteFunctionHandle["argumentTypes"]) {
      argumentTypes.push_back(element.asString());
    }
    auto inputType = deserializeArgTypes(argumentTypes);
    auto outputType =
        deserializeType(remoteFunctionHandle["returnType"].asString());

    auto serdeFormat = static_cast<remote::PageFormat>(
        jsonObj["inputs"]["pageFormat"].asInt());
    auto serde = getSerde(serdeFormat);

    // jsonObj to RowVector
    auto inputVector = IOBufToRowVector(
        *folly::IOBuf::copyBuffer(payload.asString()),
        inputType,
        *pool_,
        serde.get());

    const vector_size_t numRows = inputVector->size();
    SelectivityVector rows{numRows};

    // Expression boilerplate.
    auto queryCtx = core::QueryCtx::create();
    core::ExecCtx execCtx{pool_.get(), queryCtx.get()};
    exec::ExprSet exprSet{
        getExpressions(
            inputType,
            outputType,
            getFunctionName(
                functionPrefix_,
                remoteFunctionHandle["functionName"].asString())),
        &execCtx};
    exec::EvalCtx evalCtx(&execCtx, &exprSet, inputVector.get());

    std::vector<VectorPtr> expressionResult;
    exprSet.eval(rows, evalCtx, expressionResult);

    // Create output vector.
    auto outputRowVector = std::make_shared<RowVector>(
        pool_.get(), ROW({outputType}), BufferPtr(), numRows, expressionResult);

    // Construct a json object for REST response
    // End of Function execution.
    folly::dynamic retObj = folly::dynamic::object;
    retObj["payload"] = iobufToString(
        rowVectorToIOBuf(outputRowVector, rows.end(), *pool_, serde.get()));
    retObj["rowCount"] = outputRowVector->size();

    // LOG(INFO) << "result:" << retObj;
    ResponseBuilder(downstream_)
        .status(200, "OK")
        .body(folly::toJson(folly::dynamic::object("result", retObj)))
        .sendWithEOM();

  } catch (const std::exception& ex) {
    LOG(ERROR) << ex.what();
    ResponseBuilder(downstream_)
        .status(500, "Internal Server Error")
        .body(folly::toJson(folly::dynamic::object("err", ex.what())))
        .sendWithEOM();
  }
}

void RestRequestHandler::onBody(std::unique_ptr<folly::IOBuf> chain) noexcept {
  if (chain) {
    body_.append(reinterpret_cast<const char*>(chain->data()), chain->length());
  }
}

void RestRequestHandler::onUpgrade(UpgradeProtocol /*protocol*/) noexcept {
  // handler doesn't support upgrades
}

void RestRequestHandler::requestComplete() noexcept {
  delete this;
}

void RestRequestHandler::onError(ProxygenError /*err*/) noexcept {
  delete this;
}

// ErrorHandler
ErrorHandler::ErrorHandler(int statusCode, std::string message)
    : statusCode_(statusCode), message_(std::move(message)) {}

void ErrorHandler::onRequest(std::unique_ptr<HTTPMessage>) noexcept {
  ResponseBuilder(downstream_)
      .status(statusCode_, "Error")
      .body(std::move(message_))
      .sendWithEOM();
}

void ErrorHandler::onEOM() noexcept {}

void ErrorHandler::onBody(std::unique_ptr<folly::IOBuf> body) noexcept {}

void ErrorHandler::onUpgrade(UpgradeProtocol protocol) noexcept {
  // handler doesn't support upgrades
}

void ErrorHandler::requestComplete() noexcept {
  delete this;
}

void ErrorHandler::onError(ProxygenError err) noexcept {
  delete this;
}

// RestRequestHandlerFactory
void RestRequestHandlerFactory::onServerStart(folly::EventBase* evb) noexcept {}

void RestRequestHandlerFactory::onServerStop() noexcept {}

RequestHandler* RestRequestHandlerFactory::onRequest(
    proxygen::RequestHandler*,
    proxygen::HTTPMessage* msg) noexcept {
  if (msg->getMethod() != HTTPMethod::POST) {
    return new ErrorHandler(405, "Only POST method is allowed");
  }
  return new RestRequestHandler(functionPrefix_);
}
} // namespace facebook::velox::functions
