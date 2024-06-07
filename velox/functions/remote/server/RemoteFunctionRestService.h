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

#include <proxygen/httpserver/HTTPServer.h>
#include "velox/common/memory/Memory.h"

using namespace proxygen;

namespace facebook::velox::functions {
class ErrorHandler : public RequestHandler {
 public:
  explicit ErrorHandler(int statusCode, std::string message);
  void onRequest(std::unique_ptr<HTTPMessage> headers) noexcept override;
  void onBody(std::unique_ptr<folly::IOBuf>) noexcept override;
  void onEOM() noexcept override;
  void onUpgrade(UpgradeProtocol protocol) noexcept override;
  void requestComplete() noexcept override;
  void onError(ProxygenError err) noexcept override;

 private:
  int statusCode_;
  std::string message_;
};

class RestRequestHandler : public RequestHandler {
 public:
  explicit RestRequestHandler(const std::string& functionPrefix = "")
      : functionPrefix_(functionPrefix) {}
  void onRequest(std::unique_ptr<HTTPMessage> headers) noexcept override;
  void onBody(std::unique_ptr<folly::IOBuf> body) noexcept override;
  void onEOM() noexcept override;
  void onUpgrade(UpgradeProtocol protocol) noexcept override;
  void requestComplete() noexcept override;
  void onError(ProxygenError err) noexcept override;

 private:
  std::string body_;
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  const std::string functionPrefix_;
};

class RestRequestHandlerFactory : public RequestHandlerFactory {
 public:
  explicit RestRequestHandlerFactory(const std::string& functionPrefix = "")
      : functionPrefix_(functionPrefix) {}
  void onServerStart(folly::EventBase* evb) noexcept override;
  void onServerStop() noexcept override;
  RequestHandler* onRequest(RequestHandler*, HTTPMessage* msg) noexcept
      override;

 private:
  const std::string functionPrefix_;
};
} // namespace facebook::velox::functions
