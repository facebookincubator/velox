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

#include <folly/init/Init.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseManager.h>
#include <folly/io/async/SSLContext.h>
#include <folly/json.h>
#include <proxygen/lib/http/HTTPConnector.h>
#include <proxygen/lib/http/HTTPMessage.h>
#include <proxygen/lib/http/session/HTTPUpstreamSession.h>
#include <proxygen/lib/utils/URL.h>
#include "velox/functions/remote/client/RestClient.h"

using namespace proxygen;
using namespace folly;

namespace facebook::velox::functions {

class HttpClient : public HTTPConnector::Callback,
                   public HTTPTransactionHandler {
 public:
  HttpClient(const URL& url) : url_(url) {}

  void send(std::string requestBody) {
    requestBody_ = requestBody;
    connector_ = std::make_unique<proxygen::HTTPConnector>(
        this, WheelTimerInstance(std::chrono::milliseconds(1000)));
    connector_->connect(
        &evb_,
        SocketAddress(url_.getHost(), url_.getPort(), true),
        std::chrono::milliseconds(10000));
    evb_.loop();
  }

  std::string getResponseBody() {
    return std::move(responseBody_);
  }

 private:
  URL url_;
  EventBase evb_;
  std::unique_ptr<HTTPConnector> connector_;
  std::shared_ptr<HTTPUpstreamSession> session_;
  std::string requestBody_;
  std::string responseBody_;

  void connectSuccess(HTTPUpstreamSession* session) noexcept override {
    session_ = std::shared_ptr<HTTPUpstreamSession>(
        session, [](HTTPUpstreamSession* s) {
          // No-op deleter, managed by Proxygen
        });
    sendRequest();
  }

  void connectError(const folly::AsyncSocketException& ex) noexcept override {
    LOG(ERROR) << "Failed to connect: " << ex.what();
    evb_.terminateLoopSoon();
  }

  void sendRequest() {
    auto txn = session_->newTransaction(this);
    HTTPMessage req;
    req.setMethod(HTTPMethod::POST);
    req.setURL(url_.getUrl());
    req.getHeaders().add(HTTP_HEADER_CONTENT_TYPE, "application/json");
    req.getHeaders().add(
        HTTP_HEADER_CONTENT_LENGTH, std::to_string(requestBody_.size()));
    req.getHeaders().add(HTTP_HEADER_USER_AGENT, "Velox HTTPClient");

    txn->sendHeaders(req);
    txn->sendBody(folly::IOBuf::copyBuffer(requestBody_));
    txn->sendEOM();
  }

  void setTransaction(HTTPTransaction*) noexcept override {}
  void detachTransaction() noexcept override {
    session_.reset();
    evb_.terminateLoopSoon();
  }

  void onHeadersComplete(std::unique_ptr<HTTPMessage> msg) noexcept override {}

  void onBody(std::unique_ptr<folly::IOBuf> chain) noexcept override {
    if (chain) {
      responseBody_.append(
          reinterpret_cast<const char*>(chain->data()), chain->length());
    }
  }

  void onEOM() noexcept override {
    session_->drain();
  }

  void onError(const HTTPException& error) noexcept override {
    LOG(ERROR) << "Error: " << error.what();
  }
  void onUpgrade(UpgradeProtocol) noexcept override {}
  void onTrailers(std::unique_ptr<HTTPHeaders>) noexcept override {}
  void onEgressPaused() noexcept override {}
  void onEgressResumed() noexcept override {}
};

class RestClient {
 public:
  RestClient(const std::string& url);
  void invoke_function(const std::string& request, std::string& response);

 private:
  URL url_;
  std::shared_ptr<HttpClient> httpClient_;
};

} // namespace facebook::velox::functions
