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

#include <folly/io/IOBuf.h>
#include <memory>
#include <string>

namespace facebook::velox::functions {

/// @brief Abstract interface for an HTTP client.
/// Provides a method to invoke a function by sending an HTTP request
/// and receiving a response, both in Presto's serialized wire format.
class HttpClient {
 public:
  virtual ~HttpClient() = default;

  /// @brief Invokes a function over HTTP.
  /// @param url The endpoint URL to send the request to.
  /// @param requestPayload The request payload in Presto's serialized wire
  /// format.
  /// @return A unique pointer to the response payload in Presto's serialized
  /// wire format.
  virtual std::unique_ptr<folly::IOBuf> invokeFunction(
      const std::string& url,
      std::unique_ptr<folly::IOBuf> requestPayload) = 0;
};

/// @brief Concrete implementation of HttpClient using REST.
/// Handles HTTP communication by sending requests and receiving responses
/// using RESTful APIs with payloads in Presto's serialized wire format.
class RestClient : public HttpClient {
 public:
  /// @brief Invokes a function over HTTP using cpr.
  /// Sends an HTTP POST request to the specified URL with the request payload
  /// and receives the response payload. Both payloads are in Presto's
  /// serialized wire format.
  /// @param url The endpoint URL to send the request to.
  /// @param requestPayload The request payload in Presto's serialized wire
  /// format.
  /// @return A unique pointer to the response payload in Presto's serialized
  /// wire format.
  /// @throws VeloxException if there is an error initializing cpr or during
  /// the request.
  std::unique_ptr<folly::IOBuf> invokeFunction(
      const std::string& url,
      std::unique_ptr<folly::IOBuf> requestPayload) override;
};

/// @brief Factory function to create an instance of RestClient.
/// @return A unique pointer to an HttpClient implementation.
std::unique_ptr<HttpClient> getRestClient();

} // namespace facebook::velox::functions
