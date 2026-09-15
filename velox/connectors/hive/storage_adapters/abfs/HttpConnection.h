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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::filesystems {

/// Names the HTTP methods supported by the transport contract.
enum class HttpMethod { kGet, kHead, kPost };

/// Preserves the version advertised by an HTTP response.
struct HttpVersion {
  /// Holds the major protocol version.
  uint16_t major{1};
  /// Holds the minor protocol version.
  uint16_t minor{1};
};

/// Preserves header order and duplicate field occurrences.
using HttpHeaders = std::vector<std::pair<std::string, std::string>>;

/// Sets independent bounds for each HTTP message resource.
struct HttpLimits {
  /// Bounds the status line.
  size_t maxStatusLineBytes{8'192};
  /// Bounds all response headers.
  size_t maxHeaderBytes{64 * 1'024};
  /// Bounds a request body.
  size_t maxRequestBodyBytes{64 * 1'024 * 1'024};
  /// Bounds a buffered response body.
  size_t maxBufferedResponseBodyBytes{64 * 1'024 * 1'024};
  /// Bounds decoded ingress retained while reading.
  size_t maxIngressBytes{64 * 1'024};
  /// Bounds informational responses before the final response.
  size_t maxInformationalResponses{8};
};

/// Sets independent deadlines for HTTP transactions and pooled connections.
struct HttpTimeouts {
  /// Bounds request serialization and transport writes.
  std::chrono::milliseconds write{std::chrono::seconds(30)};
  /// Bounds receipt of the first response byte and final response headers.
  std::chrono::milliseconds firstByteAndHeaders{std::chrono::seconds(30)};
  /// Bounds each body read while bytes are still arriving.
  std::chrono::milliseconds bodyIdle{std::chrono::seconds(30)};
  /// Bounds the complete request and response transaction.
  std::chrono::milliseconds total{std::chrono::minutes(5)};
  /// Bounds how long a request waits to acquire a pooled connection.
  std::chrono::milliseconds connectionAcquire{std::chrono::seconds(30)};
  /// Bounds how long a reusable connection remains idle in its pool.
  std::chrono::milliseconds connectionIdle{std::chrono::minutes(1)};
};

/// Selects whether the parser must consume a response body.
enum class HttpResponseBodyMode { kParse, kSkip };

/// Describes a request before it is serialized on a transport.
struct HttpRequest {
  /// Identifies the request method.
  HttpMethod method{HttpMethod::kGet};
  /// Holds the request target.
  std::string target;
  /// Preserves ordered duplicate request fields.
  HttpHeaders headers;
  /// Holds the request body exactly as supplied.
  std::string body;
  /// Selects response-body parsing before the response is parsed.
  HttpResponseBodyMode responseBodyMode{HttpResponseBodyMode::kParse};

  /// Builds a request whose response body must be skipped.
  static HttpRequest head(std::string requestTarget) {
    HttpRequest request;
    request.method = HttpMethod::kHead;
    request.target = std::move(requestTarget);
    request.responseBodyMode = HttpResponseBodyMode::kSkip;
    return request;
  }
};

/// Describes the response head without discarding duplicate fields.
struct HttpResponseHead {
  /// Preserves the actual response protocol version.
  HttpVersion version;
  /// Holds the final response status.
  int statusCode{0};
  /// Holds the response reason phrase.
  std::string reason;
  /// Preserves ordered duplicate response fields.
  HttpHeaders headers;
  /// Holds the known decoded body length, when framing provides one.
  std::optional<uint64_t> contentLength;
  /// Indicates whether response framing and headers permit reuse.
  bool reusable{false};
  /// Counts informational responses discarded before the final response.
  size_t informationalResponseCount{0};
};

/// Names the outcome used when releasing an HTTP transaction.
enum class HttpTransactionOutcome {
  kReusable,
  kClosed,
  kAbandoned,
  kFailed,
  kTimedOut,
};

/// Releases the connection that owns a completed body transaction.
using HttpTransactionRelease =
    std::function<void(HttpTransactionOutcome outcome)>;

/// Owns a response body and synchronously pulls decoded bytes from it.
class HttpBodyTransaction {
 public:
  /// Destroys the body transaction.
  virtual ~HttpBodyTransaction() = default;

  /// Reads up to size bytes, waiting up to the supplied timeout.
  virtual size_t
  read(uint8_t* buffer, size_t size, std::chrono::milliseconds timeout) = 0;

  /// Reports whether the complete response body has been consumed.
  virtual bool complete() const noexcept = 0;

  /// Discards the body and prevents connection reuse.
  virtual void abandon() noexcept = 0;
};

/// Couples response metadata with its owned pull-based body transaction.
struct HttpResponseTransaction {
  /// Holds the response metadata returned after headers arrive.
  HttpResponseHead head;
  /// Owns the response body transaction.
  std::unique_ptr<HttpBodyTransaction> body;
};

/// Sends requests and owns one protocol-neutral HTTP connection transaction.
class HttpConnection {
 public:
  /// Destroys the HTTP connection interface.
  virtual ~HttpConnection() = default;

  /// Sends a request and returns after response headers are available.
  virtual HttpResponseTransaction send(
      const HttpRequest& request,
      const HttpLimits& limits,
      const HttpTimeouts& timeouts,
      HttpTransactionRelease release) = 0;
};

} // namespace facebook::velox::filesystems
