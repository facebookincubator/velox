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

#include "velox/functions/remote/client/RestClient.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <folly/io/IOBufQueue.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/remote/utils/ContentTypes.h"

using namespace folly;
namespace facebook::velox::functions {
namespace {
inline std::string getContentType(remote::PageFormat serdeFormat) {
  return serdeFormat == remote::PageFormat::SPARK_UNSAFE_ROW
      ? remote::CONTENT_TYPE_SPARK_UNSAFE_ROW
      : remote::CONTENT_TYPE_PRESTO_PAGE;
}

std::pair<std::string, std::string> parseHostAndPort(
    const std::string& fullUrl) {
  auto pos = fullUrl.find("//");
  std::string url =
      (pos != std::string::npos) ? fullUrl.substr(pos + 2) : fullUrl;
  pos = url.find("/");
  if (pos != std::string::npos) {
    url = url.substr(0, pos);
  }

  pos = url.find(":");
  if (pos != std::string::npos) {
    return {url.substr(0, pos), url.substr(pos + 1)};
  }
  VELOX_FAIL("Invalid URL: {}", fullUrl);
}

} // namespace

std::unique_ptr<IOBuf> RestClient::invokeFunction(
    const std::string& fullUrl,
    std::unique_ptr<IOBuf> requestPayload,
    remote::PageFormat serdeFormat) {
  IOBufQueue inputBufQueue(IOBufQueue::cacheChainLength());
  inputBufQueue.append(std::move(requestPayload));

  std::string requestBody;
  for (auto range : *inputBufQueue.front()) {
    requestBody.append(
        reinterpret_cast<const char*>(range.data()), range.size());
  }

  std::string contentType = getContentType(serdeFormat);

  try {
    boost::asio::io_context ioc;
    boost::asio::ip::tcp::resolver resolver(ioc);

    auto [host, port] = parseHostAndPort(fullUrl);
    auto const results = resolver.resolve(host, port);

    boost::beast::tcp_stream stream(ioc);
    stream.connect(results);

    boost::beast::flat_buffer buffer;
    boost::beast::http::request<boost::beast::http::string_body> req;
    req.method(boost::beast::http::verb::post);
    req.target(fullUrl);
    req.set(boost::beast::http::field::content_type, contentType);
    req.set(boost::beast::http::field::accept, contentType);
    req.body() = requestBody;
    req.prepare_payload();

    boost::beast::http::response<boost::beast::http::string_body> res;
    boost::beast::http::write(stream, req);
    boost::beast::http::read(stream, buffer, res);

    if (res.result_int() < 200 || res.result_int() >= 300) {
      VELOX_FAIL(
          fmt::format(
              "Server responded with status {}. Message: '{}'. URL: {}",
              res.result_int(),
              res.body(),
              fullUrl));
    }

    stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both);

    return IOBuf::copyBuffer(res.body());
  } catch (const std::exception& ex) {
    VELOX_FAIL(
        fmt::format(
            "Error communicating with server: {} URL: {}", ex.what(), fullUrl));
  }
}

std::unique_ptr<RestClient> getRestClient() {
  return std::make_unique<RestClient>();
}

} // namespace facebook::velox::functions
