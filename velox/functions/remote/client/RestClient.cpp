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

#include <cpr/cpr.h>
#include <folly/io/IOBufQueue.h>

#include "velox/common/base/Exceptions.h"

using namespace folly;
namespace facebook::velox::functions {

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

  std::string contentType;
  switch (serdeFormat) {
    case remote::PageFormat::SPARK_UNSAFE_ROW:
      contentType = "application/X-spark-unsafe-row";
      break;
    case remote::PageFormat::PRESTO_PAGE:
    default:
      contentType = "application/X-presto-pages";
      break;
  }

  cpr::Response response = cpr::Post(
      cpr::Url{fullUrl},
      cpr::Header{{"Content-Type", contentType}, {"Accept", contentType}},
      cpr::Body{requestBody});

  if (response.error) {
    VELOX_FAIL(fmt::format(
          "Error communicating with server: {} URL: {}",
          response.error.message,
          fullUrl));
  }

  auto outputBuf = IOBuf::copyBuffer(response.text);
  return outputBuf;
}

std::unique_ptr<HttpClient> getRestClient() {
  return std::make_unique<RestClient>();
}

} // namespace facebook::velox::functions
