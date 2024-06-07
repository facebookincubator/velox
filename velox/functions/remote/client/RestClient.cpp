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
#include <proxygen/lib/utils/URL.h>

using namespace facebook::velox::functions;

namespace facebook::velox::functions {

RestClient::RestClient(const std::string& url) : url_(url) {
  httpClient_ = std::make_shared<HttpClient>(url_);
};

void RestClient::invoke_function(
    const std::string& requestBody,
    std::string& responseBody) {
  httpClient_->send(requestBody);
  responseBody = httpClient_->getResponseBody();
  LOG(INFO) << responseBody;
};

} // namespace facebook::velox::functions
