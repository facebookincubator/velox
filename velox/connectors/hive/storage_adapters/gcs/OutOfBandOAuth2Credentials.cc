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

#include "OutOfBandOAuth2Credentials.h"

thread_local std::string OutOfBandOAuth2Credentials::current_path_;
thread_local std::optional<std::vector<std::string>> OutOfBandOAuth2Credentials::current_operation_;

OutOfBandOAuth2Credentials::OutOfBandOAuth2Credentials(std::shared_ptr<TokenProvider> token_provider)
    : token_provider_(std::move(token_provider)) {
}

void OutOfBandOAuth2Credentials::SetContext(const std::string& path, const std::optional<std::vector<std::string>>& operation) {
    current_path_ = path;
    current_operation_ = operation;
}

google::cloud::StatusOr<std::string> OutOfBandOAuth2Credentials::AuthorizationHeader() {
    auto token_pair = token_provider_->getAccessToken(current_path_, std::nullopt);
    std::string access_token = token_pair.first;

    if (access_token.empty()) {
        return google::cloud::Status(google::cloud::StatusCode::kPermissionDenied, "Failed to fetch access token");
    }
    return "Authorization: Bearer " + access_token;
}
