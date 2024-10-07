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

#ifndef OUT_OF_BAND_OAUTH2_CREDENTIALS_H
#define OUT_OF_BAND_OAUTH2_CREDENTIALS_H

#include "google/cloud/storage/oauth2/credentials.h"
#include <memory>
#include <string>
#include "TokenProvider.h"

class OutOfBandOAuth2Credentials : public google::cloud::storage::oauth2::Credentials {
public:
    OutOfBandOAuth2Credentials(std::shared_ptr<TokenProvider> token_provider);
    virtual ~OutOfBandOAuth2Credentials() = default;

    void SetContext(const std::string& path, const std::optional<std::vector<std::string>>& operation);

    google::cloud::StatusOr<std::string> AuthorizationHeader() override;

private:
    std::shared_ptr<TokenProvider> token_provider_;
    static thread_local std::string current_path_;
    static thread_local std::optional<std::vector<std::string>> current_operation_;
};

#endif
