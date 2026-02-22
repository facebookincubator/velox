/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <functional>
#include <memory>
#include <string>

namespace facebook::velox::connector::hive {
class HiveConfig;
}

namespace facebook::velox::filesystems {
class GcsOAuthCredentialsProvider;

// Register the GCS filesystem.
void registerGcsFileSystem();

using GcsOAuthCredentialsProviderFactory =
    std::function<std::shared_ptr<GcsOAuthCredentialsProvider>(
        const std::shared_ptr<connector::hive::HiveConfig>& hiveConfig)>;

void registerGcsOAuthCredentialsProvider(
    const std::string& providerName,
    const GcsOAuthCredentialsProviderFactory& factory);

} // namespace facebook::velox::filesystems
