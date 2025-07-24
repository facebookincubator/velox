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

#include <functional>
#include <memory>
#include <string>

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::filesystems {

class AzureClientProvider;
class AbfsPath;

using AzureClientProviderFactory =
    std::function<std::unique_ptr<AzureClientProvider>(
        const std::shared_ptr<AbfsPath>& path,
        const config::ConfigBase& config)>;

// Register the ABFS filesystem.
void registerAbfsFileSystem();

// Register a factory for creating AzureClientProvider instances.
void registerAzureClientProviderFactory(
    const std::string& account,
    const AzureClientProviderFactory& factory);

} // namespace facebook::velox::filesystems
