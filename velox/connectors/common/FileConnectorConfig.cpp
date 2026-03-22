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

#include "velox/connectors/common/FileConnectorConfig.h"
#include "velox/common/config/Config.h"

namespace facebook::velox::connector {

FileConnectorConfig::FileConnectorConfig(
    std::shared_ptr<const config::ConfigBase> config)
    : config_(std::move(config)) {}

int32_t FileConnectorConfig::numCacheFileHandles() const {
  return config_->get<int32_t>(kNumCacheFileHandles, 20'000);
}

bool FileConnectorConfig::isFileHandleCacheEnabled() const {
  return config_->get<bool>(kEnableFileHandleCache, true);
}

uint64_t FileConnectorConfig::fileHandleExpirationDurationMs() const {
  return config_->get<uint64_t>(kFileHandleExpirationDurationMs, 0);
}

} // namespace facebook::velox::connector
