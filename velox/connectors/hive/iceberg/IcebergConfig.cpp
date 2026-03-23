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

#include "velox/connectors/hive/iceberg/IcebergConfig.h"

#include "velox/common/config/Config.h"

namespace facebook::velox::connector::hive::iceberg {

IcebergConfig::IcebergConfig(
    const std::shared_ptr<const config::ConfigBase>& config)
    : config_(config) {
  VELOX_CHECK_NOT_NULL(
      config_, "Config is null for IcebergConfig initialization");
}

std::string IcebergConfig::functionPrefix() const {
  return config_->get<std::string>(
      kFunctionPrefixConfig, kDefaultFunctionPrefix);
}

} // namespace facebook::velox::connector::hive::iceberg
