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

#include <memory>
#include <string>

#include "velox/common/config/Config.h"

namespace facebook::velox::connector::hive::iceberg {

/// Iceberg-specific connector configuration wrapper.
/// Provides accessors for Iceberg-only settings while sharing the same
/// underlying ConfigBase with HiveConfig.
class IcebergConfig {
 public:
  /// Iceberg function prefix.
  static constexpr const char* kFunctionPrefixConfig =
      "presto.iceberg-namespace";

  /// Default prefix used to register Iceberg transform functions when no
  /// connector config override is provided.
  static constexpr const char* kDefaultFunctionPrefix = "$internal$.iceberg.";

  explicit IcebergConfig(
      const std::shared_ptr<const config::ConfigBase>& config);

  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

  std::string functionPrefix() const;

 private:
  const std::shared_ptr<const config::ConfigBase> config_;
};

using IcebergConfigPtr = std::shared_ptr<const IcebergConfig>;

} // namespace facebook::velox::connector::hive::iceberg
