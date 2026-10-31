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

#include "velox/common/config/ConfigProvider.h"
#include "velox/connectors/common/FileConfig.h"

namespace facebook::velox::connector::file {

/// Exposes a connector's session properties with catalog configuration
/// overrides applied to their code defaults.
class FileConfigProvider : public config::ConfigProvider {
 public:
  explicit FileConfigProvider(
      std::vector<config::ConfigProperty> properties =
          FileConfig::registeredProperties(),
      std::string catalogName = {},
      const config::ConfigBase* catalogConfig = nullptr)
      : properties_(std::move(properties)),
        catalogName_(std::move(catalogName)),
        catalogConfig_(catalogConfig) {}

  std::vector<config::ConfigProperty> properties() const override;

  std::string normalize(std::string_view name, std::string_view value)
      const override;

 private:
  const std::vector<config::ConfigProperty> properties_;
  const std::string catalogName_;
  const config::ConfigBase* catalogConfig_;
};

} // namespace facebook::velox::connector::file
