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

namespace facebook::velox::connector::hive {

/// Exposes Hive connector session-overridable properties.
class HiveConfigProvider : public config::ConfigProvider {
 public:
  /// Returns all session-overridable Hive connector properties.
  std::vector<config::ConfigProperty> properties() const override;

  /// Validates and normalizes a property value.
  std::string normalize(std::string_view name, std::string_view value)
      const override;
};

} // namespace facebook::velox::connector::hive
