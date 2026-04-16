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

#include "velox/common/config/ConfigProperty.h"

namespace facebook::velox::config {

/// Declares and normalizes configuration properties for a component.
/// A ConfigProvider is shared and stateless — it does not hold per-session
/// values. It knows only its own property names (e.g., "session_timezone"),
/// not how they are exposed to the user.
class ConfigProvider {
 public:
  virtual ~ConfigProvider() = default;

  /// Returns the list of supported configuration properties.
  virtual std::vector<ConfigProperty> properties() const = 0;

  /// Validates and returns the canonical form of the value. Called after
  /// type validation. Throws if the property name is unknown or the value
  /// is invalid.
  virtual std::string normalize(std::string_view name, std::string_view value)
      const = 0;
};

} // namespace facebook::velox::config
