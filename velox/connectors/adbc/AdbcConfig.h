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

#include <optional>
#include <string>
#include <unordered_map>

#include "velox/common/config/Config.h"

namespace facebook::velox::connector::adbc {

/// Typed accessors for ADBC connector configuration.
class AdbcConfig {
 public:
  /// Driver shared library name or path passed to the ADBC driver manager,
  /// e.g. 'adbc_driver_mysql' or '/path/to/libadbc_driver_mysql.so'. Required
  /// unless the connector is constructed with an in-process driver init
  /// function.
  static constexpr const char* kDriver = "adbc.driver";

  /// Driver entrypoint symbol. Optional; the driver manager derives a default
  /// from the driver library name.
  static constexpr const char* kEntrypoint = "adbc.entrypoint";

  /// Quote string placed around column names in generated SQL. Defaults to
  /// '"'; set to '`' for MySQL.
  static constexpr const char* kIdentifierQuote = "adbc.identifier-quote";

  /// Prefix for options forwarded to AdbcDatabaseSetOption with the prefix
  /// stripped, e.g. 'adbc.option.uri', 'adbc.option.username'. This keeps the
  /// connector generic: driver-specific options need no Velox changes.
  static constexpr const char* kDatabaseOptionPrefix = "adbc.option.";

  explicit AdbcConfig(std::shared_ptr<const config::ConfigBase> config);

  /// Returns the driver library name or path, if configured.
  std::optional<std::string> driver() const;

  /// Returns the driver entrypoint symbol, if configured.
  std::optional<std::string> entrypoint() const;

  /// Returns the identifier quote string for generated SQL.
  std::string identifierQuote() const;

  /// Returns the options to forward to AdbcDatabaseSetOption, keyed by option
  /// name with the 'adbc.option.' prefix stripped.
  std::unordered_map<std::string, std::string> databaseOptions() const;

 private:
  const std::shared_ptr<const config::ConfigBase> config_;
};

} // namespace facebook::velox::connector::adbc
