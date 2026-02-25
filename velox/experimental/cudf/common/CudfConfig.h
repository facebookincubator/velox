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

#include "velox/common/base/Exceptions.h"
#include "velox/common/config/Config.h"
#include "velox/type/Type.h"

#include <folly/Conv.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudfQueryConfig {
 public:
  /// Default constructor.
  CudfQueryConfig();

  /// Constructor that initializes config from a map of values.
  explicit CudfQueryConfig(
      std::unordered_map<std::string, std::string>&& values);

  /// Constructor with ConfigTag for compatibility.
  struct ConfigTag {};
  explicit CudfQueryConfig(
      ConfigTag /*tag*/,
      std::shared_ptr<const velox::config::IConfig> config);

  /// Represents a single configuration entry with its metadata.
  struct CudfQueryConfigEntry {
    /// Config name.
    std::string name;
    /// Data type of config, required by Presto sidecar for mapping as session
    /// property.
    velox::TypePtr type;
    /// Default value of config, required by Presto sidecar for session property
    /// maintenance.
    std::string defaultValue;
  };

  /// TODO(ps): Deprecate this from query config and move to system config.
  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  inline static const CudfQueryConfigEntry kCudfEnabledEntry{
      "cudf.enabled",
      velox::BOOLEAN(),
      "true"};

  /// Enable debug printing.
  inline static const CudfQueryConfigEntry kCudfDebugEnabledEntry{
      "cudf.debug_enabled",
      velox::BOOLEAN(),
      "false"};
  /// Whether to log a reason for falling back to Velox CPU execution.
  inline static const CudfQueryConfigEntry kCudfLogFallbackEntry{
      "cudf.log_fallback",
      velox::BOOLEAN(),
      "false"};
  /// Number of TopN batches to accumulate before merging.
  inline static const CudfQueryConfigEntry kCudfTopNBatchSizeEntry{
      "cudf.topn_batch_size",
      velox::INTEGER(),
      "4"};

  /// Singleton CudfQueryConfig instance.
  static CudfQueryConfig& getInstance();

  /// Retrieve all configuration entries.
  const std::vector<CudfQueryConfigEntry>& getConfigs() const;

  /// Individual getter methods for each configuration.
  bool debugEnabled() const {
    return get<bool>(kCudfDebugEnabledEntry);
  }

  bool logFallback() const {
    return get<bool>(kCudfLogFallbackEntry);
  }

  int32_t topNBatchSize() const {
    return get<int32_t>(kCudfTopNBatchSizeEntry);
  }

  /// Retrieve configuration value from registry with fallback to default.
  /// Used by Presto sidecar to retrieve default values with config entry list.
  template <typename T>
  T get(const CudfQueryConfigEntry& entry) const {
    T defaultValue = folly::to<T>(entry.defaultValue);
    return config_->get<T>(entry.name, defaultValue);
  }

  /// Retrieve the underlying config object.
  const std::shared_ptr<const velox::config::IConfig>& config() const {
    return config_;
  }

 private:
  /// List of all registered configuration entries, maintained for retrieval
  /// from Presto sidecar to allow for configuring as session properties.
  std::vector<CudfQueryConfigEntry> configEntries_;

  /// Optional validations for configs.
  void validateConfigs();

  /// Underlying config storage.
  std::shared_ptr<const velox::config::IConfig> config_;
};

class CudfSystemConfig : public velox::config::ConfigBase {
 public:
  /// Default constructor.
  CudfSystemConfig();

  /// Constructor that initializes config from a map of values.
  explicit CudfSystemConfig(
      std::unordered_map<std::string, std::string>&& values);

  /// Enable cudf by default.
  /// Clients must set the configs below before invoking registerCudf().
  static constexpr const char* kCudfEnabled = "cudf-enabled";
  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  static constexpr const char* kCudfMemoryResource = "cudf-memory-resource";
  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  static constexpr const char* kCudfMemoryPercent = "cudf-memory-percent";
  /// Register all the functions with the functionNamePrefix.
  static constexpr const char* kCudfFunctionNamePrefix =
      "cudf-function-name-prefix";
  /// Enable AST in expression evaluation.
  static constexpr const char* kCudfAstExpressionEnabled =
      "cudf-ast-expression-enabled";
  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  static constexpr const char* kCudfAstExpressionPriority =
      "cudf-ast-expression-priority";
  /// Enable JIT in expression evaluation.
  static constexpr const char* kCudfJitExpressionEnabled =
      "cudf-jit-expression-enabled";
  /// Priority of JIT expression.
  static constexpr const char* kCudfJitExpressionPriority =
      "cudf-jit-expression-priority";
  /// Allow fallback to CPU operators if GPU operator replacement fails.
  static constexpr const char* kCudfAllowCpuFallback =
      "cudf-allow-cpu-fallback";

  /// Singleton CudfSystemConfig instance.
  static CudfSystemConfig& getInstance();

  /// Update config from a map. Supports both '-' and '_' delimiters.
  /// Intelligently processes keys by checking dash names first, then legacy
  /// underscore names.
  void updateConfigs(std::unordered_map<std::string, std::string>&&);

  /// Individual getter methods for each configuration.
  bool cudfEnabled() const {
    return get<bool>(kCudfEnabled, true);
  }

  std::string memoryResource() const {
    return get<std::string>(kCudfMemoryResource, "async");
  }

  int32_t memoryPercent() const {
    return get<int32_t>(kCudfMemoryPercent, 50);
  }

  std::string functionNamePrefix() const {
    return get<std::string>(kCudfFunctionNamePrefix, "");
  }

  bool astExpressionEnabled() const {
    return get<bool>(kCudfAstExpressionEnabled, true);
  }

  int32_t astExpressionPriority() const {
    return get<int32_t>(kCudfAstExpressionPriority, 100);
  }

  bool jitExpressionEnabled() const {
    return get<bool>(kCudfJitExpressionEnabled, true);
  }

  int32_t jitExpressionPriority() const {
    return get<int32_t>(kCudfJitExpressionPriority, 50);
  }

  bool allowCpuFallback() const {
    return get<bool>(kCudfAllowCpuFallback, true);
  }

  /// Retrieve configuration value by key with fallback to default.
  //  template <typename T>
  //  T get(const std::string& key, const T& defaultValue) const {
  //    return velox::config::IConfig::get<T>(key, defaultValue);
  //  }

 private:
  void validateConfigs();

  std::shared_ptr<const IConfig> config_;
};

} // namespace facebook::velox::cudf_velox
