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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudfConfig : public velox::config::ConfigBase {
 public:
  /// Represents a single configuration entry with its metadata.
  struct CudfConfigEntry {
    std::string name; // Config key (e.g., "cudf.enabled")
    velox::TypePtr type; // Type of the configuration value
    std::string defaultValue; // Default value as string
  };

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  inline static const CudfConfigEntry kCudfEnabledEntry{
      "cudf.enabled",
      velox::BOOLEAN(),
      "true"};
  /// Enable debug printing.
  inline static const CudfConfigEntry kCudfDebugEnabledEntry{
      "cudf.debug_enabled",
      velox::BOOLEAN(),
      "false"};
  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  inline static const CudfConfigEntry kCudfMemoryResourceEntry{
      "cudf.memory_resource",
      velox::VARCHAR(),
      "async"};
  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  inline static const CudfConfigEntry kCudfMemoryPercentEntry{
      "cudf.memory_percent",
      velox::INTEGER(),
      "50"};
  /// Register all the functions with the functionNamePrefix.
  inline static const CudfConfigEntry kCudfFunctionNamePrefixEntry{
      "cudf.function_name_prefix",
      velox::VARCHAR(),
      ""};
  /// Enable AST in expression evaluation.
  inline static const CudfConfigEntry kCudfAstExpressionEnabledEntry{
      "cudf.ast_expression_enabled",
      velox::BOOLEAN(),
      "true"};
  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  inline static const CudfConfigEntry kCudfAstExpressionPriorityEntry{
      "cudf.ast_expression_priority",
      velox::INTEGER(),
      "100"};
  /// Enable JIT in expression evaluation.
  inline static const CudfConfigEntry kCudfJitExpressionEnabledEntry{
      "cudf.jit_expression_enabled",
      velox::BOOLEAN(),
      "true"};
  /// Priority of JIT expression.
  inline static const CudfConfigEntry kCudfJitExpressionPriorityEntry{
      "cudf.jit_expression_priority",
      velox::INTEGER(),
      "50"};
  /// Allow fallback to CPU operators if GPU operator replacement fails.
  inline static const CudfConfigEntry kCudfAllowCpuFallbackEntry{
      "cudf.allow_cpu_fallback",
      velox::BOOLEAN(),
      "true"};
  /// Whether to log a reason for falling back to Velox CPU execution.
  inline static const CudfConfigEntry kCudfLogFallbackEntry{
      "cudf.log_fallback",
      velox::BOOLEAN(),
      "false"};

  CudfConfig();

  /// Singleton CudfConfig instance.
  static CudfConfig& getInstance();

  /// Update config from a map with the above keys.
  void updateConfigs(std::unordered_map<std::string, std::string>&&);

  /// Retrieve config value by key, fails if config does not exist.
  template <typename T>
  T get(const std::string& key) const {
    auto value = velox::config::ConfigBase::get<T>(key);
    VELOX_CHECK(value.has_value(), "{} is not a valid CudfConfig.", key);
    return value.value();
  }

  /// Retrieve all configuration entries.
  const std::vector<CudfConfigEntry>& getConfigs() const;

 private:
  /// Vector of all registered configuration entries.
  std::vector<CudfConfigEntry> configEntries_;

  /// Set of valid configuration keys for fast lookup during validation.
  std::unordered_set<std::string> configKeys_;
};

} // namespace facebook::velox::cudf_velox
