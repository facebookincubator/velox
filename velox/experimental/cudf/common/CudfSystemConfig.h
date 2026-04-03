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

#include "velox/common/config/Config.h"

#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

class CudfSystemConfig : public velox::config::ConfigBase {
 public:
  /// Default constructor.
  CudfSystemConfig();

  /// Constructor that initializes config from a map of values.
  explicit CudfSystemConfig(
      std::unordered_map<std::string, std::string>&& values);

  /// Enable cudf by default.
  /// Clients must set the configs below before invoking registerCudf().
  static constexpr const char* kCudfEnabled = "cudf.enabled";

  /// Enable debug logging.
  static constexpr const char* kCudfDebugEnabled = "cudf.debug-enabled";

  /// Whether to log a reason for fallback to Velox CPU execution.
  static constexpr const char* kCudfLogFallback = "cudf.log-fallback";

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  static constexpr const char* kCudfMemoryResource = "cudf.memory-resource";

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  static constexpr const char* kCudfMemoryPercent = "cudf.memory-percent";

  /// Memory resource for output vectors.
  static constexpr const char* kCudfOutputMemoryResource = "cudf.output-mr";

  /// Register all the functions with the functionNamePrefix.
  static constexpr const char* kCudfFunctionNamePrefix =
      "cudf.function-name-prefix";

  /// Function engine used for registration and signature selection.
  static constexpr const char* kCudfFunctionEngine = "cudf.function-engine";

  /// Enable AST in expression evaluation.
  static constexpr const char* kCudfAstExpressionEnabled =
      "cudf.ast-expression-enabled";

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  static constexpr const char* kCudfAstExpressionPriority =
      "cudf.ast-expression-priority";

  /// Enable JIT in expression evaluation.
  static constexpr const char* kCudfJitExpressionEnabled =
      "cudf.jit-expression-enabled";

  /// Priority of JIT expression.
  static constexpr const char* kCudfJitExpressionPriority =
      "cudf.jit-expression-priority";

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  static constexpr const char* kCudfAllowCpuFallback =
      "cudf.allow-cpu-fallback";

  /// Whether to insert CudfBatchConcat operators before supported operators.
  static constexpr const char* kCudfConcatOptimizationEnabled =
      "cudf.concat-optimization-enabled";

  /// Minimum rows to accumulate before GPU-side concatenation.
  static constexpr const char* kCudfBatchSizeMinThreshold =
      "cudf.batch-size-min-threshold";

  /// Maximum rows allowed in a concatenated batch.
  static constexpr const char* kCudfBatchSizeMaxThreshold =
      "cudf.batch-size-max-threshold";

  /// Number of TopN batches to accumulate before merging.
  static constexpr const char* kCudfTopNBatchSize = "cudf.topn-batch-size";

  /// Singleton CudfSystemConfig instance.
  static CudfSystemConfig& getInstance();

  /// Update config from a map. Supports modern dash-delimited keys and legacy
  /// underscore-delimited keys.
  ///
  /// Example: "cudf.allow_cpu_fallback" and "cudf.allow-cpu-fallback" are
  /// normalized to canonical internal key "cudf.allow-cpu-fallback".
  void updateConfigs(std::unordered_map<std::string, std::string>&&);

  /// Individual getter methods for each configuration.
  bool cudfEnabled() const {
    return get<bool>(kCudfEnabled, true);
  }

  bool debugEnabled() const {
    return get<bool>(kCudfDebugEnabled, false);
  }

  bool logFallback() const {
    return get<bool>(kCudfLogFallback, false);
  }

  std::string memoryResource() const {
    return get<std::string>(kCudfMemoryResource, "async");
  }

  int32_t memoryPercent() const {
    return get<int32_t>(kCudfMemoryPercent, 50);
  }

  std::string outputMemoryResource() const {
    return get<std::string>(kCudfOutputMemoryResource, "");
  }

  std::string functionNamePrefix() const {
    return get<std::string>(kCudfFunctionNamePrefix, "");
  }

  std::string functionEngine() const {
    return get<std::string>(kCudfFunctionEngine, "presto");
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

  bool concatOptimizationEnabled() const {
    return get<bool>(kCudfConcatOptimizationEnabled, false);
  }

  int32_t batchSizeMinThreshold() const {
    return get<int32_t>(kCudfBatchSizeMinThreshold, 100000);
  }

  std::optional<int32_t> batchSizeMaxThreshold() const {
    return get<int32_t>(kCudfBatchSizeMaxThreshold);
  }

  int32_t topNBatchSize() const {
    return get<int32_t>(kCudfTopNBatchSize, 5);
  }

 private:
  void validateConfigs();
};

} // namespace facebook::velox::cudf_velox
