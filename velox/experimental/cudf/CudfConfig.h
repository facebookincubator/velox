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

#include <string>
#include <unordered_map>
#include <mutex>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();
  CudfConfig& operator=(CudfConfig const&) = delete;
  CudfConfig(CudfConfig const&) = delete;
  
  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  int32_t memoryPercent{50};

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;
  
  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Force replacement of operators. Throws an error if a replacement fails.
  bool allowCpuFallback{true};

  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};

  CudfConfig() {}

 private:

  bool isEnabled() {
    std::lock_guard<std::mutex> lock(kMutex);
    return enabled;
  }

  void setEnabled(bool val) {
    std::lock_guard<std::mutex> lock(kMutex);
    enabled = val;
  }

  bool isDebugEnabled() {
    std::lock_guard<std::mutex> lock(kMutex);
    return debugEnabled;
  }

  void setDebugEnabled(bool val) {
    std::lock_guard<std::mutex> lock(kMutex);
    debugEnabled = val;
  }

  bool isCpuFallbackAllowed() {
    std::lock_guard<std::mutex> lock(kMutex);
    return allowCpuFallback;
  }

  void setAllowCpuFallback(bool val) {
    std::lock_guard<std::mutex> lock(kMutex);
    allowCpuFallback = val;
  }
  

  std::mutex kMutex; // Mutex for thread safety
};

} // namespace facebook::velox::cudf_velox
