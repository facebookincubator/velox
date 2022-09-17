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
#include <boost/program_options.hpp>

namespace facebook::velox::flags {

/// Configuration flags for Velox.
class VeloxFlags {
 public:
  // Initialization using command line arguments
  void init(int argc, const char* const argv[]);

  // Initialization using map.
  void init(std::map<std::string, std::string> options);

  bool avx2() {
    return vm_["avx2"].as<bool>();
  }

  bool bmi2() {
    return vm_["bmi2"].as<bool>();
  }

  // Used in velox/builder/SimpleVectorBuilder.cpp
  int32_t getMaxBlockValueSetLength() {
    return vm_["max_block_value_set_length"].as<int32_t>();
  }

  // Used in velox/common/memory/Memory.cpp
  int32_t getMemoryUsageAggregationIntervalMillis() {
    return vm_["memory_usage_aggregation_interval_millis"].as<int32_t>();
  }

  // Used in velox/common/memory/MappedMemory.cpp
  int32_t getVeloxMemoryPoolMb() {
    return vm_["velox_memory_pool_mb"].as<int32_t>();
  }

  bool getVeloxUseMalloc() {
    return vm_["velox_use_malloc"].as<bool>();
  }

  bool getVeloxTimeAllocations() {
    return vm_["velox_time_allocations"].as<bool>();
  }

  // Used in common/base/VeloxException.cpp
  bool getVeloxExceptionUserStackTraceEnabled() {
    return vm_["velox_exception_user_stacktrace_enabled"].as<bool>();
  }

  bool getVeloxExceptionSystemStackTraceEnabled() {
    return vm_["velox_exception_system_stacktrace_enabled"].as<bool>();
  }

  int32_t getVeloxExceptionUserStackTraceRateLimitMs() {
    return vm_["velox_exception_user_stacktrace_rate_limit_ms"].as<int32_t>();
  }

  int32_t getVeloxExceptionSystemStackTraceRateLimitMs() {
    return vm_["velox_exception_system_stacktrace_rate_limit_ms"].as<int32_t>();
  }

  friend VeloxFlags& getInstance();

 private:
  VeloxFlags();
  boost::program_options::options_description addOptions();
  boost::program_options::variables_map vm_;
};

VeloxFlags& getInstance();
} // namespace facebook::velox::flags
