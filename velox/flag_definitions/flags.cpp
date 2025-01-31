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

#include <gflags/gflags.h>

#include "velox/common/config/GlobalConfig.h"

// Used in velox/common/memory/Memory.cpp

DEFINE_int32(
    velox_memory_num_shared_leaf_pools,
    32,
    "Number of shared leaf memory pools per process");

DEFINE_bool(
    velox_time_allocations,
    false,
    "Record time and volume for large allocation/free");

// Used in common/base/VeloxException.cpp
DEFINE_bool(
    velox_exception_user_stacktrace_enabled,
    false,
    "Enable the stacktrace for user type of VeloxException");

DEFINE_bool(
    velox_exception_system_stacktrace_enabled,
    true,
    "Enable the stacktrace for system type of VeloxException");

DEFINE_int32(
    velox_exception_user_stacktrace_rate_limit_ms,
    0, // effectively turns off rate-limiting
    "Min time interval in milliseconds between stack traces captured in"
    " user type of VeloxException; off when set to 0 (the default)");

DEFINE_int32(
    velox_exception_system_stacktrace_rate_limit_ms,
    0, // effectively turns off rate-limiting
    "Min time interval in milliseconds between stack traces captured in"
    " system type of VeloxException; off when set to 0 (the default)");

// Used in common/base/ProcessBase.cpp

DEFINE_bool(avx2, true, "Enables use of AVX2 when available");

DEFINE_bool(bmi2, true, "Enables use of BMI2 when available");

// Used in exec/Expr.cpp

DEFINE_string(
    velox_save_input_on_expression_any_failure_path,
    "",
    "Enable saving input vector and expression SQL on any failure during "
    "expression evaluation. Specifies the directory to use for storing the "
    "vectors and expression SQL strings.");

DEFINE_string(
    velox_save_input_on_expression_system_failure_path,
    "",
    "Enable saving input vector and expression SQL on system failure during "
    "expression evaluation. Specifies the directory to use for storing the "
    "vectors and expression SQL strings. This flag is ignored if "
    "velox_save_input_on_expression_any_failure_path is set.");

DEFINE_bool(
    force_eval_simplified,
    false,
    "Whether to overwrite queryCtx and force the "
    "use of simplified expression evaluation path.");

DEFINE_bool(
    velox_experimental_save_input_on_fatal_signal,
    false,
    "This is an experimental flag only to be used for debugging "
    "purposes. If set to true, serializes the input vector data and "
    "all the SQL expressions in the ExprSet that is currently "
    "executing, whenever a fatal signal is encountered. Enabling "
    "this flag makes the signal handler async signal unsafe, so it "
    "should only be used for debugging purposes. The vector and SQLs "
    "are serialized to files in directories specified by either "
    "'velox_save_input_on_expression_any_failure_path' or "
    "'velox_save_input_on_expression_system_failure_path'");

// TODO: deprecate this once all the memory leak issues have been fixed in
// existing meta internal use cases.
DEFINE_bool(
    velox_memory_leak_check_enabled,
    false,
    "If true, check fails on any memory leaks in memory pool and memory manager");

DEFINE_bool(
    velox_memory_pool_debug_enabled,
    false,
    "If true, 'MemoryPool' will be running in debug mode to track the allocation and free call sites to detect the source of memory leak for testing purpose");

// TODO: deprecate this after solves all the use cases that can cause
// significant performance regression by memory usage tracking.
DEFINE_bool(
    velox_enable_memory_usage_track_in_default_memory_pool,
    false,
    "If true, enable memory usage tracking in the default memory pool");

DEFINE_bool(
    velox_suppress_memory_capacity_exceeding_error_message,
    false,
    "If true, suppress the verbose error message in memory capacity exceeded "
    "exception. This is only used by test to control the test error output size");

DEFINE_bool(velox_memory_use_hugepages, true, "Use explicit huge pages");

DEFINE_bool(
    velox_memory_pool_capacity_transfer_across_tasks,
    false,
    "Whether allow to memory capacity transfer between memory pools from "
    "different tasks, which might happen in use case like Spark-Gluten");

DEFINE_int32(
    cache_prefetch_min_pct,
    80,
    "Minimum percentage of actual uses over references to a column for prefetching. No prefetch if > 100");

namespace facebook::velox {
void translateFlagsToGlobalConfig() {
  config::globalConfig.memoryNumSharedLeafPools =
      FLAGS_velox_memory_num_shared_leaf_pools;
  config::globalConfig.memoryLeakCheckEnabled =
      FLAGS_velox_memory_leak_check_enabled;
  config::globalConfig.memoryPoolDebugEnabled =
      FLAGS_velox_memory_pool_debug_enabled;
  config::globalConfig.enableMemoryUsageTrackInDefaultMemoryPool =
      FLAGS_velox_enable_memory_usage_track_in_default_memory_pool;
  config::globalConfig.timeAllocations = FLAGS_velox_time_allocations;
  config::globalConfig.memoryUseHugepages = FLAGS_velox_memory_use_hugepages;
  config::globalConfig.suppressMemoryCapacityExceedingErrorMessage =
      FLAGS_velox_suppress_memory_capacity_exceeding_error_message;
  config::globalConfig.memoryPoolCapacityTransferAcrossTasks =
      FLAGS_velox_memory_pool_capacity_transfer_across_tasks;
  config::globalConfig.exceptionSystemStacktraceEnabled =
      FLAGS_velox_exception_system_stacktrace_enabled;
  config::globalConfig.exceptionSystemStacktraceRateLimitMs =
      FLAGS_velox_exception_system_stacktrace_rate_limit_ms;
  config::globalConfig.exceptionUserStacktraceEnabled =
      FLAGS_velox_exception_user_stacktrace_enabled;
  config::globalConfig.exceptionUserStacktraceRateLimitMs =
      FLAGS_velox_exception_user_stacktrace_rate_limit_ms;
  config::globalConfig.forceEvalSimplified = FLAGS_force_eval_simplified;
  config::globalConfig.experimentalSaveInputOnFatalSignal =
      FLAGS_velox_experimental_save_input_on_fatal_signal;
  config::globalConfig.saveInputOnExpressionAnyFailurePath =
      FLAGS_velox_save_input_on_expression_any_failure_path;
  config::globalConfig.saveInputOnExpressionSystemFailurePath =
      FLAGS_velox_save_input_on_expression_system_failure_path;
}
} // namespace facebook::velox
