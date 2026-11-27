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

#include <string_view>

namespace facebook::velox::exec {

/// Names of the runtime stats reported by PythonCall.
struct PythonCallStatNames {
  /// Number of distinct functions the worker runs.
  static constexpr std::string_view kNumFunctions = "pythonNumFunctions";

  /// Time the worker took to unpickle the functions and the modules they
  /// import.
  static constexpr std::string_view kUnpickleWallNanos =
      "pythonUnpickleWallNanos";

  /// Time the worker spent in the functions, per batch.
  static constexpr std::string_view kFunctionWallNanos =
      "pythonFunctionWallNanos";

  /// CPU time of the worker process, user and system.
  static constexpr std::string_view kWorkerCpuNanos = "pythonWorkerCpuNanos";

  /// Peak resident memory of the worker process.
  static constexpr std::string_view kWorkerPeakMemoryBytes =
      "pythonWorkerPeakMemoryBytes";
};

} // namespace facebook::velox::exec
