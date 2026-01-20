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

#include <pybind11/embed.h>

#include "velox/core/PlanNode.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/python/plan_builder/PyPlanBuilder.h"
#include "velox/python/type/PyType.h"
#include "velox/python/vector/PyVector.h"

namespace facebook::velox::py {

/// Iterator class that extracts PyVectors from a TaskCursor created by a Runner
/// such as PyLocalRunner, providing an iterable API for Python.
class PyTaskIterator {
 public:
  PyTaskIterator(
      std::shared_ptr<memory::MemoryPool> pool,
      std::shared_ptr<exec::TaskCursor> cursor);

  PyTaskIterator& iter() {
    return *this;
  }

  PyVector next();

  std::optional<PyVector> current() const {
    if (!vector_) {
      return std::nullopt;
    }
    return PyVector{vector_, outputPool_};
  }

 private:
  std::shared_ptr<memory::MemoryPool> outputPool_;
  std::shared_ptr<exec::TaskCursor> cursor_;

  // Stores the current vector being iterated on.
  RowVectorPtr vector_{nullptr};
};

/// A C++ wrapper to allow Python clients to execute plans using TaskCursor.
///
/// @param pyPlanNode The plan to be executed (created using
/// pyvelox.plan_builder).
/// @param pool The memory pool to pass to the task.
/// @param executor The executor that will be used by drivers.
class PyLocalRunner {
 public:
  PyLocalRunner(
      const PyPlanNode& pyPlanNode,
      const std::shared_ptr<memory::MemoryPool>& pool,
      const std::shared_ptr<folly::CPUThreadPoolExecutor>& executor);

  /// Add a split to scan an entire file.
  ///
  /// @param pyFile The Python File object describin the file path and format.
  /// @param planId The plan node ID of the scan.
  /// @param connectorId The connector used by the scan.
  void addFileSplit(
      const PyFile& pyFile,
      const std::string& planId,
      const std::string& connectorId);

  /// Add a query configuration parameter. These values are passed to the Velox
  /// Task through a query context object.
  ///
  /// @param configName The name (key) of the configuration parameter.
  /// @param configValue The configuration value.
  void addQueryConfig(
      const std::string& configName,
      const std::string& configValue);

  /// Execute the task and returns an iterable to the output vectors. It's the
  /// caller's responsibility to ensure that the iterator will outlive the
  /// runner. This is guaranteed in the Python API by using pybind's
  /// py::keep_alive<>.
  ///
  /// Can be executed many times; it returns a new TaskIterator object with its
  /// own task cursor underneath. Consumes all splits whenever execute() is
  /// called.
  ///
  /// @param maxDrivers Maximum number of drivers to use when executing the
  /// plan.
  PyTaskIterator execute(int32_t maxDrivers = 1);

  /// Prints a descriptive debug message containing plan and execution stats.
  /// If the task hasn't finished, will print the plan with the current stats.
  std::string printPlanWithStats() const;

 private:
  // Memory pools and thread pool to be used by queryCtx.
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> outputPool_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;

  // The plan node to be executed (created using pyvelox.plan_builder).
  core::PlanNodePtr planNode_;

  // The task cursor that executed the Velox Task.
  std::shared_ptr<exec::TaskCursor> cursor_;

  // Pointer to the list of splits to be added to the task.
  TScanFiles scanFiles_;

  // Query configs to be passed to the task.
  TQueryConfigs queryConfigs_;
};

/// To avoid desctruction order issues during shutdown, this function will
/// iterate over any pending tasks created by this module and wait for their
/// task and drivers to finish.
void drainAllTasks();

} // namespace facebook::velox::py
