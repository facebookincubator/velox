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

#include <pybind11/pybind11.h>
#include <string>

#include "velox/python/runner/PyLocalRunner.h"

namespace facebook::velox::py {

/// A debugger-enabled extension of PyLocalRunner.
class PyLocalDebuggerRunner : public PyLocalRunner {
 public:
  using PyLocalRunner::PyLocalRunner;

  PyLocalDebuggerRunner(
      const PyPlanNode& pyPlanNode,
      const std::shared_ptr<memory::MemoryPool>& pool,
      const std::shared_ptr<folly::CPUThreadPoolExecutor>& executor)
      : PyLocalRunner(pyPlanNode, pool, executor) {}

  /// Sets a breakpoint at the specified plan node with no callback.
  /// The breakpoint will always stop execution.
  ///
  /// @param planNodeId The ID of the plan node where execution should pause.
  void setBreakpoint(const std::string& planNodeId);

  /// Sets a breakpoint at the specified plan node with a Python callback.
  ///
  /// The callback is invoked with a PyVector when the breakpoint is hit.
  /// If the callback returns True, execution stops and the vector is produced.
  /// If the callback returns False, execution continues without stopping.
  ///
  /// @param planNodeId The ID of the plan node where the hook is installed.
  /// @param callback Python function that takes a PyVector and returns bool.
  void setHook(const std::string& planNodeId, pybind11::function callback);

 protected:
  exec::CursorParameters createCursorParameters(int32_t maxDrivers) override;

 private:
  exec::CursorParameters::TBreakpointMap breakpoints_;
};

} // namespace facebook::velox::py
