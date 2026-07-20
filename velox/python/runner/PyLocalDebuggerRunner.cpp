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

#include "velox/python/runner/PyLocalDebuggerRunner.h"

namespace facebook::velox::py {

namespace py = pybind11;

void PyLocalDebuggerRunner::setBreakpoint(const std::string& planNodeId) {
  breakpoints_[planNodeId] = nullptr;
}

void PyLocalDebuggerRunner::setHook(
    const std::string& planNodeId,
    pybind11::function callback) {
  // Capture the Python callback and outputPool in a C++ lambda.
  // The lambda will be called when the breakpoint is hit.
  auto pool = outputPool_;
  breakpoints_[planNodeId] = [callback = std::move(callback),
                              pool](const RowVectorPtr& vector) -> bool {
    // Acquire the GIL before calling Python code.
    py::gil_scoped_acquire acquire;

    // Create a PyVector from the RowVectorPtr.
    PyVector pyVector(vector, pool);

    // Call the Python function and get the result.
    py::object result = callback(pyVector);

    // Convert the result to bool.
    return result.cast<bool>();
  };
}

exec::CursorParameters PyLocalDebuggerRunner::createCursorParameters(
    int32_t maxDrivers) {
  return exec::CursorParameters{
      .planNode = planNode_,
      .maxDrivers = maxDrivers,
      .queryCtx = core::QueryCtx::Builder()
                      .queryConfig(core::QueryConfig(queryConfigs_))
                      .pool(rootPool_)
                      .build(),
      .outputPool = outputPool_,
      .serialExecution = true,
      .breakpoints = breakpoints_,
  };
}

} // namespace facebook::velox::py
