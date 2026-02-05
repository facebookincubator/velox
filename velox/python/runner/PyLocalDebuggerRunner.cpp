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

void PyLocalDebuggerRunner::setBreakpoint(const std::string& planNodeId) {
  breakpoints_.push_back(planNodeId);
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
