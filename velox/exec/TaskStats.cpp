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

#include "velox/exec/TaskStats.h"

#include <fmt/format.h>
#include <sstream>
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox::exec {

std::string PipelineStats::toString() const {
  std::stringstream out;

  out << "[PipelineStats] " << (inputPipeline ? "INPUT " : "")
      << (outputPipeline ? "OUTPUT " : "") << operatorStats.size()
      << " operators, " << driverStats.size() << " drivers.";

  for (const auto& operatorStat : operatorStats) {
    out << "\n    " << operatorStat.toString();
  }

  for (const auto& driverStat : driverStats) {
    out << "\n    " << driverStat.toString();
  }

  return out.str();
}

std::string TaskStats::toString() const {
  std::stringstream out;
  out << "[TaskStats] " << numFinishedSplits << "/" << numTotalSplits
      << " splits completed, " << numCompletedDrivers << "/" << numTotalDrivers
      << " drivers completed, " << pipelineStats.size() << " pipelines";

  if (executionStartTimeMs > 0 && executionEndTimeMs > 0) {
    out << " (" << succinctMillis(executionEndTimeMs - executionStartTimeMs)
        << "ms)";
  }

  for (const auto& pipelineStat : pipelineStats) {
    out << "\n  " << pipelineStat.toString();
  }

  return out.str();
}

} // namespace facebook::velox::exec
