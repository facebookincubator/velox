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

#include "velox/exec/OperatorStats.h"

#include <fmt/format.h>
#include <sstream>
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox::exec {

std::string OperatorStats::toString() const {
  std::stringstream out;

  out << "[OperatorStats][" << operatorId << "] " << operatorType << " ["
      << planNodeId << "]";

  out << "\n        input: " << inputPositions << " rows ("
      << succinctBytes(inputBytes) << " bytes, " << inputVectors << " vectors)";
  out << "\n        output: " << outputPositions << " rows ("
      << succinctBytes(outputBytes) << " bytes, " << outputVectors
      << " vectors)";

  if (numSplits > 0) {
    out << "\n        splits: " << numSplits;
  }

  if (spilledBytes > 0) {
    out << "\n      spilled: " << succinctBytes(spilledBytes) << " bytes ("
        << spilledRows << " rows, " << spilledPartitions << " partitions)";
  }

  if (blockedWallNanos > 0) {
    out << "\n      blockedWallTime: " << succinctNanos(blockedWallNanos);
  }

  out << "\n        runtimeStats: {";
  for (const auto& [name, metric] : runtimeStats) {
    out << "\n          " << name << ":" << metric.sum << ",";
  }
  out << "\n        }";

  return out.str();
}

} // namespace facebook::velox::exec
