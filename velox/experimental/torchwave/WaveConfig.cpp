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

#include "velox/experimental/torchwave/WaveConfig.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace torch::wave {

std::string WaveConfig::toString() const {
  std::vector<std::string> parts;
  auto addBool = [&](const char* name, bool v, bool def) {
    if (v != def) {
      parts.push_back(std::string(name) + "=" + (v ? "true" : "false"));
    }
  };
  auto addInt = [&](const char* name, int64_t v, int64_t def) {
    if (v != def) {
      parts.push_back(std::string(name) + "=" + std::to_string(v));
    }
  };
  auto addOptBool = [&](const char* name, const std::optional<bool>& v) {
    if (v.has_value()) {
      parts.push_back(std::string(name) + "=" + (*v ? "true" : "false"));
    }
  };
  auto addStr = [&](const char* name, const std::string& v) {
    if (!v.empty()) {
      parts.push_back(std::string(name) + "=" + v);
    }
  };

  addInt("blockSize", blockSize, 256);
  addBool("allStandalone", allStandalone, false);
  addInt("numSms", numSms, 0);
  addInt("trace", trace, 0);
  addOptBool("useSingleBlock", useSingleBlock);
  addOptBool("isCg", isCg);
  if (referenceFrame != nullptr) {
    parts.emplace_back("referenceFrame=set");
  }
  addStr("saveReferenceFramePath", saveReferenceFramePath);
  addStr("kernelCacheDir", kernelCacheDir);
  addInt("maxElementwiseVars", maxElementwiseVars, 7);
  addInt("outOfLineExprSize", outOfLineExprSize, 10'000);
  addBool("printTiming", printTiming, false);
  addStr("traceValues", traceValues);
  addInt("tensorPrintElementLimit", tensorPrintElementLimit, 100);
  addBool("reverify", reverify, false);
  addBool("keepStatsOnThread", keepStatsOnThread, true);
  addBool("throwOnError", throwOnError, true);
  addBool("noElementwiseFastPath", noElementwiseFastPath, false);
  addBool("continueAfterMismatch", continueAfterMismatch, false);
  addBool("kernelDebugOutput", kernelDebugOutput, false);
  addBool("kernelLineInfo", kernelLineInfo, false);
  addBool("debugSingleOps", debugSingleOps, false);
  addBool("autoAdjustCost", autoAdjustCost, false);
  addBool("enableReuse", enableReuse, false);
  addBool("scanOutputReturnBarrier", scanOutputReturnBarrier, true);
  addBool("freeIntermediates", freeIntermediates, false);

  if (parts.empty()) {
    return "defaults";
  }
  std::string out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += parts[i];
  }
  return out;
}

} // namespace torch::wave
