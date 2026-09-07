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
  // Take the defaults from a default-constructed config rather than repeating
  // them here. A repeated default that drifts from the header still compiles
  // and silently inverts the report for that field.
  static const WaveConfig kDefaults{};

  std::vector<std::string> parts;
  auto addBool = [&](const char* name, bool WaveConfig::* field) {
    if (this->*field != kDefaults.*field) {
      parts.push_back(
          std::string(name) + "=" + (this->*field ? "true" : "false"));
    }
  };
  auto addInt = [&](const char* name, auto WaveConfig::* field) {
    if (this->*field != kDefaults.*field) {
      parts.push_back(std::string(name) + "=" + std::to_string(this->*field));
    }
  };
  auto addOptBool = [&](const char* name,
                        std::optional<bool> WaveConfig::* field) {
    const std::optional<bool>& value = this->*field;
    if (value.has_value()) {
      parts.push_back(std::string(name) + "=" + (*value ? "true" : "false"));
    }
  };
  auto addStr = [&](const char* name, std::string WaveConfig::* field) {
    const std::string& value = this->*field;
    if (!value.empty()) {
      parts.push_back(std::string(name) + "=" + value);
    }
  };

  addInt("blockSize", &WaveConfig::blockSize);
  addBool("allStandalone", &WaveConfig::allStandalone);
  addInt("numSms", &WaveConfig::numSms);
  addInt("trace", &WaveConfig::trace);
  addOptBool("useSingleBlock", &WaveConfig::useSingleBlock);
  addOptBool("isCg", &WaveConfig::isCg);
  if (referenceFrame != nullptr) {
    parts.emplace_back("referenceFrame=set");
  }
  addStr("saveReferenceFramePath", &WaveConfig::saveReferenceFramePath);
  addStr("kernelCacheDir", &WaveConfig::kernelCacheDir);
  addInt("maxElementwiseVars", &WaveConfig::maxElementwiseVars);
  addInt("outOfLineExprSize", &WaveConfig::outOfLineExprSize);
  addBool("printTiming", &WaveConfig::printTiming);
  addBool("perOpStandaloneTiming", &WaveConfig::perOpStandaloneTiming);
  addStr("traceValues", &WaveConfig::traceValues);
  addInt("tensorPrintElementLimit", &WaveConfig::tensorPrintElementLimit);
  addBool("reverify", &WaveConfig::reverify);
  addBool("keepStatsOnThread", &WaveConfig::keepStatsOnThread);
  addBool("throwOnError", &WaveConfig::throwOnError);
  addBool("noElementwiseFastPath", &WaveConfig::noElementwiseFastPath);
  addBool("continueAfterMismatch", &WaveConfig::continueAfterMismatch);
  addBool("kernelDebugOutput", &WaveConfig::kernelDebugOutput);
  addBool("kernelLineInfo", &WaveConfig::kernelLineInfo);
  addBool("debugSingleOps", &WaveConfig::debugSingleOps);
  addBool("autoAdjustCost", &WaveConfig::autoAdjustCost);
  addBool("enableReuse", &WaveConfig::enableReuse);
  addBool("elideClones", &WaveConfig::elideClones);
  addBool("scanOutputReturnBarrier", &WaveConfig::scanOutputReturnBarrier);
  addBool("freeIntermediates", &WaveConfig::freeIntermediates);
  addBool("inputContiguous", &WaveConfig::inputContiguous);
  addBool("cseCompute", &WaveConfig::cseCompute);
  addBool("cseViews", &WaveConfig::cseViews);
  addBool("decomposeLists", &WaveConfig::decomposeLists);
  addBool("foldSharedChains", &WaveConfig::foldSharedChains);
  addBool("mkSelect", &WaveConfig::mkSelect);
  addBool("stepLastUse", &WaveConfig::stepLastUse);
  addBool("syncEachStep", &WaveConfig::syncEachStep);
  addBool("deferD2h", &WaveConfig::deferD2h);
  addBool("runAhead", &WaveConfig::runAhead);
  addInt("maxDelayedFree", &WaveConfig::maxDelayedFree);
  addBool("duplicateMetadata", &WaveConfig::duplicateMetadata);
  addBool("configPerOp", &WaveConfig::configPerOp);
  addBool("deferSizeOutputs", &WaveConfig::deferSizeOutputs);
  addBool("donateBuffers", &WaveConfig::donateBuffers);
  addInt("donationCarryBytes", &WaveConfig::donationCarryBytes);
  addBool("enableAllocGroup", &WaveConfig::enableAllocGroup);
  addBool("enableConcatAllocGroup", &WaveConfig::enableConcatAllocGroup);
  addBool("parallelConcatFill", &WaveConfig::parallelConcatFill);
  addBool("singlePassSelect", &WaveConfig::singlePassSelect);
  addBool("singlePass", &WaveConfig::singlePass);

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
