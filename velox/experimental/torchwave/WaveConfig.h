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

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/core/ScalarType.h>
#include <folly/CPortability.h>

namespace c10 {
struct IValue;
}

namespace torch::wave {

/// Process-wide configuration for wave graph execution (block size, tracing,
/// grid hints).
struct WaveConfig {
  static constexpr int32_t kNodes = 1;
  static constexpr int32_t kLaunches = 2;
  static constexpr int32_t kTensors = 4;
  static constexpr int32_t kFrame = 8;
  static constexpr int32_t kTiming = 16;

  int32_t blockSize{256};
  bool allStandalone{false};

  /// If non-zero, use this as the number of SMs instead of reading from the
  /// device.
  int32_t numSms{0};

  /// Trace bit mask. kNodes prints node headers, kLaunches prints per-launch
  /// details.
  int32_t trace{0};

  /// If set, forces the grid choice between single-block and multi-block
  /// variants. If nullopt, the choice is made based on input size.
  std::optional<bool> useSingleBlock;

  /// If set and true, use the cooperative grid variant when available.
  std::optional<bool> isCg;

  /// Reference values keyed by ValueId for verifying intermediates.
  std::unordered_map<int32_t, c10::IValue>* referenceFrame{nullptr};

  /// If non-empty, save the wave execution frame to this path.
  std::string saveReferenceFramePath;

  // If non-empty, cache compiled CUDA kernels (cubin) in this directory.
  std::string kernelCacheDir;

  // Max pointer variables in elementwise codegen before inlining storage
  // expressions.
  int32_t maxElementwiseVars{7};

  // Character threshold for extracting elementwise subtrees into
  // __device__ __noinline__ helpers. 0 disables extraction.
  int32_t outOfLineExprSize{10'000};

  // Print timing for wave graph execution.
  bool printTiming{false};

  // Comma-separated list of value ids to trace during execution.
  std::string traceValues;

  // Max elements printed per tensor when tracing values. 0 means no limit.
  int32_t tensorPrintElementLimit{100};

  // Re-verify all previously passed reference values on each step to detect
  // corruption.
  bool reverify{false};

  // If true, copy per-block debug info from device to thread-local storage
  // before returning the execution state to the pool.
  bool keepStatsOnThread{true};

  // If true, throw after execution if any block reported an error.
  bool throwOnError{true};

  // If true, skip the elementwise fast path and always generate the slow
  // path with complexIdx.
  bool noElementwiseFastPath{false};

  // If true, log reference mismatches but continue execution instead of
  // throwing.
  bool continueAfterMismatch{false};

  // Enable device-side debug printfs. Emergency use only.
  bool kernelDebugOutput{false};

  // Launch kernel once per block for debugging, waiting between launches.
  // Each kernel op runs as a standalone invocation so device-side errors
  // can be attributed to a single op.
  bool debugSingleOps{false};

  // If true, adjust per-op cost multipliers after each execution based on
  // actual thread block clock distribution.
  bool autoAdjustCost{false};

  /// Not thread-safe. All mutations must happen before concurrent reads.
  FOLLY_EXPORT static WaveConfig& get() {
    static WaveConfig instance;
    return instance;
  }
};

} // namespace torch::wave
