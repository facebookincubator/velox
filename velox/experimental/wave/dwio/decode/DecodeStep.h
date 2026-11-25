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

#include "velox/experimental/wave/common/Buffer.h"
#include "velox/experimental/wave/common/GpuArena.h"
#include "velox/experimental/wave/common/ResultStaging.h"
#include "velox/experimental/wave/dwio/decode/DecodeOp.h"

namespace facebook::velox::wave {

struct DecodePrograms {
  void clear() {
    programs.clear();
    result.clear();
  }

  // Set of decode programs submitted as a unit. Each vector<DecodeStep> is run
  // on its own thread block. The consecutive DecodeSteps in the same program
  // are consecutive and the next one can depend on a previous one.
  std::vector<std::vector<std::unique_ptr<GpuDecode>>> programs;

  /// Unified or device memory   buffer where steps in 'programs' write results
  /// for the host. Decode results stay on device, only control information like
  /// filter result counts or length sums come to the host via this buffer. If
  /// nullptr, no data transfer is scheduled. 'result' should be nullptr if all
  /// steps are unconditional, like simple decoding.
  ResultBuffer result;
};

/// Describes the reader kernel launched by launchDecode().
struct DecodeLaunchInfo {
  /// Number of thread blocks.
  int32_t numBlocks{0};
  /// True if the launch requested a kernel not requested before in this
  /// process, i.e. one that had to be generated and compiled.
  bool isNewKernel{false};
  /// Host time in launching the kernel, including waiting for its compilation.
  int64_t launchMicros{0};
};

/// Launches 'programs' on 'stream' with a compiled reader kernel that has
/// the instantiations needed by the ops in 'programs'.
DecodeLaunchInfo launchDecode(
    const DecodePrograms& programs,
    LaunchParams& params,
    Stream* stream);

/// Returns the dynamic shared memory needed by 'step' for a TB of 'blockSize'
/// threads.
int32_t decodeSharedMemorySize(DecodeStep step, int32_t blockSize);

} // namespace facebook::velox::wave
