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

namespace torch::wave {
/// Header for host to torch::wave kernel communication. Includeed in both host
/// and device code.

struct Tensor {
  void* storage;
  int8_t rank;
  int32_t dims[3];
  int32_t strides[3];
};

/// Struct for returning errors to host. Each block has one. These may be
/// checked at a delay, so return status that requires action on host side must
/// be sent in returnStatus, not here.
struct DebugInfo {
  int64_t clocks{0};
  int32_t op{0};
  int32_t line{0};
  int64_t extra[2] = {};
  char message[20] = {};
};

/// Each TB fetches its instructions from BlockInfo at blockIdx.x. Copied at
/// offset 0 in dynamic shared memory of the block.
struct BlockInfo {
  /// kernel dependent op code for this block.
  int32_t op;

  /// The first blockIdx.x in grid that should execute this op.
  int32_t firstBlockIdx;

  /// Index of this block within the blocks with for the same op.
  int32_t blockInOp;

  /// Number of blocks for this op.
  int32_t numBlocksInOp;

  /// Pointer to per-op params, format depends on 'op'.
  void* params;

  DebugInfo* debugInfo;

  /// clock64() at start of block.
  int64_t start;
};

struct TorchWaveParams {
  // Pointer to BlockInfo for blockIdx.x 0. gridDim.x consecutive BlockInfos. If
  // nullptr the BlockInfos are in inlineInfo.
  BlockInfo* info;
  DebugInfo* debugInfo;
  int32_t extraCounter{0};
  int32_t numExtraBlocks{0};
  BlockInfo inlineInfo[100];
};

} // namespace torch::wave
