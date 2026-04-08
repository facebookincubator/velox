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
  int32_t code;
  int32_t line;
  char message[80];
  int64_t extra[2];
};

/// Each TB fetches its instructions from BlockInfo at blockIdx.x. Copied at
/// offset 0 in dynamic shared memory of the block.
struct BlockInfo {
  /// kernel dependent op code for this block.
  int32_t op;

  /// Index of this block within the blocks with for the same op.
  int32_t blockInOp;

  /// Number of blocks for this op.
  int32_t numBlocksInOp;

  /// Number of data items to process in this block. Suppose 2 blocks of 256
  /// with 600 elements to process, the first would have 512 and the next would
  /// have 88.
  int32_t rowsForBlock;

  /// Index of row processed by lane 0 of this block. The block increments this
  /// by blockDim.x on each iteration until this is >= rowsForBlock.
  int32_t rowIdx{0};
  /// Pointer to per-op params, format depends on 'op'.
  void* params;

  // Pointer to per block return status. If not nullptr, the block must write a
  // per-block status to pass to host here. For example, for stream compaction,
  // the result length for data processed by this block. Consolidated with
  // return data of other blocks for single D to H transfer.
  void* returnData;

  DebugInfo* debug;
};

struct TorchWaveParams {
  // Pointer to BlockInfo for blockIdx.x 0. gridDim.x consecutive BlockInfos. If
  // nullptr the BlockInfos are in inlineInfo.
  BlockInfo* info;
  BlockInfo inlineInfo[100];
};

} // namespace torch::wave
