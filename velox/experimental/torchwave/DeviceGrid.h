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

#include <ATen/ATen.h>
#include <cstdint>
#include <vector>

namespace facebook::velox::wave {
class Stream;
} // namespace facebook::velox::wave

namespace torch::wave {

struct DebugInfo;
struct FixupArgs;
struct GridDescriptor;

/// A step's grid computed on the device. Before each launch of the step, a
/// one-block kernel reads each launch's element count from the step's
/// parameters, gives the launches blocks so that all complete at the same
/// time, and writes the BlockInfo table the step's kernel reads. Each launch's
/// estimated cost per element is scaled by a measured multiplier that the
/// kernel keeps on the device, updated from the per-block clocks of the
/// previous execution.
struct DeviceGrid {
  /// The GridDescriptor packed for the kernel.
  std::vector<int32_t> words;
  int32_t numLaunches{0};
  /// The most blocks the step launches with: the grid's target.
  int32_t launchBlocks{0};
  /// The measured multipliers to start from: the host's cost adjustments.
  std::vector<float> multipliers;

  /// On the device, in one allocation: 'words', then per launch the
  /// multiplier and the previous cost (floats), then the previous block count
  /// and grid total followed per launch by the previous blocks and cap, and
  /// the launch of each block (ints), then per launch the element count for
  /// checking (longs). The offsets are in 32-bit words.
  at::Tensor device;
  int64_t floatsOffset{0};
  int64_t intsOffset{0};
  int64_t longsOffset{0};
};

/// Whether the grid kernel can hold a step of 'numLaunches' launches whose
/// descriptor is 'grid': no launch counted on the host, and the per-launch
/// arrays fit in shared memory. A descriptor too large for shared memory stays
/// in global memory.
bool deviceGridFits(const GridDescriptor& grid, int32_t numLaunches);

/// Packs 'grid' for the kernel. 'paramOffsets' are the launches' offsets in
/// the step's parameter region, 'costAdjust' their host cost adjustments.
/// Returns false when some launch's element count needs the host, or the step
/// has more launches than the kernel holds.
bool packDeviceGrid(
    const GridDescriptor& grid,
    const std::vector<int32_t>& paramOffsets,
    const std::vector<float>& costAdjust,
    int32_t numSMs,
    DeviceGrid& out);

/// Launches the grid kernel for the step whose parameters are at
/// 'deviceBase', after they and their fixups are in place. The host has put
/// one byte per launch, 1 when an operand is absent this execution, at
/// 'blockInfoOffset'; the kernel overwrites them with the BlockInfo table.
/// The step launches with 'launchBlocks' blocks; those past the computed grid
/// do nothing. 'deviceClocks' is the step's clocks in the execution's clock
/// buffer, followed by its barrier clocks 'launchBlocks' words later. With
/// 'previousClocks' they still hold the clocks of the step's previous
/// execution, which correct the launches' cost multipliers.
/// With 'fixups' the same kernel first runs the step's symbol
/// fixups, which the counts then see. The kernel returns at once when
/// 'failure' (the execution's failure words, or nullptr) names a step before
/// 'firstStep'.
void runDeviceGrid(
    DeviceGrid& grid,
    uint8_t* deviceBase,
    int64_t blockInfoOffset,
    int32_t launchBlocks,
    DebugInfo* deviceDebugBase,
    int64_t* deviceClocks,
    bool previousClocks,
    const FixupArgs* fixups,
    facebook::velox::wave::Stream* stream,
    const int32_t* failure,
    int32_t firstStep);

/// What the last grid kernel of 'grid' computed, copied back after a stream
/// sync: per launch the element count, the multiplier and the blocks.
struct DeviceGridResult {
  std::vector<int64_t> counts;
  std::vector<float> multipliers;
  std::vector<int32_t> blocks;
};

DeviceGridResult readDeviceGrid(const DeviceGrid& grid);

} // namespace torch::wave
