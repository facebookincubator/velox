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

#include "velox/experimental/cudf/exec/GpuMemoryLedger.h"

#include <cuda/memory_resource>

namespace facebook::velox::exec {
class Operator;
}

namespace facebook::velox::cudf_velox {

using GpuMemoryResource = cuda::mr::any_resource<cuda::mr::device_accessible>;

/// The default and output resources, both feeding one ledger: simultaneous
/// ownership across the two is the useful signal when debugging an OOM.
struct GpuMemoryResourcePair {
  GpuMemoryResource main;
  GpuMemoryResource output;
};

/// Wraps both resources in place so every allocation they serve is attributed
/// to the operator that made it. Leaves them untouched and returns false when
/// cudf.memory_tracking_enabled is off.
bool installGpuMemoryTracking(
    GpuMemoryResource& mainMr,
    GpuMemoryResource& outputMr);

/// Gives 'op' a counter row before it allocates, so a capture can tell an
/// operator that held no memory from one that never ran.
///
/// Which operators get a row is a display choice only. Attribution correctness
/// rests on discardGpuMemoryOwnerCache().
void registerGpuMemoryOperator(exec::Operator* op) noexcept;

/// Drops every thread's cached operator-to-owner resolution.
///
/// That cache is keyed on the raw exec::Operator address, and a destroyed
/// driver's addresses can be reused by a later one. Call this for every
/// compiled driver, not just for operators that get a counter row: an operator
/// without a row still allocates, and would be charged to whoever last held its
/// address.
void discardGpuMemoryOwnerCache() noexcept;

/// Two wrappers over one shared ledger, published as the process-wide ledger.
[[nodiscard]] GpuMemoryResourcePair createGpuMemoryTrackingResources(
    GpuMemoryResource mainUpstream,
    GpuMemoryResource outputUpstream);

/// Drops the process-wide ledger and zeroes its counters.
void resetGpuMemoryTracking();

/// An empty snapshot when no ledger is installed.
[[nodiscard]] GpuMemorySnapshot getGpuMemorySnapshot();

} // namespace facebook::velox::cudf_velox
