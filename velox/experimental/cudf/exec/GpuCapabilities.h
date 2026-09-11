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
#include <string>

namespace facebook::velox::cudf_velox {

/// Static description of the GPU this process executes on.
///
/// Every memory-related default in the cuDF operators is a size in bytes or a
/// row count, and a constant that is right for a 48 GiB device is wrong for a
/// 16 GiB one and wasteful on a 180 GiB one. Capturing what the device actually
/// is lets those defaults be derived rather than guessed, so a deployment gets
/// sensible behaviour without hand-tuning per machine.
///
/// The fields cover four axes that operator decisions depend on: capacity (how
/// much can be resident), compute (how much parallelism to feed), cache and
/// bandwidth (how big a working set stays efficient), and interconnect (what a
/// transfer off the device costs). Deliberately broader than today's callers
/// need, because adding a field later means rebuilding every consumer, while an
/// unused field costs one query at startup.
///
/// Populated once per process from the CUDA runtime API only - no NVML - so it
/// works in containers that do not expose the management library.
struct GpuCapabilities {
  /// Incremented when the meaning of a field changes or a field is removed, so
  /// a consumer reading a serialized copy (see toJson()) can reject one it does
  /// not understand. Adding a field does not require a bump.
  static constexpr int32_t kSchemaVersion = 1;

  /// Ordinal of the device within this process, i.e. what
  /// cudaSetDevice() was called with.
  int32_t deviceIndex{-1};

  /// Marketing name, e.g. "NVIDIA RTX 5880 Ada Generation".
  std::string name;

  /// Device UUID, stable across reboots and driver reloads. This is the field
  /// to key persisted per-device tuning on; deviceIndex and pciBusId are not
  /// stable across topology changes.
  std::string uuid;

  /// PCI bus id in the "domain:bus:device.function" form the driver reports.
  std::string pciBusId;

  /// Compute capability, which gates which kernels and instructions exist.
  int32_t computeCapabilityMajor{0};
  int32_t computeCapabilityMinor{0};

  /// Total device memory. The anchor for every capacity-derived default; note
  /// it is the physical size, not what is free, which is a runtime quantity.
  int64_t totalMemoryBytes{0};

  /// Streaming multiprocessor count and the threads each can hold resident.
  /// Together these bound useful concurrency, which is what decides how many
  /// operator instances can share the device before they only contend.
  int32_t multiprocessorCount{0};
  int32_t maxThreadsPerMultiprocessor{0};

  /// L2 cache size. A working set that fits here behaves very differently from
  /// one that does not, which is why batch-size defaults reference it.
  int64_t l2CacheBytes{0};

  /// Memory interface, and the peak bandwidth derived from it. Bandwidth is the
  /// right denominator when judging whether an extra pass over data is cheap.
  int32_t memoryBusWidthBits{0};
  int32_t memoryClockRateKhz{0};
  int64_t peakMemoryBandwidthBytesPerSecond{0};

  /// Whether the driver supports stream-ordered memory pools, i.e. whether
  /// cudf.memory_resource=async is available at all on this device.
  bool memoryPoolsSupported{false};

  /// Managed (unified) memory support, and whether the device can access it
  /// concurrently with the host. Oversubscription strategies depend on both.
  bool managedMemorySupported{false};
  bool concurrentManagedAccess{false};

  /// Number of other GPUs this device can reach directly. Zero means anything
  /// leaving the device crosses PCIe to the host, which is what makes spilling
  /// and exchange expensive; non-zero means a peer link (NVLink or PCIe P2P)
  /// exists and moving data sideways may beat moving it down.
  int32_t numPeerDevices{0};

  /// Serializes to JSON. Intended for a worker to advertise what it is running
  /// on - a Presto worker announcing to its coordinator, a log line that makes
  /// a benchmark reproducible, or a test fixture asserting on the shape of the
  /// machine. Includes kSchemaVersion so the reader can version-check.
  std::string toJson() const;

  /// Human-readable one-liner for startup logs.
  std::string toString() const;
};

/// Reads the capabilities of `deviceIndex`, or of the current device when it is
/// negative. Call once, after the CUDA context exists and before anything that
/// derives a default from it. Safe to call again; the last call wins.
///
/// Never throws: a device query that fails leaves the corresponding field at
/// its default and logs, because a missing capability must degrade to today's
/// fixed defaults rather than fail to start the worker.
void initializeGpuCapabilities(int32_t deviceIndex = -1);

/// The capabilities captured by initializeGpuCapabilities(). If that has not
/// run, returns a default-constructed value whose totalMemoryBytes is zero,
/// which every derivation below treats as "unknown, use the fixed default".
const GpuCapabilities& gpuCapabilities();

/// Defaults derived from the device.
///
/// Kept separate from the data so that adding or retuning a derivation does not
/// touch GpuCapabilities itself, and so each rule states the reasoning behind
/// its fraction. Every function returns `fallback` unchanged when the device is
/// unknown, so a caller can pass today's constant and get exactly today's
/// behaviour on a machine that could not be queried.
namespace gpu_defaults {

/// Byte target for GPU-side batch accumulation (CudfBatchConcat), per driver.
///
/// Batches exist to amortise kernel launches, so bigger is better until the
/// batches of all concurrent drivers stop fitting alongside operator state. A
/// small fraction of the device divided by the drivers sharing it keeps that
/// true on any card: the same fraction is a few hundred MiB on a 48 GiB device
/// and scales up on a larger one.
uint64_t batchSizeMinBytes(uint64_t fallback, int32_t numDriversPerTask);

/// State size at which a final aggregation starts spilling partitions to host.
///
/// Set below the point where one operator's state would crowd out the rest of
/// the query, so spilling begins while there is still room to do it.
uint64_t finalGroupbySpillBytes(uint64_t fallback);

/// Build row count above which a denser hash-join table is worth its probe
/// cost. A build only threatens the device once its hash table is a material
/// fraction of it, so the threshold follows device capacity.
uint64_t hashJoinDenseLoadFactorMinRows(uint64_t fallback);

} // namespace gpu_defaults

} // namespace facebook::velox::cudf_velox
