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
/// Describes exactly one device. A process that uses several has to keep one of
/// these per device; the process-wide instance below is the device cuDF was
/// registered on.
///
/// Populated from the CUDA runtime API only - no NVML - so it works in
/// containers that do not expose the management library.
///
/// wave::Device (velox/experimental/wave/common/Cuda.h) describes a GPU from
/// the same cudaGetDeviceProperties() call and overlaps this struct. The two
/// stay separate by design: that header deliberately keeps CUDA types out
/// because they interfere with BitUtils.h and SimdUtils.h, whereas this
/// collector works from cudaDeviceProp directly. Do not merge them.
struct GpuCapabilities {
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
  /// it is the physical size, not what is free, which is a runtime quantity,
  /// and not what an operator may use, which the memory resource caps at
  /// CudfConfig::memoryPercent.
  int64_t totalMemoryBytes{0};

  /// Streaming multiprocessor count and the threads each can hold resident.
  /// Together these bound useful concurrency, which is what decides how many
  /// operator instances can share the device before they only contend.
  int32_t multiprocessorCount{0};
  int32_t maxThreadsPerMultiprocessor{0};

  /// L2 cache size. A working set that fits here behaves very differently from
  /// one that does not.
  int64_t l2CacheBytes{0};

  /// Memory interface, and the peak bandwidth derived from it. Bandwidth is the
  /// right denominator when judging whether an extra pass over data is cheap.
  /// CUDA 13 removed the memory clock from cudaDeviceProp, so both the clock
  /// and the bandwidth stay zero when the runtime does not report it.
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

  /// Serializes to JSON, for a log line that makes a benchmark reproducible or
  /// a test fixture asserting on the shape of the machine.
  std::string toJson() const;

  /// Human-readable one-liner for startup logs.
  std::string toString() const;
};

/// Passed as the device index to describe whichever device is current.
constexpr int32_t kCurrentDevice = -1;

/// Reads and returns the capabilities of `deviceIndex`, or of the current
/// device when it is kCurrentDevice. Touches no shared state, so it can be
/// called at any time and on any thread.
///
/// Never throws: a device query that fails leaves the corresponding field at
/// its default and logs, because a missing capability must degrade to today's
/// fixed defaults rather than fail to start the worker. A device that could not
/// be read at all comes back default-constructed, with totalMemoryBytes zero.
GpuCapabilities readGpuCapabilities(int32_t deviceIndex);

/// Publishes the description of `deviceIndex` as the one gpuCapabilities()
/// returns, and logs it. Reads the device on the first call only; later calls
/// do nothing, whatever index they name.
void initializeGpuCapabilities(int32_t deviceIndex);

/// The published description. Default-constructed, with totalMemoryBytes zero,
/// until initializeGpuCapabilities() has run. The reference stays valid and
/// unchanging for the life of the process.
const GpuCapabilities& gpuCapabilities();

} // namespace facebook::velox::cudf_velox
