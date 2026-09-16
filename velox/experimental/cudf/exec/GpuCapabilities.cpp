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

#include "velox/experimental/cudf/exec/GpuCapabilities.h"

#include <cuda_runtime_api.h>

#include <fmt/format.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>

namespace facebook::velox::cudf_velox {
namespace {

GpuCapabilities& mutableGpuCapabilities() {
  static GpuCapabilities capabilities;
  return capabilities;
}

// Drops the runtime's per-thread error slot after a failure that is being
// swallowed. Without this the next cudaGetLastError() on this thread - the
// CUDF_CUDA_TRY(cudaGetLastError()) after a kernel launch, for instance - would
// report a startup query failure as that kernel's failure.
void clearLastCudaError() {
  cudaGetLastError();
}

// Formats the 16-byte CUDA UUID the way nvidia-smi does, so an operator can
// match a log line against the device list without transformation.
std::string formatUuid(const cudaUUID_t& uuid) {
  static constexpr std::array<int, 5> kGroupSizes{4, 2, 2, 2, 6};
  std::string formatted = "GPU-";
  size_t byte = 0;
  for (size_t group = 0; group < kGroupSizes.size(); ++group) {
    if (group > 0) {
      formatted += '-';
    }
    for (int i = 0; i < kGroupSizes[group]; ++i, ++byte) {
      formatted +=
          fmt::format("{:02x}", static_cast<unsigned char>(uuid.bytes[byte]));
    }
  }
  return formatted;
}

// Counts the devices this one can address directly. Peer access is what makes
// a sideways transfer cheaper than a trip through the host, so its presence -
// not its exact topology - is what operator decisions turn on.
int32_t countPeerDevices(int32_t deviceIndex) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    clearLastCudaError();
    return 0;
  }
  int32_t peers = 0;
  for (int peer = 0; peer < deviceCount; ++peer) {
    if (peer == deviceIndex) {
      continue;
    }
    int canAccess = 0;
    if (cudaDeviceCanAccessPeer(&canAccess, deviceIndex, peer) != cudaSuccess) {
      clearLastCudaError();
      continue;
    }
    if (canAccess != 0) {
      ++peers;
    }
  }
  return peers;
}

// Reads the memory clock, which CUDA 13 removed from cudaDeviceProp in favour
// of the device attribute. Returns zero when neither source has it, which
// leaves the derived bandwidth at zero rather than reporting a wrong one.
int32_t readMemoryClockRateKhz(
    const cudaDeviceProp& properties,
    int32_t deviceIndex) {
#if CUDART_VERSION < 13000
  (void)deviceIndex;
  return properties.memoryClockRate;
#else
  (void)properties;
  int clockRateKhz = 0;
  if (cudaDeviceGetAttribute(
          &clockRateKhz, cudaDevAttrMemoryClockRate, deviceIndex) !=
      cudaSuccess) {
    clearLastCudaError();
    return 0;
  }
  return clockRateKhz;
#endif
}

} // namespace

std::string GpuCapabilities::toJson() const {
  folly::dynamic object = folly::dynamic::object;
  object["schemaVersion"] = kSchemaVersion;
  object["deviceIndex"] = deviceIndex;
  object["name"] = name;
  object["uuid"] = uuid;
  object["pciBusId"] = pciBusId;
  object["computeCapability"] =
      fmt::format("{}.{}", computeCapabilityMajor, computeCapabilityMinor);
  object["totalMemoryBytes"] = totalMemoryBytes;
  object["multiprocessorCount"] = multiprocessorCount;
  object["maxThreadsPerMultiprocessor"] = maxThreadsPerMultiprocessor;
  object["l2CacheBytes"] = l2CacheBytes;
  object["memoryBusWidthBits"] = memoryBusWidthBits;
  object["memoryClockRateKhz"] = memoryClockRateKhz;
  object["peakMemoryBandwidthBytesPerSecond"] =
      peakMemoryBandwidthBytesPerSecond;
  object["memoryPoolsSupported"] = memoryPoolsSupported;
  object["managedMemorySupported"] = managedMemorySupported;
  object["concurrentManagedAccess"] = concurrentManagedAccess;
  object["numPeerDevices"] = numPeerDevices;
  return folly::toJson(object);
}

std::string GpuCapabilities::toString() const {
  if (totalMemoryBytes == 0) {
    return "GPU capabilities unknown";
  }
  return fmt::format(
      "{} (device {}, sm_{}{}, {:.1f} GiB, {} SMs, {} peer device(s))",
      name,
      deviceIndex,
      computeCapabilityMajor,
      computeCapabilityMinor,
      static_cast<double>(totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0),
      multiprocessorCount,
      numPeerDevices);
}

void initializeGpuCapabilities(int32_t deviceIndex) {
  int32_t resolvedIndex = deviceIndex;
  if (resolvedIndex < 0) {
    int current = 0;
    if (cudaGetDevice(&current) != cudaSuccess) {
      clearLastCudaError();
      LOG(WARNING) << "Could not determine the current CUDA device; cuDF "
                      "memory defaults will use their fixed values";
      return;
    }
    resolvedIndex = current;
  }

  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, resolvedIndex);
  if (status != cudaSuccess) {
    // Not fatal: every derived default falls back to its fixed value, which is
    // exactly the behaviour before this existed.
    clearLastCudaError();
    LOG(WARNING) << "cudaGetDeviceProperties failed for device "
                 << resolvedIndex << ": " << cudaGetErrorString(status)
                 << "; cuDF memory defaults will use their fixed values";
    return;
  }

  // Built separately and published only once complete, so a failed re-read
  // cannot leave a half-populated description behind.
  GpuCapabilities capabilities;
  capabilities.deviceIndex = resolvedIndex;
  capabilities.name = properties.name;
  capabilities.uuid = formatUuid(properties.uuid);
  capabilities.computeCapabilityMajor = properties.major;
  capabilities.computeCapabilityMinor = properties.minor;
  capabilities.totalMemoryBytes =
      static_cast<int64_t>(properties.totalGlobalMem);
  capabilities.multiprocessorCount = properties.multiProcessorCount;
  capabilities.maxThreadsPerMultiprocessor =
      properties.maxThreadsPerMultiProcessor;
  capabilities.l2CacheBytes = properties.l2CacheSize;
  capabilities.memoryBusWidthBits = properties.memoryBusWidth;
  capabilities.memoryClockRateKhz =
      readMemoryClockRateKhz(properties, resolvedIndex);
  capabilities.managedMemorySupported = properties.managedMemory != 0;
  capabilities.concurrentManagedAccess =
      properties.concurrentManagedAccess != 0;
  capabilities.memoryPoolsSupported = properties.memoryPoolsSupported != 0;

  // Peak bandwidth is the standard double-data-rate product: clock is reported
  // in kHz and the bus in bits, so the result is bytes per second.
  capabilities.peakMemoryBandwidthBytesPerSecond =
      static_cast<int64_t>(capabilities.memoryClockRateKhz) * 1000LL *
      (static_cast<int64_t>(capabilities.memoryBusWidthBits) / 8LL) * 2LL;

  std::array<char, 32> busId{};
  if (cudaDeviceGetPCIBusId(busId.data(), busId.size(), resolvedIndex) ==
      cudaSuccess) {
    capabilities.pciBusId = busId.data();
  } else {
    clearLastCudaError();
  }

  capabilities.numPeerDevices = countPeerDevices(resolvedIndex);

  LOG(INFO) << "cuDF GPU capabilities: " << capabilities.toString();
  mutableGpuCapabilities() = std::move(capabilities);
}

const GpuCapabilities& gpuCapabilities() {
  return mutableGpuCapabilities();
}

namespace gpu_defaults {
namespace {

// Returns `fraction` of device memory, or zero when the device is unknown so
// the caller can keep its fixed default.
int64_t fractionOfDevice(const GpuCapabilities& capabilities, double fraction) {
  const auto total = capabilities.totalMemoryBytes;
  if (total <= 0) {
    return 0;
  }
  return static_cast<int64_t>(static_cast<double>(total) * fraction);
}

} // namespace

int64_t batchSizeMinBytes(
    const GpuCapabilities& capabilities,
    int64_t fallback,
    int32_t numDriversPerTask) {
  // Batches from every driver of every concurrently running pipeline are
  // resident at once, so the share any single driver may hold has to stay well
  // under the device even when several pipelines overlap. A twentieth of the
  // device split across the drivers sharing it is ~250 MiB per driver on a
  // 48 GiB card with two drivers, which measured as the point where larger
  // batches stopped paying for themselves.
  const auto share = fractionOfDevice(capabilities, 0.05);
  if (share == 0) {
    return fallback;
  }
  static constexpr int64_t kMinBytes = 32LL << 20;
  const auto drivers = static_cast<int64_t>(std::max(numDriversPerTask, 1));
  return std::max<int64_t>(share / drivers, kMinBytes);
}

int64_t finalGroupbySpillBytes(
    const GpuCapabilities& capabilities,
    int64_t fallback) {
  // Spilling has to start while there is still room to build the partitions it
  // spills, so this sits below the point where one operator's state would
  // crowd out the rest of the query rather than at it.
  const auto budget = fractionOfDevice(capabilities, 0.10);
  return budget == 0 ? fallback : budget;
}

int64_t hashJoinDenseLoadFactorMinRows(
    const GpuCapabilities& capabilities,
    int64_t fallback) {
  // A hash table costs roughly two slots per key, and a key plus its payload is
  // on the order of 16 bytes, so a build of N rows occupies about 32N bytes.
  // Densifying only earns its probe cost once that is a noticeable share of the
  // device; a twentieth is the point where it starts to matter.
  const auto budget = fractionOfDevice(capabilities, 0.05);
  if (budget == 0) {
    return fallback;
  }
  static constexpr int64_t kApproxBytesPerBuildRow = 32;
  return budget / kApproxBytesPerBuildRow;
}

} // namespace gpu_defaults
} // namespace facebook::velox::cudf_velox
