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

#include "velox/experimental/cudf/CudfDefaultStreamOverload.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

#include <cuda_runtime.h>

#include <common/base/Exceptions.h>
#include <glog/logging.h>

#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox {

namespace {

std::mutex deviceMemoryStateMutex;
std::unordered_map<int, cudaMemPool_t> currentAsyncPools;
std::unordered_map<int, uint64_t> inProcessReservations;

uint64_t addSaturated(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left + right;
}

std::optional<CudfDeviceMemoryInfo> currentDeviceMemoryInfoLocked(int device) {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  const auto memInfoStatus = cudaMemGetInfo(&freeBytes, &totalBytes);
  if (memInfoStatus != cudaSuccess) {
    VLOG(1) << "cudaMemGetInfo failed while reading cuDF memory info: "
            << cudaGetErrorString(memInfoStatus);
    return std::nullopt;
  }

  CudfDeviceMemoryInfo info{
      .deviceId = device,
      .freeBytes = static_cast<uint64_t>(freeBytes),
      .totalBytes = static_cast<uint64_t>(totalBytes)};
  if (const auto reservation = inProcessReservations.find(device);
      reservation != inProcessReservations.end()) {
    info.inProcessReservedBytes = reservation->second;
  }

  const auto pool = currentAsyncPools.find(device);
  if (pool == currentAsyncPools.end()) {
    return info;
  }

  uint64_t reservedBytes = 0;
  uint64_t usedBytes = 0;
  const auto reservedStatus = cudaMemPoolGetAttribute(
      pool->second, cudaMemPoolAttrReservedMemCurrent, &reservedBytes);
  const auto usedStatus = cudaMemPoolGetAttribute(
      pool->second, cudaMemPoolAttrUsedMemCurrent, &usedBytes);
  if (reservedStatus != cudaSuccess || usedStatus != cudaSuccess) {
    VLOG(1) << "cudaMemPoolGetAttribute failed while reading cuDF pool info: "
            << "reservedStatus=" << cudaGetErrorString(reservedStatus)
            << " usedStatus=" << cudaGetErrorString(usedStatus);
    return info;
  }

  info.poolReservedBytes = reservedBytes;
  info.poolUsedBytes = usedBytes;
  info.poolReusableBytes =
      reservedBytes > usedBytes ? reservedBytes - usedBytes : 0;
  info.hasPoolStats = true;
  return info;
}

void trackCurrentAsyncPool(
    std::optional<cudaMemPool_t> poolHandle,
    bool trackAsCurrent) {
  if (!trackAsCurrent) {
    return;
  }

  int device = 0;
  const auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    VLOG(1) << "cudaGetDevice failed while tracking cuDF memory resource: "
            << cudaGetErrorString(status);
    return;
  }

  std::lock_guard<std::mutex> lock(deviceMemoryStateMutex);
  if (poolHandle.has_value()) {
    currentAsyncPools[device] = *poolHandle;
  } else {
    currentAsyncPools.erase(device);
  }
}

} // namespace

cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent, bool trackAsCurrent) {
  // A tracked non-async resource must replace, rather than inherit, any pool
  // handle previously associated with this device.
  trackCurrentAsyncPool(std::nullopt, trackAsCurrent);

  if (mode == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (mode == "pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "async") {
    auto asyncResource = rmm::mr::cuda_async_memory_resource{};
    const auto poolHandle = asyncResource.pool_handle();
    // Type erasure may allocate. Publish the non-owning pool handle only
    // after the returned owner exists; moving that owner is noexcept.
    cuda::mr::any_resource<cuda::mr::device_accessible> resource{
        std::move(asyncResource)};
    trackCurrentAsyncPool(poolHandle, trackAsCurrent);
    return resource;
  } else if (mode == "arena") {
    return rmm::mr::arena_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed") {
    return rmm::mr::managed_memory_resource{};
  } else if (mode == "managed_pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::managed_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed_async") {
    auto asyncResource = rmm::mr::cuda_async_managed_memory_resource{};
    const auto poolHandle = asyncResource.pool_handle();
    cuda::mr::any_resource<cuda::mr::device_accessible> resource{
        std::move(asyncResource)};
    trackCurrentAsyncPool(poolHandle, trackAsCurrent);
    return resource;
  } else if (mode == "prefetch_managed") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::managed_memory_resource{});
  } else if (mode == "prefetch_managed_pool") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::pool_memory_resource(
            rmm::mr::managed_memory_resource{},
            rmm::percent_of_free_device_memory(percent)));
  } else if (mode == "prefetch_managed_async") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::cuda_async_managed_memory_resource{});
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::current_cuda_stream_pool();
};

namespace {

// Device owning the registration-time primary context
int cudfContextDevice = -1;

} // namespace

void setCudfContextDevice(int device) {
  cudfContextDevice = device;
}

void ensureCudaContextForThread() {
  // Device is recorded by setCudfContextDevice() from registerCudf().
  VELOX_CHECK_GE(
      cudfContextDevice,
      0,
      "cuDF is not yet registered. Please call registerCudf() first.");
  static thread_local int threadDevice = -1;
  if (threadDevice == cudfContextDevice) {
    return;
  }

  VELOX_CHECK_EQ(
      static_cast<int>(cudaSuccess),
      static_cast<int>(cudaSetDevice(cudfContextDevice)),
      "Failed to set CUDA device for cuDF worker thread",
      cudfContextDevice);
  VELOX_CHECK_EQ(
      static_cast<int>(cudaSuccess),
      static_cast<int>(cudaFree(nullptr)),
      "Failed to initialize CUDA context on cuDF worker thread");
  int boundDevice = -1;
  VELOX_CHECK_EQ(
      static_cast<int>(cudaSuccess),
      static_cast<int>(cudaGetDevice(&boundDevice)),
      "Failed to get current CUDA device ordinal");
  VELOX_CHECK_EQ(boundDevice, cudfContextDevice);

  threadDevice = boundDevice;
}

cuda::stream_ref getDefaultStreamForCurrentThread() {
  ensureCudaContextForThread();
  return cudf::get_default_stream(cudf::allow_default_stream);
}

std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> output_mr_;
std::optional<rmm::mr::statistics_resource_adaptor> statsMr_;
std::optional<rmm::mr::statistics_resource_adaptor> outputStatsMr_;

rmm::device_async_resource_ref get_output_mr() {
  return output_mr_.value();
}

int64_t cudfAllocatedBytes() {
  if (!statsMr_.has_value()) {
    return -1;
  }
  int64_t bytes = statsMr_->get_bytes_counter().value;
  if (outputStatsMr_.has_value()) {
    bytes += outputStatsMr_->get_bytes_counter().value;
  }
  return bytes;
}

std::optional<CudfDeviceMemoryInfo> currentDeviceMemoryInfo() {
  int device = 0;
  const auto deviceStatus = cudaGetDevice(&device);
  if (deviceStatus != cudaSuccess) {
    VLOG(1) << "cudaGetDevice failed while reading cuDF memory info: "
            << cudaGetErrorString(deviceStatus);
    return std::nullopt;
  }

  // Keep the lock through the CUDA queries. unregisterCudf() clears this map
  // before destroying the owning RMM resource; serialization here prevents a
  // query from racing with invalidation of the raw cudaMemPool_t handle.
  std::lock_guard<std::mutex> lock(deviceMemoryStateMutex);
  return currentDeviceMemoryInfoLocked(device);
}

CudfDeviceMemoryAdmission tryReserveCurrentDeviceMemory(
    uint64_t bytes,
    uint64_t requiredHeadroomBytes) {
  int device = 0;
  const auto deviceStatus = cudaGetDevice(&device);
  if (deviceStatus != cudaSuccess) {
    VLOG(1) << "cudaGetDevice failed while reserving cuDF device memory: "
            << cudaGetErrorString(deviceStatus);
    return {};
  }

  std::lock_guard<std::mutex> lock(deviceMemoryStateMutex);
  const auto memoryInfo = currentDeviceMemoryInfoLocked(device);
  if (!memoryInfo.has_value()) {
    return {};
  }

  const auto effectiveFreeBytes =
      addSaturated(memoryInfo->freeBytes, memoryInfo->poolReusableBytes);
  const auto requiredBytes = addSaturated(
      addSaturated(memoryInfo->inProcessReservedBytes, bytes),
      requiredHeadroomBytes);
  if (effectiveFreeBytes <= requiredBytes) {
    return CudfDeviceMemoryAdmission{
        .status = CudfDeviceMemoryAdmissionStatus::kInsufficient,
        .deviceId = device,
        .bytes = bytes};
  }

  inProcessReservations[device] =
      addSaturated(memoryInfo->inProcessReservedBytes, bytes);
  return CudfDeviceMemoryAdmission{
      .status = CudfDeviceMemoryAdmissionStatus::kAdmitted,
      .deviceId = device,
      .bytes = bytes};
}

void releaseDeviceMemoryReservation(
    const CudfDeviceMemoryAdmission& admission) noexcept {
  if (admission.status != CudfDeviceMemoryAdmissionStatus::kAdmitted ||
      admission.bytes == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(deviceMemoryStateMutex);
  const auto reservation = inProcessReservations.find(admission.deviceId);
  if (reservation == inProcessReservations.end()) {
    VLOG(1) << "No cuDF device-memory reservation found for device "
            << admission.deviceId << " while releasing " << admission.bytes
            << " bytes";
    return;
  }
  if (reservation->second < admission.bytes) {
    VLOG(1) << "cuDF device-memory reservation underflow for device "
            << admission.deviceId << ": tracked=" << reservation->second
            << " release=" << admission.bytes;
    inProcessReservations.erase(reservation);
    return;
  }

  reservation->second -= admission.bytes;
  if (reservation->second == 0) {
    inProcessReservations.erase(reservation);
  }
}

void clearCurrentDeviceMemoryInfo() {
  std::lock_guard<std::mutex> lock(deviceMemoryStateMutex);
  currentAsyncPools.clear();
  inProcessReservations.clear();
}

} // namespace facebook::velox::cudf_velox

// This must NOT be in a file that includes CudfNoDefaults.h, because
// CudfNoDefaults.h redeclares cudf::get_default_stream() with
// __attribute__((error)). The overload below calls the real function.
namespace cudf {

cuda::stream_ref const get_default_stream(allow_default_stream_t) {
  return cudf::get_default_stream();
}

} // namespace cudf
