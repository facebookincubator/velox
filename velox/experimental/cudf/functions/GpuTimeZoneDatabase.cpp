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

#include "velox/experimental/cudf/functions/GpuTimeZoneDatabase.h"

#include "velox/common/base/Exceptions.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

// Lays out every zone tz::locateZone() resolves in one device array, with the
// absent entry in the ids the Velox database leaves unassigned.
GpuTimeZoneDatabase buildDatabase() {
  const auto ids = tz::getTimeZoneIDs();
  VELOX_CHECK_GT(ids.size(), 0);

  GpuTimeZone absent;
  absent.fixedOffset = GpuTimeZoneDatabase::kAbsentOffset;
  std::vector<GpuTimeZone> zones(ids.back() + 1, absent);
  for (const int16_t id : ids) {
    zones[id] = gpuTimeZone(*tz::locateZone(id));
  }

  void* device{nullptr};
  const auto bytes = zones.size() * sizeof(GpuTimeZone);
  CUDF_CUDA_TRY(cudaMalloc(&device, bytes));
  CUDF_CUDA_TRY(
      cudaMemcpy(device, zones.data(), bytes, cudaMemcpyHostToDevice));
  return GpuTimeZoneDatabase{
      static_cast<const GpuTimeZone*>(device),
      static_cast<int32_t>(zones.size()),
  };
}

} // namespace

// One database per device, kept for the life of the process: freeing it at
// exit would race CUDA's own teardown. Built under the lock, so concurrent
// first callers wait for one build rather than each tabulating every zone.
GpuTimeZoneDatabase gpuTimeZoneDatabase() {
  int device{0};
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  static std::mutex mutex;
  static auto* databases = new std::unordered_map<int, GpuTimeZoneDatabase>();
  const std::lock_guard<std::mutex> lock(mutex);

  if (const auto it = databases->find(device); it != databases->end()) {
    return it->second;
  }
  const auto database = buildDatabase();
  databases->emplace(device, database);
  return database;
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
