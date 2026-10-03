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

#include "velox/experimental/cudf/functions/GpuTimeZone.h"

#include "velox/core/QueryConfig.h"
#include "velox/external/tzdb/time_zone.h"
#include "velox/functions/lib/TimeUtils.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

struct HostTable {
  std::vector<int64_t> transitions;
  std::vector<int32_t> offsets;
};

// Walks the zone's offset history up to GpuTimeZone::kTableEndSeconds, keeping
// only changes of total offset, the only part that moves local time.
HostTable tabulate(const tzdb::time_zone& zone) {
  HostTable table;
  const auto append = [&](int64_t begin, std::chrono::seconds offset) {
    const auto offsetSeconds = static_cast<int32_t>(offset.count());
    if (!table.offsets.empty() && table.offsets.back() == offsetSeconds) {
      return;
    }
    table.transitions.push_back(begin);
    table.offsets.push_back(offsetSeconds);
  };

  // Year 1 precedes every transition in the database, so this is the zone's
  // earliest offset, which applies back to the beginning of time.
  auto info = zone.get_info(date::sys_days{date::year{1} / 1 / 1});
  append(std::numeric_limits<int64_t>::min(), info.offset);
  while (info.end.time_since_epoch().count() < GpuTimeZone::kTableEndSeconds) {
    const auto next = info.end;
    info = zone.get_info(next);
    append(info.begin.time_since_epoch().count(), info.offset);
    if (info.end <= next) {
      break;
    }
  }
  return table;
}

struct DeviceTable {
  const int64_t* transitions;
  const int32_t* offsets;
  int32_t size;
};

template <typename T>
T* copyToDevice(const std::vector<T>& values) {
  void* device{nullptr};
  CUDF_CUDA_TRY(cudaMalloc(&device, values.size() * sizeof(T)));
  CUDF_CUDA_TRY(cudaMemcpy(
      device,
      values.data(),
      values.size() * sizeof(T),
      cudaMemcpyHostToDevice));
  return static_cast<T*>(device);
}

// One table per (device, zone), built on first use and kept for the life of
// the process: a few tens of kilobytes each, and freeing them at exit would
// race CUDA's own teardown.
DeviceTable deviceTable(const tz::TimeZone& zone) {
  int device{0};
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  static std::mutex mutex;
  static auto* tables = new std::unordered_map<std::string, DeviceTable>();
  const std::lock_guard<std::mutex> lock(mutex);

  auto key = std::to_string(device) + ":" + zone.name();
  if (const auto it = tables->find(key); it != tables->end()) {
    return it->second;
  }
  const auto table = tabulate(*zone.tz());
  const DeviceTable deviceTable{
      copyToDevice(table.transitions),
      copyToDevice(table.offsets),
      static_cast<int32_t>(table.transitions.size())};
  tables->emplace(std::move(key), deviceTable);
  return deviceTable;
}

} // namespace

GpuTimeZone gpuSessionTimeZone(const core::QueryConfig& config) {
  const tz::TimeZone* zone = functions::getTimeZoneFromConfig(config);
  if (zone == nullptr) {
    return GpuTimeZone{};
  }
  return gpuTimeZone(*zone);
}

GpuTimeZone gpuTimeZone(const tz::TimeZone& zone) {
  if (zone.tz() == nullptr) {
    GpuTimeZone fixed;
    fixed.fixedOffset = static_cast<int32_t>(
        std::chrono::seconds(zone.offset().value()).count());
    return fixed;
  }
  const auto table = deviceTable(zone);
  GpuTimeZone named;
  named.transitions = table.transitions;
  named.offsets = table.offsets;
  named.numTransitions = table.size;
  return named;
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
