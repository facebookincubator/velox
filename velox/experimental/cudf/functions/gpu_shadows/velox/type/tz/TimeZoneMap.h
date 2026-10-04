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

// GPU shadow for velox/type/tz/TimeZoneMap.h.
//
// On the device a tz::TimeZone is a GpuTimeZone in device memory. The host
// builds one per Velox zone from the database the CPU reads, hands a function
// struct its pointer in initialize(), and the conversions the real bodies make
// through to_local(), to_sys(), correct_nonexistent_time() and
// Timestamp::toGMT() and toTimezone() run over its table. locateZone(id) reads
// a device copy of the whole database, which each translation unit owns and
// uploads to a device before its first kernel there that may resolve a zone.
//
// The host side of a shadow translation unit must not resolve a zone through
// this header. The inline functions here share their host symbols with the
// real library's, and the linker keeps one of the two; initialize() reaches
// zones through GpuTimeZone.h, whose helpers have names of their own.
#pragma once

#include "velox/experimental/cudf/functions/GpuTimeZone.h"
#include "velox/experimental/cudf/functions/GpuTimeZoneDatabase.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Macros.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace facebook::velox::tz {

class TimeZone;

/// The zone with the given id, read from the device database. For an id the
/// database leaves empty: null when failOnError is false, otherwise the runtime
/// error tz::locateZone() raises on the CPU, and UTC so that the declined row
/// still dereferences a zone. Null on the host, which never resolves a zone by
/// id on this side.
VELOX_GPU_COMPATIBLE const TimeZone* locateZone(
    int16_t timeZoneID,
    bool failOnError = true);

/// The id of a fixed offset, as tz::getTimeZoneID(int32_t) assigns it. The
/// host asks the real function, so a constant offset it rejects raises its
/// error in initialize(); the device applies the same formula and declines the
/// row outside the range.
VELOX_GPU_COMPATIBLE int16_t getTimeZoneID(int32_t offsetMinutes);

/// The id of a zone name. The host asks the real function. A kernel cannot
/// read a strings column, so on the device this only declines the row: every
/// registered signature takes the name as a constant, which initialize()
/// resolves on the host.
VELOX_GPU_COMPATIBLE int16_t
getTimeZoneID(std::string_view timeZone, bool failOnError = true);

// Host only, as on the CPU, declared so that the host-only bodies in the real
// headers parse. A call links to the real function, whose result names a host
// object no kernel can read.
const TimeZone* locateZone(std::string_view timeZone, bool failOnError = true);
std::string getTimeZoneName(int64_t timeZoneID);
std::vector<int16_t> getTimeZoneIDs();

/// A Velox time zone under the name the real headers use. Never constructed on
/// this side: the host places a GpuTimeZone in device memory, and TimeZone adds
/// no state, so a pointer to one is read as a pointer to the other.
class TimeZone final : public cudf_velox::gpu_sfi::GpuTimeZone {
 public:
  using seconds = std::chrono::seconds;
  using milliseconds = std::chrono::milliseconds;

  enum class TChoose {
    kFail = 0,
    kEarliest = 1,
    kLatest = 2,
  };

  TimeZone() = delete;
  TimeZone(const TimeZone&) = delete;
  TimeZone& operator=(const TimeZone&) = delete;

  /// The local wall-clock time of an instant. A fraction of a second is
  /// carried over whole, so the last second before an offset change reads
  /// with the offset before it, as the seconds form reads it on the CPU.
  VELOX_GPU_COMPATIBLE seconds to_local(seconds timestamp) const {
    return seconds(toLocal(timestamp.count()));
  }
  VELOX_GPU_COMPATIBLE milliseconds to_local(milliseconds timestamp) const {
    return milliseconds(toLocalMillis(timestamp.count()));
  }

  /// The instant of a local wall-clock time. A time an offset decrease
  /// repeats resolves to the earlier instant, which is kEarliest's and what
  /// toSysChecked() picks; no registered struct passes kFail, under which the
  /// CPU raises for a repeated time, and none passes kLatest, which declines
  /// the row. A time an offset increase skipped is the user error
  /// toSysChecked() raises, under every choice: the CPU returns the gap's
  /// first instant under kEarliest, which the table lookup does not report,
  /// and the registered structs move such a time past the gap through
  /// correct_nonexistent_time() before converting it.
  VELOX_GPU_COMPATIBLE seconds
  to_sys(seconds timestamp, TChoose choose = TChoose::kFail) const {
    VELOX_CHECK(
        choose != TChoose::kLatest, "to_sys(kLatest) has no device form");
    const UtcInstant instant = toUtc(timestamp.count());
    VELOX_USER_CHECK(
        instant.exists, "Local time is in a gap: {}", timestamp.count());
    return seconds(instant.utcSeconds);
  }
  VELOX_GPU_COMPATIBLE milliseconds
  to_sys(milliseconds timestamp, TChoose choose = TChoose::kFail) const {
    VELOX_CHECK(
        choose != TChoose::kLatest, "to_sys(kLatest) has no device form");
    const UtcMillis instant = toUtcMillis(timestamp.count());
    VELOX_USER_CHECK(
        instant.exists, "Local time is in a gap: {}", timestamp.count());
    return milliseconds(instant.utcMillis);
  }

  /// A local time an offset increase skipped, moved past the gap by the size
  /// of the increase; any other local time unchanged.
  VELOX_GPU_COMPATIBLE seconds
  correct_nonexistent_time(seconds timestamp) const {
    return seconds(correctNonexistent(timestamp.count()));
  }

  /// to_local(), as Timestamp::toTimezone() reads it.
  VELOX_GPU_COMPATIBLE seconds toLocalChecked(seconds timestamp) const {
    return to_local(timestamp);
  }

  /// to_sys() resolving a repeated time to the earlier instant and raising for
  /// a skipped one, as Timestamp::toGMT() reads it.
  VELOX_GPU_COMPATIBLE seconds toSysChecked(seconds timestamp) const {
    return to_sys(timestamp, TChoose::kEarliest);
  }

  // Host-only members of the real class, declared so that the host-only bodies
  // naming them parse. None is defined: a call fails to link rather than run
  // against a zone this side cannot see.
  const std::string& name() const;
  int16_t id() const;
  std::optional<std::chrono::minutes> offset() const;
  std::string getShortName(
      milliseconds timestamp,
      TChoose choose = TChoose::kFail) const;
  std::string getLongName(
      milliseconds timestamp,
      TChoose choose = TChoose::kFail) const;
};

namespace gpu_shadow_detail {
namespace {

// One copy per translation unit: whole-program device compilation gives each
// unit its own device symbols, and the kernels that read it are instantiated
// in the unit that uploads it. Unused in a unit that launches no time zone
// function.
[[maybe_unused]] __device__ cudf_velox::gpu_sfi::GpuTimeZoneDatabase
    deviceTimeZoneDatabase;

// Zero-initialized: a fixed offset of zero, which is UTC.
[[maybe_unused]] __device__ cudf_velox::gpu_sfi::GpuTimeZone deviceUtcTimeZone;

// Uploads the database to the current device unless this unit already did,
// and returns whether this call uploaded.
//
// A __device__ variable is one allocation per device, so a copy made when the
// functions are registered reaches only the device current at that moment; on
// any other the view stays zero and every lookup reports its zone absent. The
// adapter calls this instead before each launch that may resolve a zone, on
// the thread whose current device the launch stream belongs to, so a device
// receives the view before its first lookup and a repeat costs a lock and a
// set lookup. The copy goes down the launch stream and is waited for before
// the device is recorded as holding the view, so that a kernel another thread
// then launches on another stream of the device reads the view, not the zero.
[[maybe_unused]] bool uploadDeviceTimeZoneDatabase(cudaStream_t stream) {
  static std::mutex mutex;
  static std::unordered_set<int> uploadedDevices;

  int device{0};
  auto status = cudaGetDevice(&device);
  VELOX_CHECK_EQ(
      status,
      cudaSuccess,
      "Reading the current device failed: {}",
      cudaGetErrorString(status));

  const std::lock_guard<std::mutex> lock(mutex);
  if (uploadedDevices.count(device) > 0) {
    return false;
  }
  const auto database = cudf_velox::gpu_sfi::gpuTimeZoneDatabase();
  status = cudaMemcpyToSymbolAsync(
      deviceTimeZoneDatabase,
      &database,
      sizeof(database),
      0,
      cudaMemcpyHostToDevice,
      stream);
  VELOX_CHECK_EQ(
      status,
      cudaSuccess,
      "Uploading the device time zone database failed: {}",
      cudaGetErrorString(status));
  status = cudaStreamSynchronize(stream);
  VELOX_CHECK_EQ(
      status,
      cudaSuccess,
      "Uploading the device time zone database failed: {}",
      cudaGetErrorString(status));
  uploadedDevices.insert(device);
  return true;
}

} // namespace
} // namespace gpu_shadow_detail

VELOX_GPU_COMPATIBLE inline const TimeZone* locateZone(
    int16_t timeZoneID,
    bool failOnError) {
#ifdef __CUDA_ARCH__
  const auto* zone = gpu_shadow_detail::deviceTimeZoneDatabase.zone(timeZoneID);
  if (zone != nullptr) {
    return static_cast<const TimeZone*>(zone);
  }
  if (!failOnError) {
    return nullptr;
  }
  // As TimeZoneMap.cpp raises it: a runtime error, which no TRY swallows.
  VELOX_FAIL("Unable to resolve timeZoneID '{}'", timeZoneID);
  return static_cast<const TimeZone*>(&gpu_shadow_detail::deviceUtcTimeZone);
#else
  (void)timeZoneID;
  (void)failOnError;
  return nullptr;
#endif
}

VELOX_GPU_COMPATIBLE inline int16_t getTimeZoneID(int32_t offsetMinutes) {
#ifdef __CUDA_ARCH__
  VELOX_USER_CHECK_LE(
      cudf_velox::gpu_sfi::kMinFixedOffsetMinutes,
      offsetMinutes,
      "Invalid timezone offset minutes: {}",
      offsetMinutes);
  VELOX_USER_CHECK_LE(
      offsetMinutes,
      cudf_velox::gpu_sfi::kMaxFixedOffsetMinutes,
      "Invalid timezone offset minutes: {}",
      offsetMinutes);
  return cudf_velox::gpu_sfi::fixedOffsetTimeZoneId(offsetMinutes);
#else
  return cudf_velox::gpu_sfi::gpuTimeZoneId(offsetMinutes);
#endif
}

VELOX_GPU_COMPATIBLE inline int16_t getTimeZoneID(
    std::string_view timeZone,
    bool failOnError) {
#ifdef __CUDA_ARCH__
  (void)timeZone;
  (void)failOnError;
  VELOX_USER_FAIL("A zone name cannot be resolved on the device");
  return 0;
#else
  return cudf_velox::gpu_sfi::gpuTimeZoneId(timeZone, failOnError);
#endif
}

} // namespace facebook::velox::tz
