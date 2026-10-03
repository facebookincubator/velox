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

// Looks up every Velox time zone on the device and compares the local time it
// yields with tz::TimeZone on the host. Compiled with gpu_shadows/ ahead of
// the Velox source root, as the kernels that will carry the database are.

#include "velox/experimental/cudf/functions/GpuTimeZoneDatabase.h"
#include "velox/experimental/cudf/tests/MapOnDevice.h"

#include "velox/external/tzdb/time_zone.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

// The zone id a TIMESTAMP WITH TIME ZONE row carries and a UTC instant.
struct Lookup {
  int32_t timeZoneId;
  int64_t utcSeconds;
};

// Whether the id names a zone and, if so, the local wall-clock seconds.
struct LocalTime {
  int64_t localSeconds;
  bool present;
};

__global__ void lookupLocalTime(
    GpuTimeZoneDatabase database,
    const Lookup* lookups,
    LocalTime* out,
    int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const GpuTimeZone* zone = database.zone(lookups[i].timeZoneId);
  out[i].present = zone != nullptr;
  out[i].localSeconds =
      zone == nullptr ? 0 : zone->toLocal(lookups[i].utcSeconds);
}

// Resolves every lookup through the device table in one kernel launch.
std::vector<LocalTime> lookupOnDevice(const std::vector<Lookup>& lookups) {
  const auto database = gpuTimeZoneDatabase();
  return mapOnDevice<Lookup, LocalTime>(
      lookups, [database](const Lookup* in, LocalTime* out, int count) {
        lookupLocalTime<<<(count + 255) / 256, 256>>>(database, in, out, count);
      });
}

int64_t secondsSinceEpoch(int32_t year) {
  return std::chrono::duration_cast<std::chrono::seconds>(
             date::sys_days{date::year{year} / 1 / 1}.time_since_epoch())
      .count();
}

// Instants that stress one zone: the epoch and its neighbours, the ends of
// the span, one second and one hour either side of every offset change the
// zone records in the span, and a deterministic spread over it.
std::vector<int64_t> instants(
    const tz::TimeZone& zone,
    int32_t fromYear,
    int32_t toYear,
    int32_t numSpread) {
  const int64_t fromSeconds = secondsSinceEpoch(fromYear);
  const int64_t toSeconds = secondsSinceEpoch(toYear);
  std::vector<int64_t> values{-1, 0, 1, fromSeconds, toSeconds - 1};

  if (const auto* tzdbZone = zone.tz()) {
    auto info = tzdbZone->get_info(
        date::sys_seconds{
            std::chrono::seconds{fromSeconds},
        });
    while (info.end.time_since_epoch().count() < toSeconds) {
      const int64_t change = info.end.time_since_epoch().count();
      for (const int64_t delta : {-3'600, -1, 0, 1, 3'599}) {
        values.push_back(change + delta);
      }
      const auto next = tzdbZone->get_info(info.end);
      if (next.end <= info.end) {
        break;
      }
      info = next;
    }
  }

  uint64_t state = 0x9e37'79b9'7f4a'7c15;
  const auto span = static_cast<uint64_t>(toSeconds - fromSeconds);
  for (int32_t i = 0; i < numSpread; ++i) {
    state = state * 6'364'136'223'846'793'005ULL + 1'442'695'040'888'963'407ULL;
    values.push_back(fromSeconds + static_cast<int64_t>((state >> 11) % span));
  }
  return values;
}

int64_t hostLocalSeconds(const tz::TimeZone& zone, int64_t utcSeconds) {
  return zone.to_local(std::chrono::seconds{utcSeconds}).count();
}

// The whole id space of TIMESTAMP WITH TIME ZONE, in one kernel launch.
TEST(GpuTimeZoneDatabaseTest, everyZoneMatchesTheHost) {
  const auto ids = tz::getTimeZoneIDs();
  ASSERT_FALSE(ids.empty());

  // Fixed offsets are a single addition, so a few instants each suffice; a
  // named zone gets its transitions and a wider spread.
  std::vector<Lookup> lookups;
  for (const int16_t id : ids) {
    const auto* zone = tz::locateZone(id);
    const int32_t numSpread = zone->tz() == nullptr ? 8 : 512;
    for (const int64_t utcSeconds : instants(*zone, 1900, 2100, numSpread)) {
      lookups.push_back(Lookup{id, utcSeconds});
    }
  }

  const auto got = lookupOnDevice(lookups);
  ASSERT_EQ(got.size(), lookups.size());

  // Every lookup is compared, so the count below is the whole picture; only
  // the first few are spelled out.
  int32_t numMismatches{0};
  for (size_t i = 0; i < lookups.size(); ++i) {
    const auto* zone =
        tz::locateZone(static_cast<int16_t>(lookups[i].timeZoneId));
    const int64_t expected = hostLocalSeconds(*zone, lookups[i].utcSeconds);
    if (got[i].present && got[i].localSeconds == expected) {
      continue;
    }
    if (++numMismatches <= 20) {
      ADD_FAILURE() << zone->name() << " (id " << lookups[i].timeZoneId
                    << ") at " << lookups[i].utcSeconds << ": device "
                    << (got[i].present ? std::to_string(got[i].localSeconds)
                                       : "absent")
                    << ", host " << expected;
    }
  }
  EXPECT_EQ(numMismatches, 0);
}

// A zone id is twelve bits of a packed value, so a kernel can meet any id the
// Velox database never assigned: negative, beyond the table, or a hole in it.
TEST(GpuTimeZoneDatabaseTest, unknownIdIsAbsent) {
  const auto ids = tz::getTimeZoneIDs();
  const auto database = gpuTimeZoneDatabase();
  EXPECT_EQ(database.numZones, ids.back() + 1);

  std::vector<Lookup> lookups;
  for (int32_t id = 0; id <= ids.back(); ++id) {
    if (!std::binary_search(ids.begin(), ids.end(), id)) {
      lookups.push_back(Lookup{id, 0});
    }
  }
  const auto numHoles = lookups.size();
  for (const int32_t id :
       {-1,
        -32'768,
        database.numZones,
        database.numZones + 1,
        4'095,
        65'535,
        INT32_MAX}) {
    lookups.push_back(Lookup{id, 0});
  }

  const auto got = lookupOnDevice(lookups);
  for (size_t i = 0; i < lookups.size(); ++i) {
    SCOPED_TRACE(lookups[i].timeZoneId);
    EXPECT_FALSE(got[i].present);
  }
  // The Velox database leaves some ids unassigned, so holes were exercised.
  EXPECT_GT(numHoles, 0);
}

// The ids below 1681 are the offsets from -14:00 to +14:00 and never carry a
// table; the device must still apply the right sign and minutes.
TEST(GpuTimeZoneDatabaseTest, fixedOffsetIdsApplyTheirOffset) {
  const std::vector<std::pair<std::string, int64_t>> offsets{
      {"+00:00", 0},
      {"+00:01", 60},
      {"-00:01", -60},
      {"+05:30", 5 * 3'600 + 30 * 60},
      {"-08:00", -8 * 3'600},
      {"+14:00", 14 * 3'600},
      {"-14:00", -14 * 3'600},
  };
  const std::vector<int64_t> utcs{
      -2'000'000'000, -1, 0, 1, 86'399, 1'700'000'000, 4'000'000'000};

  std::vector<Lookup> lookups;
  for (const auto& [name, offset] : offsets) {
    const int16_t id = tz::getTimeZoneID(name);
    for (const int64_t utcSeconds : utcs) {
      lookups.push_back(Lookup{id, utcSeconds});
    }
  }

  const auto got = lookupOnDevice(lookups);
  size_t i{0};
  for (const auto& [name, offset] : offsets) {
    for (const int64_t utcSeconds : utcs) {
      SCOPED_TRACE(name + " at " + std::to_string(utcSeconds));
      EXPECT_TRUE(got[i].present);
      EXPECT_EQ(got[i].localSeconds - utcSeconds, offset);
      ++i;
    }
  }
}

// The table is built once per device and shared by every caller.
TEST(GpuTimeZoneDatabaseTest, builtOncePerDevice) {
  const auto first = gpuTimeZoneDatabase();
  const auto second = gpuTimeZoneDatabase();
  EXPECT_NE(first.zones, nullptr);
  EXPECT_EQ(first.zones, second.zones);
  EXPECT_EQ(first.numZones, second.numZones);
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
