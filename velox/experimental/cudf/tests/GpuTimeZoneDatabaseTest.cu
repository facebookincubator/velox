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

// Compares the device time zone table with tz::TimeZone on the host, for every
// id a TIMESTAMP WITH TIME ZONE can carry. Compiled against the real time zone
// headers, so the host half of each comparison is the real library; the
// kernels read the table the shadow tz::locateZone() reads. The conversions
// themselves are held to the CPU function by function in GpuSfiTimestampTest.

#include "velox/experimental/cudf/functions/GpuTimeZoneDatabase.h"
#include "velox/experimental/cudf/tests/MapOnDevice.h"

#include "velox/external/tzdb/time_zone.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
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

// A count of milliseconds against one zone, read as a UTC instant and as a
// local wall-clock time.
struct Conversion {
  int32_t timeZoneId;
  int64_t millis;
};

// The local reading of the instant, the instant of the local time with whether
// its second exists, and the local second moved past any gap.
struct Converted {
  int64_t localMillis;
  int64_t utcMillis;
  bool exists;
  int64_t correctedSeconds;
};

__global__ void convertOnDevice(
    GpuTimeZoneDatabase database,
    const Conversion* conversions,
    Converted* out,
    int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const GpuTimeZone* zone = database.zone(conversions[i].timeZoneId);
  const int64_t millis = conversions[i].millis;
  out[i].localMillis = zone->toLocalMillis(millis);
  const auto instant = zone->toUtcMillis(millis);
  out[i].utcMillis = instant.utcMillis;
  out[i].exists = instant.exists;
  out[i].correctedSeconds = zone->correctNonexistent(floorSeconds(millis));
}

__global__ void
fixedOffsetIds(const int32_t* offsets, int16_t* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  out[i] = fixedOffsetTimeZoneId(offsets[i]);
}

int64_t secondsSinceEpoch(int32_t year) {
  return std::chrono::duration_cast<std::chrono::seconds>(
             date::sys_days{date::year{year} / 1 / 1}.time_since_epoch())
      .count();
}

// Every zone the Velox database assigns, at the epoch and at the two ends of
// the years the tables cover, so a zone's earliest offset, its current one and
// its last tabulated rule are all read. In one kernel launch.
TEST(GpuTimeZoneDatabaseTest, everyZoneMatchesTheHost) {
  const auto ids = tz::getTimeZoneIDs();
  ASSERT_FALSE(ids.empty());
  const std::vector<int64_t> instants{
      secondsSinceEpoch(1900), 0, secondsSinceEpoch(2799)};

  std::vector<Lookup> lookups;
  for (const int16_t id : ids) {
    for (const int64_t utcSeconds : instants) {
      lookups.push_back(Lookup{id, utcSeconds});
    }
  }

  const auto got = lookupOnDevice(lookups);
  ASSERT_EQ(got.size(), lookups.size());
  for (size_t i = 0; i < lookups.size(); ++i) {
    const auto* zone =
        tz::locateZone(static_cast<int16_t>(lookups[i].timeZoneId));
    SCOPED_TRACE(zone->name() + " at " + std::to_string(lookups[i].utcSeconds));
    EXPECT_TRUE(got[i].present);
    EXPECT_EQ(
        got[i].localSeconds,
        zone->to_local(std::chrono::seconds{lookups[i].utcSeconds}).count());
  }
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

// The millisecond conversions and the gap correction, which the registered
// date arithmetic reaches through the shadow tz::TimeZone, against the real
// one: a millisecond either side of every offset change from 1970 to 2030,
// read as an instant and as a local time, so that the last second before a
// change, a time inside a gap and a time inside an overlap are each read. A
// gap holds no instant, so the instant is compared only where the host finds
// one. Instants after the epoch alone, since the vendored tzdb truncates a
// negative sub-second time point toward zero where the device rounds it down;
// whole seconds before the epoch complete the rows.
TEST(GpuTimeZoneDatabaseTest, millisecondConversionsMatchTheHost) {
  std::vector<Conversion> conversions;
  std::vector<std::string> traces;
  const auto add = [&](const std::string& zone, int16_t id, int64_t millis) {
    conversions.push_back(Conversion{id, millis});
    traces.push_back(zone + " at " + std::to_string(millis));
  };
  for (const auto* name :
       {"America/Los_Angeles",
        "Europe/London",
        "Australia/Lord_Howe",
        "America/Sao_Paulo",
        "Pacific/Apia",
        "Asia/Kathmandu",
        "+05:30"}) {
    const auto* zone = tz::locateZone(name);
    const auto id = zone->id();
    for (const int64_t whole : {-86'400'000LL, -1'000LL, 0LL, 1'000LL}) {
      add(name, id, whole);
    }
    if (zone->tz() == nullptr) {
      add(name, id, 1'700'000'000'250);
      continue;
    }
    auto info = zone->tz()->get_info(
        date::sys_seconds{std::chrono::seconds{secondsSinceEpoch(1970)}});
    const int64_t endSeconds = secondsSinceEpoch(2030);
    while (info.end.time_since_epoch().count() < endSeconds) {
      const int64_t change = info.end.time_since_epoch().count();
      const int64_t before = info.offset.count();
      const int64_t after = zone->tz()->get_info(info.end).offset.count();
      for (const int64_t delta : {-1'000LL, -1LL, 0LL, 1LL, 999LL}) {
        // As an instant, and as the local time either offset gives it.
        add(name, id, change * 1'000 + delta);
        add(name, id, (change + before) * 1'000 + delta);
        add(name, id, (change + after) * 1'000 + delta);
      }
      // Half way through the gap or the overlap, with a fraction.
      add(name, id, (change + (before + after) / 2) * 1'000 + 250);
      info = zone->tz()->get_info(info.end);
    }
  }
  // The named zones recorded offset changes in the span.
  ASSERT_GT(conversions.size(), 1'000);

  const auto database = gpuTimeZoneDatabase();
  const auto got = mapOnDevice<Conversion, Converted>(
      conversions, [database](const Conversion* in, Converted* out, int count) {
        convertOnDevice<<<(count + 255) / 256, 256>>>(database, in, out, count);
      });
  ASSERT_EQ(got.size(), conversions.size());
  for (size_t i = 0; i < conversions.size(); ++i) {
    SCOPED_TRACE(traces[i]);
    const auto* zone =
        tz::locateZone(static_cast<int16_t>(conversions[i].timeZoneId));
    const std::chrono::milliseconds millis{conversions[i].millis};
    const std::chrono::seconds seconds{floorSeconds(conversions[i].millis)};
    EXPECT_EQ(got[i].localMillis, zone->to_local(millis).count());
    EXPECT_EQ(
        got[i].correctedSeconds,
        zone->correct_nonexistent_time(seconds).count());
    const bool hostExists = zone->tz() == nullptr ||
        zone->tz()->get_info(date::local_seconds{seconds}).result !=
            tzdb::local_info::nonexistent;
    EXPECT_EQ(got[i].exists, hostExists);
    if (hostExists) {
      EXPECT_EQ(
          got[i].utcMillis,
          zone->to_sys(millis, tz::TimeZone::TChoose::kEarliest).count());
    }
  }
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

// The device derives a zone id from an offset column with its own formula,
// which has to assign every offset the id tz::getTimeZoneID(int32_t) assigns
// and stop exactly where it stops.
TEST(GpuTimeZoneDatabaseTest, fixedOffsetIdFormulaMatchesTheHost) {
  std::vector<int32_t> offsets;
  for (int32_t offset = kMinFixedOffsetMinutes;
       offset <= kMaxFixedOffsetMinutes;
       ++offset) {
    offsets.push_back(offset);
  }
  const auto got = mapOnDevice<int32_t, int16_t>(
      offsets, [](const int32_t* in, int16_t* out, int count) {
        fixedOffsetIds<<<(count + 255) / 256, 256>>>(in, out, count);
      });
  for (size_t i = 0; i < offsets.size(); ++i) {
    SCOPED_TRACE(offsets[i]);
    EXPECT_EQ(got[i], tz::getTimeZoneID(offsets[i]));
  }
  // Velox's exception headers do not parse under nvcc, so only the throw is
  // checked here; FilterProjectTest holds the message to the CPU's.
  EXPECT_THROW(tz::getTimeZoneID(kMinFixedOffsetMinutes - 1), std::exception);
  EXPECT_THROW(tz::getTimeZoneID(kMaxFixedOffsetMinutes + 1), std::exception);
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
