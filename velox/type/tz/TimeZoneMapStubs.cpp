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

// Windows implementation for timezone functions.
// Builds a full timezone database from static data in TimeZoneDatabase.cpp.
// Named zones (IDs 1681+) use offset 0 (UTC) as fallback since
// velox_external_tzdb is not available on Windows.

#ifdef _WIN32

#include "velox/type/tz/TimeZoneMap.h"
#include "velox/common/base/Exceptions.h"
#include <boost/algorithm/string.hpp>
#include <unordered_map>
#include <unordered_set>

namespace facebook::velox::tz {

using TTimeZoneDatabase = std::vector<std::unique_ptr<TimeZone>>;
using TTimeZoneIndex = std::unordered_map<std::string, const TimeZone*>;

// Defined in TimeZoneDatabase.cpp
extern const std::vector<std::pair<int16_t, std::string>>& getTimeZoneEntries();
extern const std::vector<std::pair<std::string, std::string>>&
getShortNameTimeZoneMapping();

namespace {

inline std::chrono::minutes getTimeZoneOffset(int16_t tzID) {
  return std::chrono::minutes{(tzID <= 840) ? (tzID - 841) : (tzID - 840)};
}

TTimeZoneDatabase buildTimeZoneDatabase(
    const std::vector<std::pair<int16_t, std::string>>& dbInput) {
  TTimeZoneDatabase tzDatabase;
  tzDatabase.resize(dbInput.back().first + 1);

  for (const auto& entry : dbInput) {
    std::unique_ptr<TimeZone> timeZonePtr;
    if (entry.first == 0) {
      // UTC
      timeZonePtr = std::make_unique<TimeZone>(
          "UTC", entry.first, std::chrono::minutes(0));
    } else if (entry.first <= 1680) {
      // Offset-based timezone
      std::chrono::minutes offset = getTimeZoneOffset(entry.first);
      timeZonePtr =
          std::make_unique<TimeZone>(entry.second, entry.first, offset);
    } else {
      // Named timezone - use UTC offset as fallback (no tzdb on Windows)
      timeZonePtr = std::make_unique<TimeZone>(
          entry.second, entry.first, std::chrono::minutes(0));
    }
    tzDatabase[entry.first] = std::move(timeZonePtr);
  }
  return tzDatabase;
}

const TTimeZoneDatabase& getTimeZoneDatabase() {
  static TTimeZoneDatabase timeZoneDatabase =
      buildTimeZoneDatabase(getTimeZoneEntries());
  return timeZoneDatabase;
}

TTimeZoneIndex buildTimeZoneIndex(const TTimeZoneDatabase& tzDatabase) {
  TTimeZoneIndex reversed;
  reversed.reserve(tzDatabase.size() + 2);
  for (int16_t i = 0; i < static_cast<int16_t>(tzDatabase.size()); ++i) {
    if (tzDatabase[i] != nullptr) {
      reversed.emplace(
          boost::algorithm::to_lower_copy(tzDatabase[i]->name()),
          tzDatabase[i].get());
    }
  }
  reversed.emplace("+00:00", tzDatabase.front().get());
  reversed.emplace("-00:00", tzDatabase.front().get());
  return reversed;
}

const TTimeZoneIndex& getTimeZoneIndex() {
  static TTimeZoneIndex timeZoneIndex =
      buildTimeZoneIndex(getTimeZoneDatabase());
  return timeZoneIndex;
}

inline bool isDigit(char c) {
  return c >= '0' && c <= '9';
}

inline bool startsWith(std::string_view str, const char* prefix) {
  return str.rfind(prefix, 0) == 0;
}

inline bool isTimeZoneOffset(std::string_view str) {
  return str.size() >= 3 && (str[0] == '+' || str[0] == '-');
}

inline bool isUtcEquivalentName(std::string_view zone) {
  static std::unordered_set<std::string> utcSet = {
      "utc", "uct", "gmt", "gmt0", "greenwich", "universal", "zulu", "z"};
  return utcSet.find(std::string(zone)) != utcSet.end();
}

std::string normalizeTimeZoneOffset(const std::string& zoneOffset) {
  if (zoneOffset.size() == 3 && isDigit(zoneOffset[1]) &&
      isDigit(zoneOffset[2])) {
    return zoneOffset + ":00";
  } else if (
      zoneOffset.size() == 5 && isDigit(zoneOffset[1]) &&
      isDigit(zoneOffset[2]) && isDigit(zoneOffset[3]) &&
      isDigit(zoneOffset[4])) {
    return zoneOffset.substr(0, 3) + ':' + zoneOffset.substr(3, 2);
  }
  return zoneOffset;
}

std::string normalizeTimeZone(const std::string& originalZoneId) {
  if (isTimeZoneOffset(originalZoneId)) {
    return normalizeTimeZoneOffset(originalZoneId);
  }
  std::string_view zoneId = originalZoneId;
  const bool startsWithEtc = startsWith(zoneId, "etc/");
  if (startsWithEtc) {
    zoneId = zoneId.substr(4);
  }
  if (isUtcEquivalentName(zoneId)) {
    return "utc";
  }
  bool startsWithUtc = startsWith(zoneId, "utc");
  bool startsWithGmt = startsWith(zoneId, "gmt");
  bool startsWithUt = !startsWithUtc && startsWith(zoneId, "ut");
  if ((zoneId.size() > 4 && (startsWithUtc || startsWithGmt)) ||
      (zoneId.size() > 3 && startsWithUt)) {
    if (startsWithUtc || startsWithGmt) {
      zoneId = zoneId.substr(3);
    } else {
      zoneId = zoneId.substr(2);
    }
    char signChar = zoneId[0];
    if (signChar == '+' || signChar == '-') {
      if (startsWithEtc && startsWithGmt) {
        signChar = (signChar == '-') ? '+' : '-';
      }
      char hourTens;
      char hourOnes;
      if (zoneId.size() == 2) {
        hourTens = '0';
        hourOnes = zoneId[1];
      } else {
        hourTens = zoneId[1];
        hourOnes = zoneId[2];
      }
      if (hourTens == '0' && hourOnes == '0') {
        return "utc";
      }
      if (isDigit(hourTens) && isDigit(hourOnes)) {
        return std::string() + signChar + hourTens + hourOnes + ":00";
      }
    }
  }
  return originalZoneId;
}

// Simplified validateRange to avoid MSVC ICE with date:: templates.
// Validates the timestamp is within a reasonable range without using
// date::year_month_day which triggers compiler bugs.
template <typename TDuration>
void validateRangeImpl(time_point<TDuration> timePoint) {
  // Check if timepoint seconds are within safe range for millisecond conversion
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
      timePoint.time_since_epoch());
  constexpr int64_t kMinSeconds = std::numeric_limits<int64_t>::min() / 1000;
  constexpr int64_t kMaxSeconds = std::numeric_limits<int64_t>::max() / 1000;

  if (seconds.count() < kMinSeconds || seconds.count() > kMaxSeconds) {
    VELOX_USER_FAIL(
        "Timepoint is outside of supported timestamp seconds since epoch range: [{}, {}], got {}",
        kMinSeconds,
        kMaxSeconds,
        static_cast<int64_t>(timePoint.time_since_epoch().count()));
  }

  // Approximate year check without date:: templates (avoids MSVC ICE).
  // 365.25 days/year * 86400 seconds/day ≈ 31557600 seconds/year
  constexpr int64_t kSecondsPerYear = 31557600LL;
  int64_t approxYear = 1970 + seconds.count() / kSecondsPerYear;
  // year::min() = -32767, year::max() = 32767
  constexpr int64_t kMinYear = -32767;
  constexpr int64_t kMaxYear = 32767;

  if (approxYear < kMinYear || approxYear > kMaxYear) {
    VELOX_USER_FAIL(
        "Timepoint is outside of supported year range: [{}, {}], got {}",
        kMinYear,
        kMaxYear,
        approxYear);
  }
}

} // namespace

void validateRange(time_point<std::chrono::seconds> timePoint) {
  validateRangeImpl(timePoint);
}

void validateRange(time_point<std::chrono::milliseconds> timePoint) {
  validateRangeImpl(timePoint);
}

std::string getTimeZoneName(int64_t timeZoneID) {
  return locateZone(static_cast<int16_t>(timeZoneID), true)->name();
}

const TimeZone* locateZone(int16_t timeZoneID, bool failOnError) {
  const auto& timeZoneDatabase = getTimeZoneDatabase();
  if (timeZoneID >= static_cast<int16_t>(timeZoneDatabase.size()) ||
      timeZoneDatabase[timeZoneID] == nullptr) {
    if (failOnError) {
      VELOX_FAIL("Unable to resolve timeZoneID '{}'", timeZoneID);
    }
    return nullptr;
  }
  return timeZoneDatabase[timeZoneID].get();
}

const TimeZone* locateZone(std::string_view timeZone, bool failOnError) {
  const auto& timeZoneIndex = getTimeZoneIndex();

  if (timeZone.size() == 3) {
    const auto& shortNameTimeZoneMapping = getShortNameTimeZoneMapping();
    for (const auto& entry : shortNameTimeZoneMapping) {
      if (entry.first == timeZone) {
        timeZone = entry.second;
        break;
      }
    }
  }

  std::string timeZoneLowered;
  boost::algorithm::to_lower_copy(
      std::back_inserter(timeZoneLowered), timeZone);

  auto it = timeZoneIndex.find(timeZoneLowered);
  if (it != timeZoneIndex.end()) {
    return it->second;
  }

  it = timeZoneIndex.find(normalizeTimeZone(timeZoneLowered));
  if (it != timeZoneIndex.end()) {
    return it->second;
  }

  if (failOnError) {
    VELOX_USER_FAIL("Unknown time zone: '{}'", timeZone);
  }
  return nullptr;
}

int16_t getTimeZoneID(std::string_view timeZone, bool failOnError) {
  const TimeZone* tz = locateZone(timeZone, failOnError);
  return tz == nullptr ? -1 : tz->id();
}

int16_t getTimeZoneID(int32_t offsetMinutes) {
  static constexpr int32_t kMinOffset = -14 * 60;
  static constexpr int32_t kMaxOffset = 14 * 60;

  if (offsetMinutes == 0) {
    return 0;
  }

  VELOX_USER_CHECK_LE(
      kMinOffset,
      offsetMinutes,
      "Invalid timezone offset minutes: {}",
      offsetMinutes);
  VELOX_USER_CHECK_LE(
      offsetMinutes,
      kMaxOffset,
      "Invalid timezone offset minutes: {}",
      offsetMinutes);

  if (offsetMinutes < 0) {
    return 1 + (offsetMinutes - kMinOffset);
  } else {
    return offsetMinutes - kMinOffset;
  }
}

std::vector<int16_t> getTimeZoneIDs() {
  const auto& timeZoneDatabase = getTimeZoneDatabase();
  std::vector<int16_t> ids;
  ids.reserve(timeZoneDatabase.size());
  for (int16_t i = 0; i < static_cast<int16_t>(timeZoneDatabase.size()); ++i) {
    if (timeZoneDatabase[i] != nullptr) {
      ids.push_back(i);
    }
  }
  return ids;
}

// TimeZone method implementations - offset-based only (no DST on Windows)
TimeZone::seconds TimeZone::to_sys(seconds timestamp, TChoose) const {
  return timestamp - offset_;
}

TimeZone::milliseconds TimeZone::to_sys(milliseconds timestamp, TChoose) const {
  return timestamp - std::chrono::duration_cast<milliseconds>(offset_);
}

TimeZone::seconds TimeZone::to_local(seconds timestamp) const {
  return timestamp + offset_;
}

TimeZone::milliseconds TimeZone::to_local(milliseconds timestamp) const {
  return timestamp + std::chrono::duration_cast<milliseconds>(offset_);
}

TimeZone::seconds TimeZone::correct_nonexistent_time(seconds timestamp) const {
  return timestamp;
}

std::string TimeZone::getShortName(milliseconds, TChoose) const {
  return timeZoneName_;
}

std::string TimeZone::getLongName(milliseconds, TChoose) const {
  return timeZoneName_;
}

} // namespace facebook::velox::tz

#endif // _WIN32
