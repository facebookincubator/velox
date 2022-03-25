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

#include <iomanip>
#include <sstream>
#include "cmath"

#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox {
static constexpr std::string_view kTimeUnits[] = {"ns", "us", "ms", "s"};
static constexpr uint64_t kTimeUnitsInSecond[] = {
    1'000'000'000,
    1'000'000,
    1'000,
    1};
static constexpr std::string_view kByteUnits[] = {"B", "KB", "MB", "GB", "TB"};
static int kTimeScale = 1'000;
static int kByteScale = 1'024;
static int kSecondsInMinute = 60;
static int kSecondsInHour = 60 * kSecondsInMinute;
static int kSecondsInDay = 24 * kSecondsInHour;

namespace {
// Print the input time in seconds as the most appropriate units
// and return a string value.
// Possible units are minutes(m), hours(h), days(d), seconds(s).
std::string succinctSeconds(uint64_t seconds) {
  std::stringstream out;
  int days = seconds / kSecondsInDay;
  if (days) {
    out << days << "d";
  }
  seconds -= days * kSecondsInDay;

  int hours = seconds / kSecondsInHour;
  if (days || hours) {
    out << hours << "h";
  }
  seconds -= hours * kSecondsInHour;

  int minutes = seconds / kSecondsInMinute;
  if (days || hours || minutes) {
    out << minutes << "m";
  }
  seconds -= minutes * kSecondsInMinute;
  out << seconds << "s";
  return out.str();
}
// Match the input 'value' to the most appropriate unit and return
// a string value. The units are specified in the 'units' array.
// unitOffset is used to skip the starting units.
// unitScale is used to determine the unit.
// precision is used to set the decimal digits in the final output.
std::string succinctPrint(
    uint64_t value,
    const std::string_view* units,
    int unitsSize,
    int unitOffset,
    int unitScale,
    int precision) {
  std::stringstream out;
  int offset = unitOffset;
  double decimalValue = static_cast<double>(value);
  while ((decimalValue / unitScale) >= 1 && offset < (unitsSize - 1)) {
    decimalValue = decimalValue / unitScale;
    offset++;
  }
  if (offset == unitOffset) {
    // Print the default value.
    precision = 0;
  }
  out << std::fixed << std::setprecision(precision) << decimalValue
      << units[offset];
  return out.str();
}

std::string succinctTime(uint64_t time, int unitOffset, int precision) {
  // Print time as days, hours, minutes, seconds if time is more than a minute.
  if (time > (kSecondsInMinute * kTimeUnitsInSecond[unitOffset])) {
    uint64_t seconds =
        std::round((time * 1.0) / kTimeUnitsInSecond[unitOffset]);
    return succinctSeconds(seconds);
  }
  return succinctPrint(
      time,
      &kTimeUnits[0],
      sizeof(kTimeUnits) / sizeof(std::string_view),
      unitOffset,
      kTimeScale,
      precision);
}
} // namespace

std::string succinctMillis(uint64_t time, int precision) {
  return succinctTime(time, 2, precision);
}

std::string succinctNanos(uint64_t time, int precision) {
  return succinctTime(time, 0, precision);
}

std::string succinctBytes(uint64_t bytes, int precision) {
  return succinctPrint(
      bytes,
      &kByteUnits[0],
      sizeof(kByteUnits) / sizeof(std::string_view),
      0,
      kByteScale,
      precision);
}

} // namespace facebook::velox
