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

#include "velox/common/base/SuccinctPrintUtil.h"

namespace facebook::velox::print {
static constexpr std::string_view timeUnits[4] = {"ns", "us", "ms", "s"};
static constexpr std::string_view byteUnits[4] = {"B", "KB", "MB", "GB"};
static int kTimeScale = 1000;
static int kByteScale = 1024;

namespace {
std::string succinctPrint(
    uint64_t value,
    const std::string_view units[],
    int unitSize,
    int unitOffset,
    int precision,
    int scale) {
  std::stringstream out;
  int offset = unitOffset;
  double outValue = static_cast<double>(value);
  while ((outValue / scale) >= 1 && offset < (unitSize - 1)) {
    outValue = outValue / scale;
    offset++;
  }
  if (offset == unitOffset) {
    // Print the default value.
    precision = 0;
  }
  out << std::fixed << std::setprecision(precision) << outValue
      << units[offset];
  return out.str();
}
} // namespace

std::string succinctMillis(uint64_t time, int precision) {
  return succinctPrint(
      time,
      timeUnits,
      sizeof(timeUnits) / sizeof(std::string_view),
      2,
      precision,
      kTimeScale);
}

std::string succinctNanos(uint64_t time, int precision) {
  return succinctPrint(
      time,
      timeUnits,
      sizeof(timeUnits) / sizeof(std::string_view),
      0,
      precision,
      kTimeScale);
}

std::string succinctBytes(uint64_t bytes, int precision) {
  return succinctPrint(
      bytes,
      byteUnits,
      sizeof(byteUnits) / sizeof(std::string_view),
      0,
      precision,
      kByteScale);
}

} // namespace facebook::velox::print
