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
#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>

namespace facebook::velox {

class MicrosecondTimer {
 public:
  MicrosecondTimer(uint64_t* timer) : timer_(timer) {
    start_ = std::chrono::steady_clock::now();
  }

  ~MicrosecondTimer() {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_);

    (*timer_) += duration.count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
  uint64_t* timer_;
};

// Returns the current epoch time in milliseconds.
size_t getCurrentTimeMs();

// Returns the current epoch time in microseconds.
size_t getCurrentTimeMicro();

// Match the input time in nanoseconds to the most appropriate unit and return a
// string value. Possible units are nanoseconds(ns), microseconds(us),
// milliseconds(ms), seconds(s). The default precision is 4 decimal digits.
static std::string prettyPrintTimeInNanos(uint64_t time, int precision = 4) {
  std::stringstream out;
  std::string units[4] = {"ns", "us", "ms", "s"};
  int count = 0;
  double outTime = static_cast<double>(time);
  while ((outTime / 1000) > 1 && count < sizeof(units)) {
    outTime = outTime / 1000;
    count++;
  }
  out << std::fixed << std::setprecision(precision) << outTime << units[count];
  return out.str();
}

} // namespace facebook::velox
