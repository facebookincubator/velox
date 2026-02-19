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

#ifdef ENABLE_HW_TIMER

#if !defined(__x86_64__)
#error "HardwareTimer requires x86-64 architecture (RDTSCP instruction)"
#endif
#include <glog/logging.h>
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "folly/chrono/Hardware.h"
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox {

// Note: this class is not thread-safe. It is intended to be used in a single
// thread context only to do micro-benchmarks.
class HardwareTimer {
 public:
  explicit HardwareTimer(const char* contextName) : contextName_(contextName) {}

  HardwareTimer(const HardwareTimer&) = delete;
  HardwareTimer& operator=(const HardwareTimer&) = delete;
  HardwareTimer(HardwareTimer&&) = delete;
  HardwareTimer& operator=(HardwareTimer&&) = delete;

  void start() {
    startCycles_ = folly::hardware_timestamp();
  }

  void end() {
    const uint64_t endCycles = folly::hardware_timestamp();
    totalCycles_ += endCycles - startCycles_;
    ++iterCount_;
  }

  ~HardwareTimer() {
    if (iterCount_ == 0) {
      return;
    }
    const double totalNs =
        static_cast<double>(totalCycles_) / estimateTscFreqGhz();
    const double avgNs = totalNs / static_cast<double>(iterCount_);
    std::ostringstream keyStream;
    keyStream << contextName_ << " [tid:" << std::this_thread::get_id() << "]";
    entries()[keyStream.str()] = {iterCount_, totalNs, avgNs};
  }

  static void init(const std::string& name = "") {
    auto& map = entries();
    if (!map.empty()) {
      throw std::runtime_error(
          "HardwareTimer::init() called but entries map is not empty. "
          "Call cleanup() before re-initializing.");
    }
    globalContextName() = name;
  }

  static void cleanup() {
    auto& map = entries();
    if (map.empty()) {
      return;
    }
    printTable(map, globalContextName());
    map.clear();
    globalContextName().clear();
  }

 private:
  struct Entry {
    // The number of iterations the instrumented section of the code is invoked.
    uint64_t iterCount;
    // The total execution time the instrumented section of the code is invoked.
    double totalNs;
    // The avg execution time the instrumented section of the code is invoked.
    double avgNs;
  };

  using EntriesMap = std::map<std::string, Entry>;

  static EntriesMap& entries() {
    static EntriesMap map;
    return map;
  }

  static std::string& globalContextName() {
    static std::string name;
    return name;
  }

  double estimateTscFreqGhz(
      std::chrono::milliseconds window = std::chrono::milliseconds(100)) {
    static const double cached = [&]() {
      using clock = std::chrono::steady_clock;
      // Warm-up (reduces first-use noise)
      (void)folly::hardware_timestamp();
      auto t0 = clock::now();
      uint64_t c0 = folly::hardware_timestamp();
      // Busy-wait until window elapsed (less jitter than sleep)
      while (clock::now() - t0 < window) {
#if defined(__x86_64__)
        _mm_pause();
#else
#error "HardwareTimer requires x86-64 architecture (for pause)"
#endif
      }
      auto t1 = clock::now();
      uint64_t c1 = folly::hardware_timestamp();
      std::chrono::duration<double> seconds = t1 - t0;
      return (static_cast<double>(c1 - c0) / seconds.count()) / 1e9;
    }();
    return cached;
  }

  static void printTable(const EntriesMap& map, const std::string& title) {
    int nameWidth = static_cast<int>(std::string("Context").size());
    int runsWidth = static_cast<int>(std::string("Runs").size());
    int totalWidth = static_cast<int>(std::string("Total").size());
    int avgWidth = static_cast<int>(std::string("Average").size());

    struct Row {
      std::string name;
      std::string runs;
      std::string total;
      std::string avg;
    };
    std::vector<Row> rows;
    rows.reserve(map.size());

    for (const auto& [name, entry] : map) {
      Row row;
      row.name = name;
      row.runs = std::to_string(entry.iterCount);
      row.total = succinctNanos(static_cast<uint64_t>(entry.totalNs));
      row.avg = succinctNanos(static_cast<uint64_t>(entry.avgNs));
      nameWidth = std::max(nameWidth, static_cast<int>(row.name.size()));
      runsWidth = std::max(runsWidth, static_cast<int>(row.runs.size()));
      totalWidth = std::max(totalWidth, static_cast<int>(row.total.size()));
      avgWidth = std::max(avgWidth, static_cast<int>(row.avg.size()));
      rows.push_back(std::move(row));
    }

    nameWidth += 2;
    runsWidth += 2;
    totalWidth += 2;
    avgWidth += 2;

    auto sep = std::string(
        static_cast<size_t>(nameWidth + runsWidth + totalWidth + avgWidth + 7),
        '-');

    std::ostringstream oss;
    oss << "\n\033[1;34m" << sep << "\033[0m\n";
    if (!title.empty()) {
      oss << "\033[1;34m| \033[0m"
          << "\033[1;37m" << std::left
          << std::setw(nameWidth + runsWidth + totalWidth + avgWidth + 4)
          << title << "\033[0m"
          << "\033[1;34m|\033[0m\n";
      oss << "\033[1;34m" << sep << "\033[0m\n";
    }
    oss << "\033[1;34m| \033[0m"
        << "\033[1;33m" << std::left << std::setw(nameWidth) << "Context"
        << "\033[0m"
        << "\033[1;34m| \033[0m"
        << "\033[1;33m" << std::setw(runsWidth) << "Runs"
        << "\033[0m"
        << "\033[1;34m| \033[0m"
        << "\033[1;33m" << std::setw(totalWidth) << "Total"
        << "\033[0m"
        << "\033[1;34m| \033[0m"
        << "\033[1;33m" << std::setw(avgWidth) << "Average"
        << "\033[0m"
        << "\033[1;34m|\033[0m\n";
    oss << "\033[1;34m" << sep << "\033[0m\n";

    for (const auto& row : rows) {
      oss << "\033[1;34m| \033[0m"
          << "\033[1;32m" << std::left << std::setw(nameWidth) << row.name
          << "\033[0m"
          << "\033[1;34m| \033[0m"
          << "\033[0m" << std::setw(runsWidth) << row.runs
          << "\033[1;34m| \033[0m"
          << "\033[1;36m" << std::setw(totalWidth) << row.total << "\033[0m"
          << "\033[1;34m| \033[0m"
          << "\033[1;35m" << std::setw(avgWidth) << row.avg << "\033[0m"
          << "\033[1;34m|\033[0m\n";
    }

    oss << "\033[1;34m" << sep << "\033[0m\n";
    std::cerr << "\nBreakdown:\n" << oss.str();
  }

  const char* contextName_{};
  uint64_t totalCycles_{};
  uint64_t startCycles_{};
  uint64_t iterCount_{};
};

} // namespace facebook::velox

// Convenience macros for HardwareTimer instrumentation.
// Results are printed via LOG(INFO) when the timer goes out of scope.
//
// HWT(name)       - Declare a timer (for loop accumulation pattern).
// HWT_START(name) - Start (or restart) a declared timer.
// HWT_END(name)   - End a measurement interval.
//
// Single-shot: HWT(t); HWT_START(t); ... HWT_END(t);
// Loop:        HWT(t); for (...) { HWT_START(t); ... HWT_END(t); }
//
// All call sites must be wrapped in #ifdef ENABLE_HW_TIMER
// so that these macros are invisible to the IDE when the flag is off.
#define HWT(name) facebook::velox::HardwareTimer name(#name)
#define HWT_START(name) (name).start()
#define HWT_END(name) (name).end()

#endif // ENABLE_HW_TIMER
