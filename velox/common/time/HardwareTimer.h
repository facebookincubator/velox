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

#if !defined(__x86_64__)
#error "HardwareTimer requires x86-64 architecture (RDTSCP instruction)"
#endif
#include <glog/logging.h>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "folly/chrono/Hardware.h"

namespace facebook::velox {

// Note: this class is not thread-safe. It is intended to be used in a single
// thread context only to do micro-benchmarks.
class HardwareTimer {
 public:
  struct Entry {
    uint64_t iter_count;
    double total_ns;
    double avg_ns;
  };

  explicit HardwareTimer(const char* context_name)
      : context_name_(context_name) {}

  HardwareTimer(const HardwareTimer&) = delete;
  HardwareTimer& operator=(const HardwareTimer&) = delete;
  HardwareTimer(HardwareTimer&&) = delete;
  HardwareTimer& operator=(HardwareTimer&&) = delete;

  void start() {
    start_cycles_ = folly::hardware_timestamp();
  }

  void end() {
    const uint64_t end_cycles = folly::hardware_timestamp();
    total_cycles_ += end_cycles - start_cycles_;
    ++iter_count_;
  }

  ~HardwareTimer() {
    if (iter_count_ == 0) {
      return;
    }
    const double total_ns =
        static_cast<double>(total_cycles_) / estimate_tsc_freq_ghz();
    const double avg_ns = total_ns / static_cast<double>(iter_count_);
    std::ostringstream keyStream;
    keyStream << context_name_ << " [tid:" << std::this_thread::get_id() << "]";
    entries()[keyStream.str()] = {iter_count_, total_ns, avg_ns};
  }

  static void cleanup() {
    auto& map = entries();
    if (map.empty()) {
      return;
    }
    printTable(map);
    map.clear();
  }

 private:
  const char* context_name_{};

  uint64_t total_cycles_{};
  uint64_t start_cycles_{};
  uint64_t iter_count_{};

  using EntriesMap = std::map<std::string, Entry>;

  static EntriesMap& entries() {
    static EntriesMap map;
    return map;
  }

  static double estimate_tsc_freq_ghz(
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

  static void printTable(const EntriesMap& map) {
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
      row.runs = std::to_string(entry.iter_count);
      row.total = formatAutoUnit(entry.total_ns);
      row.avg = formatAutoUnit(entry.avg_ns);
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

    oss << "\033[1;34m" << sep << "\033[0m";
    LOG(INFO) << oss.str();
  }

  static std::string formatAutoUnit(double valueInNanoseconds) {
    static constexpr struct {
      double threshold_ns;
      const char* label;
      double divisor;
    } kUnits[] = {
        {1e3, " ns", 1.0},
        {1e6, " us", 1e3},
        {1e9, " ms", 1e6},
        {60.0 * 1e9, " s", 1e9},
        {std::numeric_limits<double>::infinity(), " min", 60.0 * 1e9}};

    double v = valueInNanoseconds;
    const char* label = " ns";
    const double abs_v = std::fabs(v);
    for (const auto& u : kUnits) {
      label = u.label;
      if (abs_v < u.threshold_ns) {
        v /= u.divisor;
        break;
      }
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(5);
    oss << v << label;
    return oss.str();
  }
};

} // namespace facebook::velox
