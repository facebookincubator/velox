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
#include <sstream>
#include <string>
#include <vector>
#include "folly/chrono/Hardware.h"
#include "folly/container/F14Map.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox {

/// Low-overhead cycle-counting timer for micro-benchmarks on x86-64.
///
/// Uses the RDTSCP instruction via folly::hardware_timestamp() to measure
/// elapsed CPU cycles, then converts to nanoseconds using a one-time TSC
/// frequency calibration. A single HardwareTimer instance manages multiple
/// named code sections via an internal map. Each call to start(name)/end(name)
/// records one timing sample for that section, accumulating results in a
/// RuntimeMetric (sum, count, min, max).
///
/// On destruction (or when printStats() is called explicitly), the collected
/// results are printed as a formatted table to stderr via glog.
///
/// Not thread-safe — each instance should be used on a single thread.
///
/// Guarded by ENABLE_HW_TIMER; compiles to nothing when the flag is absent.
///
/// Usage:
///   {
///     HardwareTimer timer("my benchmark");
///     timer.start("section_a");
///     // ... work ...
///     timer.end("section_a");
///   } // prints results table to stderr
class HardwareTimer {
 public:
  /// Constructs a timer session with an optional name. The name is displayed
  /// in the output table header.
  explicit HardwareTimer(const std::string& name = "") : name_(name) {}

  HardwareTimer(const HardwareTimer&) = delete;
  HardwareTimer& operator=(const HardwareTimer&) = delete;
  HardwareTimer(HardwareTimer&&) = delete;
  HardwareTimer& operator=(HardwareTimer&&) = delete;

  /// Prints the results table if any entries were recorded.
  ~HardwareTimer() {
    printStats();
  }

  /// Records the current hardware timestamp as the start of a timed interval
  /// for the given context name. Must be paired with a subsequent call to
  /// end() with the same context name.
  void start(const char* contextName) {
    openContexts_[contextName] = folly::hardware_timestamp();
  }

  /// Records the end of a timed interval for the given context name. Converts
  /// elapsed cycles to nanoseconds and accumulates the result into the
  /// RuntimeMetric for that context. Must be preceded by a call to start()
  /// with the same context name.
  void end(const char* contextName) {
    const uint64_t endCycles = folly::hardware_timestamp();
    auto it = openContexts_.find(contextName);
    VELOX_CHECK(
        it != openContexts_.end(),
        "HardwareTimer::end() called without matching start(), context: {}",
        contextName);
    VELOX_CHECK_GE(endCycles, it->second);
    const auto elapsedNs = static_cast<int64_t>(
        static_cast<double>(endCycles - it->second) / estimateTscFreqGhz());
    auto [entryIt, inserted] =
        entries_.try_emplace(contextName, RuntimeCounter::Unit::kNanos);
    entryIt->second.addValue(elapsedNs);
    openContexts_.erase(it);
  }

  /// Prints the results table and clears all accumulated entries. This allows
  /// the timer to be reused for a new session.
  void printStats() {
    if (entries_.empty()) {
      return;
    }
    printStatsTable(entries_, name_);
    entries_.clear();
  }

 private:
  using EntriesMap = folly::F14FastMap<std::string, RuntimeMetric>;

  static double estimateTscFreqGhz(
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

  static void printStatsTable(const EntriesMap& map, const std::string& name) {
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

    for (const auto& [contextName, metric] : map) {
      Row row;
      row.name = contextName;
      row.runs = std::to_string(metric.count);
      row.total = succinctNanos(metric.sum);
      row.avg =
          succinctNanos(metric.count == 0 ? 0 : metric.sum / metric.count);
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
    if (!name.empty()) {
      oss << "\033[1;34m| \033[0m"
          << "\033[1;37m" << std::left
          << std::setw(nameWidth + runsWidth + totalWidth + avgWidth + 4)
          << name << "\033[0m"
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
    LOG(INFO) << "\nBreakdown:\n" << oss.str();
  }

  /// Optional name displayed in the output table header.
  const std::string name_;
  /// Aggregated stats indexed by context name.
  EntriesMap entries_;
  /// Pending start timestamps for contexts that have been started but not yet
  /// ended.
  folly::F14FastMap<std::string, uint64_t> openContexts_;
};

} // namespace facebook::velox

#endif // ENABLE_HW_TIMER
