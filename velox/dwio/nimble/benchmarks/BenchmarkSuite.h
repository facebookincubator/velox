/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// BenchmarkSuite — runs each benchmark in its own BenchmarkingState so
// start/end hooks can fire per-benchmark.  Uses a standalone
// folly::detail::BenchmarkingState per benchmark, enabling std::function
// callbacks that fire exactly once per benchmark function (start and end).

#pragma once

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/FileUtil.h>
#include <folly/String.h>
#include <folly/dynamic.h>
#include <folly/json/json.h>
#include <glog/logging.h>

namespace facebook::nimble {

class BenchmarkSuite {
 public:
  using Clock = std::chrono::high_resolution_clock;
  using StartHook = std::function<void(const std::string& name)>;
  using EndHook = std::function<
      void(const std::string& name, const folly::detail::BenchmarkResult&)>;

  void setOnStart(StartHook hook) {
    onStart_ = std::move(hook);
  }
  void setOnEnd(EndHook hook) {
    onEnd_ = std::move(hook);
  }

  void setTitle(std::string title) {
    title_ = std::move(title);
  }

  void setJsonOutputPath(std::string path) {
    jsonOutputPath_ = std::move(path);
  }

  void addBenchmark(std::string name, std::function<void(unsigned)> fn) {
    entries_.push_back(Entry{std::move(name), std::move(fn), /*sep=*/false});
  }

  void addSeparator() {
    entries_.push_back(Entry{"", nullptr, /*sep=*/true});
  }

  void run() {
    std::vector<folly::detail::BenchmarkResult> allResults;

    // Print table header.
    printHeader();

    bool pendingSeparator = false;

    for (size_t idx = 0; idx < entries_.size(); ++idx) {
      auto& entry = entries_[idx];

      if (entry.isSeparator) {
        pendingSeparator = true;
        continue;
      }

      // Create a fresh BenchmarkingState for this benchmark.
      folly::detail::BenchmarkingState<Clock> state;

      // Register baseline benchmarks (required by the measurement engine).
      auto baselineName = std::string("baseline");
      auto suspenderBaselineName = std::string("suspender_baseline");

      state.addBenchmark(
          "BenchmarkSuite", std::move(baselineName), [](unsigned) {
      // Minimal baseline — matches folly's global baseline.
#ifdef _MSC_VER
            _ReadWriteBarrier();
#else
            asm volatile("");
#endif
            return 1u;
          });
      state.addBenchmark(
          "BenchmarkSuite", std::move(suspenderBaselineName), [](unsigned) {
            folly::BenchmarkSuspender sus;
            return 1u;
          });

      // Register the actual benchmark.
      auto fn = entry.fn; // copy the function for the lambda capture
      state.addBenchmark(
          "BenchmarkSuite", entry.name, [fn](unsigned iters) -> unsigned {
            fn(iters);
            return iters;
          });

      // Fire start hook.
      if (onStart_) {
        onStart_(entry.name);
      }

      // Run measurement.
      auto results = state.runBenchmarksWithResults();

      // Find the result for our benchmark (skip baselines).
      folly::detail::BenchmarkResult bmResult;
      bmResult.name = entry.name;
      bmResult.file = "BenchmarkSuite";
      bmResult.timeInNs = 0;
      for (auto& r : results) {
        if (r.name == entry.name) {
          bmResult = r;
          break;
        }
      }
      // Print separator if one was pending before this benchmark.
      if (pendingSeparator) {
        printSeparator('-');
        pendingSeparator = false;
      } else if (!allResults.empty()) {
        printSeparator('.');
      }

      // Print this result immediately.
      printResult(bmResult);

      // Fire end hook.
      if (onEnd_) {
        onEnd_(entry.name, bmResult);
      }

      allResults.push_back(bmResult);
    }

    // Print table footer.
    printSeparator('=');

    // Write JSON output if a path was provided.
    if (!jsonOutputPath_.empty()) {
      writeJsonResults(allResults, jsonOutputPath_);
    }
  }

 private:
  struct Entry {
    std::string name;
    std::function<void(unsigned)> fn;
    bool isSeparator;
  };

  // Human-readable formatting helpers matching folly's output.
  struct ScaleInfo {
    double boundary;
    const char* suffix;
  };

  static std::string
  humanReadable(double n, unsigned int decimals, const ScaleInfo* scales) {
    if (std::isinf(n) || std::isnan(n)) {
      return folly::to<std::string>(n);
    }
    const double absValue = fabs(n);
    const ScaleInfo* scale = scales;
    while (absValue < scale[0].boundary && scale[1].suffix != nullptr) {
      ++scale;
    }
    const double scaledValue = n / scale->boundary;
    return fmt::format("{:.{}f}{}", scaledValue, decimals, scale->suffix);
  }

  static std::string readableTime(double n, unsigned int decimals) {
    static const ScaleInfo kTimeSuffixes[] = {
        {365.25 * 24 * 3600, "years"},
        {24 * 3600, "days"},
        {3600, "hr"},
        {60, "min"},
        {1, "s"},
        {1E-3, "ms"},
        {1E-6, "us"},
        {1E-9, "ns"},
        {1E-12, "ps"},
        {1E-15, "fs"},
        {0, nullptr},
    };
    return humanReadable(n, decimals, kTimeSuffixes);
  }

  static std::string metricReadable(double n, unsigned int decimals) {
    static const ScaleInfo kMetricSuffixes[] = {
        {1E24, "Y"},
        {1E21, "Z"},
        {1E18, "X"},
        {1E15, "P"},
        {1E12, "T"},
        {1E9, "G"},
        {1E6, "M"},
        {1E3, "K"},
        {1, ""},
        {1E-3, "m"},
        {1E-6, "u"},
        {1E-9, "n"},
        {1E-12, "p"},
        {1E-15, "f"},
        {1E-18, "a"},
        {1E-21, "z"},
        {1E-24, "y"},
        {0, nullptr},
    };
    return humanReadable(n, decimals, kMetricSuffixes);
  }

  static constexpr unsigned int kColumns = 76;
  static constexpr std::string_view kHeader = "time/iter   iters/s";

  // ANSI escape helpers.
  static constexpr const char* kBold = "\033[1m";
  static constexpr const char* kBoldBlue = "\033[1;34m";
  static constexpr const char* kBoldCyan = "\033[1;36m";
  static constexpr const char* kBoldGreen = "\033[1;32m";
  static constexpr const char* kBoldYellow = "\033[1;33m";
  static constexpr const char* kReset = "\033[0m";

  static void printSeparator(char pad) {
    printf("%s%s%s\n", kBoldBlue, std::string(kColumns, pad).c_str(), kReset);
  }

  void printHeader() const {
    printSeparator('=');
    static const std::string kDefaultTitle = "BenchmarkSuite";
    const std::string& title = title_.empty() ? kDefaultTitle : title_;
    size_t padLen = (kColumns > title.size() + kHeader.size())
        ? (kColumns - title.size() - kHeader.size())
        : 1;
    printf(
        "%s%s%*s%s%.*s%s\n",
        kBold,
        title.c_str(),
        static_cast<int>(padLen),
        "",
        kBoldYellow,
        static_cast<int>(kHeader.size()),
        kHeader.data(),
        kReset);
    printSeparator('=');
  }

  static void printResult(const folly::detail::BenchmarkResult& r) {
    const double nsPerIter = r.timeInNs;
    const double secPerIter = nsPerIter / 1E9;
    const double itersPerSec = (secPerIter == 0)
        ? std::numeric_limits<double>::infinity()
        : (1.0 / secPerIter);

    std::string name = r.name;
    size_t nameWidth = kColumns - kHeader.size();
    name.resize(nameWidth, ' ');

    printf(
        "%s%*s%s%s%9.9s  %s%8.8s%s\n",
        kBoldGreen,
        static_cast<int>(name.size()),
        name.c_str(),
        kReset,
        kBoldCyan,
        readableTime(secPerIter, 2).c_str(),
        kBoldYellow,
        metricReadable(itersPerSec, 2).c_str(),
        kReset);
  }

  static void writeJsonResults(
      const std::vector<folly::detail::BenchmarkResult>& results,
      const std::string& outputPath) {
    folly::dynamic d;
    folly::benchmarkResultsToDynamic(results, d);
    auto jsonStr = folly::toPrettyJson(d);
    if (folly::writeFile(jsonStr, outputPath.c_str())) {
      LOG(INFO) << "JSON results written to " << outputPath;
    } else {
      LOG(ERROR) << "Failed to write JSON results to " << outputPath;
    }
  }

  std::vector<Entry> entries_;
  StartHook onStart_;
  EndHook onEnd_;
  std::string title_;
  std::string jsonOutputPath_;
};

} // namespace facebook::nimble
