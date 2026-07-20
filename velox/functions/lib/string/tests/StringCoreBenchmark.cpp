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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

#include "velox/functions/lib/string/StringCore.h"

using namespace facebook::velox::functions;

DEFINE_int32(num_strings, 10000, "Number of strings to process");
DEFINE_int32(max_chars, 128, "maxChars argument to cappedByteLengthUnicode");

namespace facebook::velox::functions {

class StringCoreBenchmark {
 public:
  static void init(int32_t length, int32_t ratio) {
    strings_.clear();
    strings_.reserve(FLAGS_num_strings);
    std::mt19937 rng(42);
    for (int32_t i = 0; i < FLAGS_num_strings; ++i) {
      strings_.push_back(generateString(length, ratio, rng));
    }
  }

  static const std::vector<std::string>& strings() {
    return strings_;
  }

 private:
  static std::string
  generateString(int32_t length, int32_t ratio, std::mt19937& rng) {
    if (ratio == 100) {
      return generateAscii(length, rng);
    }
    if (ratio == 0) {
      return generateUnicode(length, rng);
    }
    return generateMixed(length, ratio, rng);
  }

  static std::string generateAscii(int32_t length, std::mt19937& rng) {
    std::string s(length, '\0');
    std::uniform_int_distribution<int32_t> dist(0x20, 0x7E);
    for (int32_t i = 0; i < length; ++i) {
      s[i] = static_cast<char>(dist(rng));
    }
    return s;
  }

  static std::string
  generateMixed(int32_t length, int32_t ratio, std::mt19937& rng) {
    std::string s;
    s.reserve(length);
    std::uniform_int_distribution<int32_t> asciiDist(0x20, 0x7E);
    std::uniform_int_distribution<int32_t> unicodeDist(0x4E00, 0x9FFF);
    std::uniform_int_distribution<int32_t> chance(1, 100);

    while (static_cast<int32_t>(s.size()) < length) {
      if (chance(rng) > ratio && static_cast<int32_t>(s.size()) + 3 <= length) {
        int32_t cp = unicodeDist(rng);
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        s.push_back(static_cast<char>(asciiDist(rng)));
      }
    }
    return s;
  }

  static std::string generateUnicode(int32_t length, std::mt19937& rng) {
    std::string s;
    s.reserve(length);
    std::uniform_int_distribution<int32_t> dist(0x4E00, 0x9FFF);

    while (static_cast<int32_t>(s.size()) + 3 <= length) {
      int32_t cp = dist(rng);
      s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
  }

  static std::vector<std::string> strings_;
};

} // namespace facebook::velox::functions

std::vector<std::string> StringCoreBenchmark::strings_;

void runCappedByteLengthUnicode(uint32_t iters, int32_t length, int32_t ratio) {
  folly::BenchmarkSuspender suspender;
  StringCoreBenchmark::init(length, ratio);
  suspender.dismiss();

  int64_t total = 0;
  const int32_t maxChars = std::min(FLAGS_max_chars, length);
  for (unsigned i = 0; i < iters; ++i) {
    for (const auto& s : StringCoreBenchmark::strings()) {
      total +=
          stringCore::cappedByteLengthUnicode(s.data(), s.size(), maxChars);
    }
  }
  folly::doNotOptimizeAway(total);
}

void runCappedLengthUnicode(uint32_t iters, int32_t length, int32_t ratio) {
  folly::BenchmarkSuspender suspender;
  StringCoreBenchmark::init(length, ratio);
  suspender.dismiss();

  int64_t total = 0;
  const int32_t maxChars = std::min(FLAGS_max_chars, length);
  for (unsigned i = 0; i < iters; ++i) {
    for (const auto& s : StringCoreBenchmark::strings()) {
      total += stringCore::cappedLengthUnicode(s.data(), s.size(), maxChars);
    }
  }
  folly::doNotOptimizeAway(total);
}

int32_t main(int32_t argc, char** argv) {
  folly::Init init{&argc, &argv};

  std::vector<int32_t> ratios = {0, 1, 25, 50, 75, 99, 100};
  std::vector<int32_t> lengths = {4, 16, 64, 256, 1024};

  for (auto length : lengths) {
    for (auto ratio : ratios) {
      std::string suffix =
          std::to_string(length) + "_" + std::to_string(ratio) + "%";
      folly::addBenchmark(
          __FILE__,
          "cappedByteLengthUnicode_" + suffix,
          [length, ratio](unsigned iters) {
            runCappedByteLengthUnicode(iters, length, ratio);
            return iters;
          });
    }
  }

  folly::addBenchmark(__FILE__, "-", []() { return 0; });

  for (auto length : lengths) {
    for (auto ratio : ratios) {
      std::string suffix =
          std::to_string(length) + "_" + std::to_string(ratio) + "%";
      folly::addBenchmark(
          __FILE__,
          "cappedLengthUnicode_" + suffix,
          [length, ratio](unsigned iters) {
            runCappedLengthUnicode(iters, length, ratio);
            return iters;
          });
    }
  }

  folly::runBenchmarks();
  return 0;
}
