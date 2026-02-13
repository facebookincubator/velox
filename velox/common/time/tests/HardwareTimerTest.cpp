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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "velox/common/time/HardwareTimer.h"

using namespace facebook::velox;

namespace facebook::velox::test {

namespace {

// Intercepts glog messages so tests can inspect logged output.
class CapturingLogSink : public google::LogSink {
 public:
  void send(
      google::LogSeverity /*severity*/,
      const char* /*full_filename*/,
      const char* /*base_filename*/,
      int /*line*/,
      const struct ::tm* /*tm_time*/,
      const char* message,
      size_t message_len) override {
    captured_ += std::string(message, message_len);
  }

  const std::string& captured() const {
    return captured_;
  }

  void clear() {
    captured_.clear();
  }

 private:
  std::string captured_;
};

// Strips ANSI escape sequences (e.g. \033[1;34m) from a string.
std::string stripAnsi(const std::string& s) {
  std::string result;
  result.reserve(s.size());
  bool inEscape = false;
  for (char c : s) {
    if (c == '\033') {
      inEscape = true;
    } else if (inEscape && c == 'm') {
      inEscape = false;
    } else if (!inEscape) {
      result += c;
    }
  }
  return result;
}

} // namespace

TEST(HardwareTimerTest, startEnd) {
  {
    HardwareTimer timer("startEnd");
    timer.start();
    [[maybe_unused]] int x = 0;
    for (int i = 0; i < 1000; ++i) {
      x += i;
    }
    timer.end();
  }
  HardwareTimer::cleanup();
}

TEST(HardwareTimerTest, multipleIterations) {
  {
    HardwareTimer timer("multipleIterations");
    for (int i = 0; i < 10; ++i) {
      timer.start();
      [[maybe_unused]] int x = 0;
      for (int j = 0; j < 100; ++j) {
        x += j;
      }
      timer.end();
    }
  }
  HardwareTimer::cleanup();
}

TEST(HardwareTimerTest, zeroIterationsNoOutput) {
  CapturingLogSink sink;
  google::AddLogSink(&sink);
  {
    HardwareTimer timer("zeroIterations");
  }
  HardwareTimer::cleanup();
  google::RemoveLogSink(&sink);

  // A timer with zero iterations should produce no table output.
  EXPECT_EQ(sink.captured().find("zeroIterations"), std::string::npos);
}

TEST(HardwareTimerTest, outputTableContextAndRuns) {
  CapturingLogSink sink;
  google::AddLogSink(&sink);

  {
    HardwareTimer timer("myTimer");
    for (int i = 0; i < 7; ++i) {
      timer.start();
      [[maybe_unused]] int x = 0;
      for (int j = 0; j < 100; ++j) {
        x += j;
      }
      timer.end();
    }
  }
  HardwareTimer::cleanup();
  google::RemoveLogSink(&sink);

  std::string output = stripAnsi(sink.captured());

  // Context name appears (with thread id suffix).
  EXPECT_NE(output.find("myTimer"), std::string::npos) << output;
  EXPECT_NE(output.find("[tid:"), std::string::npos) << output;

  // Iteration count of 7 appears in the Runs column.
  EXPECT_NE(output.find("7"), std::string::npos) << output;
}

TEST(HardwareTimerTest, outputTableTimeUnits) {
  CapturingLogSink sink;
  google::AddLogSink(&sink);

  {
    HardwareTimer timer("unitTest");
    timer.start();
    [[maybe_unused]] int x = 0;
    for (int i = 0; i < 100; ++i) {
      x += i;
    }
    timer.end();
  }
  HardwareTimer::cleanup();
  google::RemoveLogSink(&sink);

  std::string output = stripAnsi(sink.captured());

  // At least one time unit label must appear (Total and Average columns).
  bool hasUnit = output.find(" ns") != std::string::npos ||
      output.find(" us") != std::string::npos ||
      output.find(" ms") != std::string::npos ||
      output.find(" s") != std::string::npos;
  EXPECT_TRUE(hasUnit) << "No time unit found in output: " << output;
}

TEST(HardwareTimerTest, outputTableMultipleTimers) {
  CapturingLogSink sink;
  google::AddLogSink(&sink);

  {
    HardwareTimer t1("alphaTimer");
    t1.start();
    [[maybe_unused]] int x = 0;
    for (int i = 0; i < 100; ++i) {
      x += i;
    }
    t1.end();
  }
  {
    HardwareTimer t2("betaTimer");
    for (int i = 0; i < 3; ++i) {
      t2.start();
      [[maybe_unused]] int x = 0;
      for (int j = 0; j < 100; ++j) {
        x += j;
      }
      t2.end();
    }
  }
  HardwareTimer::cleanup();
  google::RemoveLogSink(&sink);

  std::string output = stripAnsi(sink.captured());

  // Both timer context names appear in the same table.
  EXPECT_NE(output.find("alphaTimer"), std::string::npos) << output;
  EXPECT_NE(output.find("betaTimer"), std::string::npos) << output;
}

} // namespace facebook::velox::test
