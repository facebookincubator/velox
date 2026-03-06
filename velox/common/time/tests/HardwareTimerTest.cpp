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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sstream>

#include "velox/common/time/HardwareTimer.h"

using namespace facebook::velox;

namespace facebook::velox::test {

namespace {

// Intercepts glog INFO messages so tests can inspect logged output.
class LogCapture : public google::LogSink {
 public:
  LogCapture() {
    google::AddLogSink(this);
  }

  ~LogCapture() override {
    google::RemoveLogSink(this);
  }

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

  LogCapture(const LogCapture&) = delete;
  LogCapture& operator=(const LogCapture&) = delete;

  const std::string& captured() const {
    return captured_;
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

// Helper to simulate a function deeper in the call stack that uses
// a timer by context name.
void innerWork(HardwareTimer& timer) {
  timer.start("crossStackTimer");
  [[maybe_unused]] int x = 0;
  for (int i = 0; i < 100; ++i) {
    x += i;
  }
  timer.end("crossStackTimer");
}

void nestedInner(HardwareTimer& timer) {
  timer.start("innerTimer");
  [[maybe_unused]] int x = 0;
  for (int i = 0; i < 100; ++i) {
    x += i;
  }
  timer.end("innerTimer");
}

void nestedOuter(HardwareTimer& timer) {
  timer.start("outerTimer");
  [[maybe_unused]] int x = 0;
  for (int i = 0; i < 100; ++i) {
    x += i;
  }
  nestedInner(timer);
  timer.end("outerTimer");
}

void doWork() {
  [[maybe_unused]] int x = 0;
  for (int i = 0; i < 100; ++i) {
    x += i;
  }
}

// Represents a parsed data row from the printed table.
struct TableRow {
  std::string context;
  uint64_t runs;
  std::string total;
  std::string avg;
};

// Parses data rows from stripped-ANSI table output. Data rows are identified
// as pipe-separated lines with 4+ fields that are not the header row.
std::vector<TableRow> parseTableRows(const std::string& output) {
  std::vector<TableRow> rows;
  std::istringstream iss(output);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find('|') == std::string::npos) {
      continue;
    }
    std::vector<std::string> fields;
    std::istringstream ls(line);
    std::string field;
    while (std::getline(ls, field, '|')) {
      auto start = field.find_first_not_of(' ');
      auto end = field.find_last_not_of(' ');
      if (start != std::string::npos) {
        fields.push_back(field.substr(start, end - start + 1));
      }
    }
    // Skip header row and title rows (fewer than 4 fields).
    if (fields.size() >= 4 && fields[0] != "Context") {
      rows.push_back({fields[0], std::stoull(fields[1]), fields[2], fields[3]});
    }
  }
  return rows;
}

// Finds all rows whose context starts with the given name prefix.
std::vector<TableRow> findRows(
    const std::vector<TableRow>& rows,
    const std::string& namePrefix) {
  std::vector<TableRow> result;
  for (const auto& row : rows) {
    if (row.context.find(namePrefix) == 0) {
      result.push_back(row);
    }
  }
  return result;
}

void assertTableHeaders(const std::string& output) {
  EXPECT_NE(output.find("Breakdown:"), std::string::npos) << output;
  EXPECT_NE(output.find("Context"), std::string::npos) << output;
  EXPECT_NE(output.find("Runs"), std::string::npos) << output;
  EXPECT_NE(output.find("Total"), std::string::npos) << output;
  EXPECT_NE(output.find("Average"), std::string::npos) << output;
}

bool hasValidTimeUnit(const std::string& s) {
  return s.find("ns") != std::string::npos ||
      s.find("us") != std::string::npos || s.find("ms") != std::string::npos;
}

} // namespace

TEST(HardwareTimerTest, startEnd) {
  HardwareTimer timer;
  timer.start("startEnd");
  [[maybe_unused]] int x = 0;
  for (int i = 0; i < 1000; ++i) {
    x += i;
  }
  timer.end("startEnd");
  timer.printStats();
}

TEST(HardwareTimerTest, multipleIterations) {
  HardwareTimer timer;
  for (int i = 0; i < 10; ++i) {
    timer.start("multipleIterations");
    [[maybe_unused]] int x = 0;
    for (int j = 0; j < 100; ++j) {
      x += j;
    }
    timer.end("multipleIterations");
  }
  timer.printStats();
}

TEST(HardwareTimerTest, noEntriesNoOutput) {
  LogCapture capture;
  {
    HardwareTimer timer;
  }

  // A timer with no entries should produce no table output.
  EXPECT_TRUE(capture.captured().empty()) << capture.captured();
}

TEST(HardwareTimerTest, outputTableContextAndRuns) {
  LogCapture capture;

  {
    HardwareTimer timer;
    for (int i = 0; i < 7; ++i) {
      timer.start("myTimer");
      [[maybe_unused]] int x = 0;
      for (int j = 0; j < 100; ++j) {
        x += j;
      }
      timer.end("myTimer");
    }
  }

  std::string output = stripAnsi(capture.captured());
  assertTableHeaders(output);

  auto rows = parseTableRows(output);
  ASSERT_EQ(rows.size(), 1) << output;
  EXPECT_EQ(rows[0].context, "myTimer") << rows[0].context;
  EXPECT_EQ(rows[0].runs, 7);
  EXPECT_TRUE(hasValidTimeUnit(rows[0].total)) << rows[0].total;
  EXPECT_TRUE(hasValidTimeUnit(rows[0].avg)) << rows[0].avg;
}

TEST(HardwareTimerTest, outputTableTimeUnits) {
  LogCapture capture;

  {
    HardwareTimer timer;
    timer.start("unitTest");
    [[maybe_unused]] int x = 0;
    for (int i = 0; i < 100; ++i) {
      x += i;
    }
    timer.end("unitTest");
  }

  std::string output = stripAnsi(capture.captured());
  assertTableHeaders(output);

  auto rows = parseTableRows(output);
  ASSERT_EQ(rows.size(), 1) << output;
  EXPECT_EQ(rows[0].runs, 1);
  EXPECT_TRUE(hasValidTimeUnit(rows[0].total)) << rows[0].total;
  EXPECT_TRUE(hasValidTimeUnit(rows[0].avg)) << rows[0].avg;
}

TEST(HardwareTimerTest, outputTableMultipleContexts) {
  LogCapture capture;

  {
    HardwareTimer timer;
    timer.start("alphaTimer");
    [[maybe_unused]] int x = 0;
    for (int i = 0; i < 100; ++i) {
      x += i;
    }
    timer.end("alphaTimer");

    for (int i = 0; i < 3; ++i) {
      timer.start("betaTimer");
      x = 0;
      for (int j = 0; j < 100; ++j) {
        x += j;
      }
      timer.end("betaTimer");
    }
  }

  std::string output = stripAnsi(capture.captured());
  assertTableHeaders(output);

  auto rows = parseTableRows(output);
  ASSERT_EQ(rows.size(), 2) << output;

  auto alpha = findRows(rows, "alphaTimer");
  ASSERT_EQ(alpha.size(), 1) << output;
  EXPECT_EQ(alpha[0].runs, 1);
  EXPECT_TRUE(hasValidTimeUnit(alpha[0].total)) << alpha[0].total;
  EXPECT_TRUE(hasValidTimeUnit(alpha[0].avg)) << alpha[0].avg;

  auto beta = findRows(rows, "betaTimer");
  ASSERT_EQ(beta.size(), 1) << output;
  EXPECT_EQ(beta[0].runs, 3);
  EXPECT_TRUE(hasValidTimeUnit(beta[0].total)) << beta[0].total;
  EXPECT_TRUE(hasValidTimeUnit(beta[0].avg)) << beta[0].avg;
}

TEST(HardwareTimerTest, timerAcrossStack) {
  LogCapture capture;

  {
    HardwareTimer timer;
    // Timer is passed to a helper function, accumulating across
    // multiple calls.
    innerWork(timer);
    innerWork(timer);
    innerWork(timer);
  }

  std::string output = stripAnsi(capture.captured());
  assertTableHeaders(output);

  auto rows = parseTableRows(output);
  ASSERT_EQ(rows.size(), 1) << output;
  auto found = findRows(rows, "crossStackTimer");
  ASSERT_EQ(found.size(), 1) << output;
  EXPECT_EQ(found[0].runs, 3);
  EXPECT_TRUE(hasValidTimeUnit(found[0].total)) << found[0].total;
  EXPECT_TRUE(hasValidTimeUnit(found[0].avg)) << found[0].avg;
}

TEST(HardwareTimerTest, scopedTimer) {
  LogCapture capture;

  {
    HardwareTimer timer;
    for (int i = 0; i < 5; ++i) {
      timer.start("scopedTest");
      [[maybe_unused]] int x = 0;
      for (int j = 0; j < 100; ++j) {
        x += j;
      }
      timer.end("scopedTest");
    }
  }

  std::string output = stripAnsi(capture.captured());
  assertTableHeaders(output);

  auto rows = parseTableRows(output);
  ASSERT_EQ(rows.size(), 1) << output;
  auto found = findRows(rows, "scopedTest");
  ASSERT_EQ(found.size(), 1) << output;
  EXPECT_EQ(found[0].runs, 5);
  EXPECT_TRUE(hasValidTimeUnit(found[0].total)) << found[0].total;
  EXPECT_TRUE(hasValidTimeUnit(found[0].avg)) << found[0].avg;
}

TEST(HardwareTimerTest, destructorPrintsTitleAndResults) {
  LogCapture capture;

  {
    HardwareTimer timer("MyBenchmarkTitle");
    timer.start("sessionTimer");
    doWork();
    timer.end("sessionTimer");
  } // Destructor prints the table.

  std::string output = stripAnsi(capture.captured());
  assertTableHeaders(output);
  EXPECT_NE(output.find("MyBenchmarkTitle"), std::string::npos) << output;

  auto rows = parseTableRows(output);
  ASSERT_EQ(rows.size(), 1) << output;
  auto found = findRows(rows, "sessionTimer");
  ASSERT_EQ(found.size(), 1) << output;
  EXPECT_EQ(found[0].runs, 1);
  EXPECT_TRUE(hasValidTimeUnit(found[0].total)) << found[0].total;
  EXPECT_TRUE(hasValidTimeUnit(found[0].avg)) << found[0].avg;
}

TEST(HardwareTimerTest, nestedScopedTimers) {
  LogCapture capture;

  {
    HardwareTimer timer;
    // outerTimer and innerTimer are used from nested function calls.
    nestedOuter(timer);
    nestedOuter(timer);
  }

  std::string output = stripAnsi(capture.captured());
  assertTableHeaders(output);

  auto rows = parseTableRows(output);
  ASSERT_EQ(rows.size(), 2) << output;

  auto outer = findRows(rows, "outerTimer");
  ASSERT_EQ(outer.size(), 1) << output;
  EXPECT_EQ(outer[0].runs, 2);
  EXPECT_TRUE(hasValidTimeUnit(outer[0].total)) << outer[0].total;

  auto inner = findRows(rows, "innerTimer");
  ASSERT_EQ(inner.size(), 1) << output;
  EXPECT_EQ(inner[0].runs, 2);
  EXPECT_TRUE(hasValidTimeUnit(inner[0].total)) << inner[0].total;
}

TEST(HardwareTimerTest, printStatsClearsEntries) {
  LogCapture capture1;
  HardwareTimer timer("Session");

  timer.start("firstTimer");
  doWork();
  timer.end("firstTimer");
  timer.printStats();

  std::string output1 = stripAnsi(capture1.captured());
  assertTableHeaders(output1);
  EXPECT_NE(output1.find("Session"), std::string::npos) << output1;
  auto rows1 = parseTableRows(output1);
  ASSERT_EQ(rows1.size(), 1) << output1;
  auto first = findRows(rows1, "firstTimer");
  ASSERT_EQ(first.size(), 1) << output1;
  EXPECT_EQ(first[0].runs, 1);
  EXPECT_TRUE(hasValidTimeUnit(first[0].total)) << first[0].total;

  // After printStats(), entries should be cleared.
  // Destructor should produce no further output.
  LogCapture capture2;
  timer.printStats();
  EXPECT_TRUE(capture2.captured().empty()) << capture2.captured();
}

TEST(HardwareTimerTest, mixedScopedAndManualTimers) {
  LogCapture capture;

  {
    HardwareTimer timer;

    // Manual timer.
    timer.start("manualTimer");
    doWork();
    timer.end("manualTimer");

    // Scoped timer.
    for (int i = 0; i < 3; ++i) {
      timer.start("scopedMixed");
      doWork();
      timer.end("scopedMixed");
    }

    // Another manual timer.
    timer.start("anotherManual");
    doWork();
    timer.end("anotherManual");
  }

  std::string output = stripAnsi(capture.captured());
  assertTableHeaders(output);

  auto rows = parseTableRows(output);
  ASSERT_EQ(rows.size(), 3) << output;

  auto manual = findRows(rows, "manualTimer");
  ASSERT_EQ(manual.size(), 1) << output;
  EXPECT_EQ(manual[0].runs, 1);
  EXPECT_TRUE(hasValidTimeUnit(manual[0].total)) << manual[0].total;

  auto scoped = findRows(rows, "scopedMixed");
  ASSERT_EQ(scoped.size(), 1) << output;
  EXPECT_EQ(scoped[0].runs, 3);
  EXPECT_TRUE(hasValidTimeUnit(scoped[0].total)) << scoped[0].total;

  auto another = findRows(rows, "anotherManual");
  ASSERT_EQ(another.size(), 1) << output;
  EXPECT_EQ(another[0].runs, 1);
  EXPECT_TRUE(hasValidTimeUnit(another[0].total)) << another[0].total;
}

} // namespace facebook::velox::test
