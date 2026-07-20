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

#include <fmt/core.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/base/VeloxException.h"
#include "velox/common/time/HierarchicalTimer.h"

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

void doWork() {
  int x = 0;
  for (int i = 0; i < 100; ++i) {
    x += i;
  }
  // Prevent the compiler from optimizing away the loop.
  EXPECT_GE(x, 0);
}

/// Builds the expected header block for toString() output.
std::string expectedHeader(
    const std::string& title = "",
    bool verbose = false) {
  constexpr int kCompactWidth = 110;
  constexpr int kVerboseWidth = 176;
  const int totalWidth = verbose ? kVerboseWidth : kCompactWidth;

  std::string out;
  out += std::string(totalWidth, '=') + "\n";
  if (!title.empty()) {
    out += fmt::format("{:^{}s}\n", title, totalWidth);
  }
  out += fmt::format("{:^{}s}\n", "HIERARCHICAL TIMING BREAKDOWN", totalWidth);
  out += std::string(totalWidth, '=') + "\n\n";
  out += fmt::format(
      "{:<60s} {:>10s} {:>8s} {:>10s} {:>8s} {:>8s}",
      "Section",
      "Wall",
      "Wall %",
      "CPU",
      "CPU %",
      "Calls");
  if (verbose) {
    out += fmt::format(
        " {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}",
        "Avg Wall",
        "Min Wall",
        "Max Wall",
        "Avg CPU",
        "Min CPU",
        "Max CPU");
  }
  out += "\n";
  out += std::string(totalWidth, '-') + "\n";
  return out;
}

/// Builds a single expected compact node line (Section, Wall, Wall%, CPU, CPU%,
/// Calls).
std::string expectedLine(
    const std::string& prefix,
    const std::string& name,
    const std::string& wall,
    const std::string& wallPct,
    const std::string& cpu,
    const std::string& cpuPct,
    const std::string& calls) {
  auto section = prefix + name;
  constexpr int kSectionWidth = 60;
  // Use display width for padding (UTF-8 aware).
  int dw = 0;
  for (unsigned char c : section) {
    if ((c & 0xC0) != 0x80) {
      ++dw;
    }
  }
  if (dw < kSectionWidth) {
    section += std::string(kSectionWidth - dw, ' ');
  }
  return fmt::format(
      "{} {:>10s} {:>8s} {:>10s} {:>8s} {:>8s}\n",
      section,
      wall,
      wallPct,
      cpu,
      cpuPct,
      calls);
}

/// Builds a single expected verbose node line (compact columns + avg/min/max).
std::string expectedVerboseLine(
    const std::string& prefix,
    const std::string& name,
    const std::string& wall,
    const std::string& wallPct,
    const std::string& cpu,
    const std::string& cpuPct,
    const std::string& calls,
    const std::string& avgWall,
    const std::string& minWall,
    const std::string& maxWall,
    const std::string& avgCpu,
    const std::string& minCpu,
    const std::string& maxCpu) {
  auto section = prefix + name;
  constexpr int kSectionWidth = 60;
  int dw = 0;
  for (unsigned char c : section) {
    if ((c & 0xC0) != 0x80) {
      ++dw;
    }
  }
  if (dw < kSectionWidth) {
    section += std::string(kSectionWidth - dw, ' ');
  }
  return fmt::format(
      "{} {:>10s} {:>8s} {:>10s} {:>8s} {:>8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}\n",
      section,
      wall,
      wallPct,
      cpu,
      cpuPct,
      calls,
      avgWall,
      minWall,
      maxWall,
      avgCpu,
      minCpu,
      maxCpu);
}

std::string expectedFooter(bool verbose = false) {
  constexpr int kCompactWidth = 110;
  constexpr int kVerboseWidth = 176;
  const int totalWidth = verbose ? kVerboseWidth : kCompactWidth;
  return std::string(totalWidth, '=') + "\n";
}

} // namespace

// =====================================================================
// Tests - Tree structure tests
// =====================================================================

class HierarchicalTimerTest : public ::testing::Test {
 protected:
  TimerTree tree_;
};

TEST_F(HierarchicalTimerTest, hierarchicalNesting) {
  auto* node = tree_.getOrCreateNode("a/b/c");

  const auto& roots = tree_.root().children();
  ASSERT_EQ(1, roots.size());
  EXPECT_EQ("a", roots[0]->name());

  const auto& bChildren = roots[0]->children();
  ASSERT_EQ(1, bChildren.size());
  EXPECT_EQ("b", bChildren[0]->name());

  const auto& cChildren = bChildren[0]->children();
  ASSERT_EQ(1, cChildren.size());
  EXPECT_EQ("c", cChildren[0]->name());
  EXPECT_EQ(node, cChildren[0].get());
}

TEST_F(HierarchicalTimerTest, multipleRoots) {
  tree_.getOrCreateNode("alpha");
  tree_.getOrCreateNode("beta");
  tree_.getOrCreateNode("gamma");

  const auto& roots = tree_.root().children();
  ASSERT_EQ(3, roots.size());
  EXPECT_EQ("alpha", roots[0]->name());
  EXPECT_EQ("beta", roots[1]->name());
  EXPECT_EQ("gamma", roots[2]->name());
}

TEST_F(HierarchicalTimerTest, resetZerosCountersKeepsStructure) {
  tree_.getOrCreateNode("a/b");
  auto* node = tree_.getOrCreateNode("a/b");
  node->addTime(100);
  node->addCpuTime(50);
  node->incrementCallCount();

  tree_.reset();

  const auto& roots = tree_.root().children();
  ASSERT_EQ(1, roots.size());
  EXPECT_EQ("a", roots[0]->name());
  ASSERT_EQ(1, roots[0]->children().size());

  const auto& b = *roots[0]->children()[0];
  EXPECT_EQ(0, b.totalTimeNs());
  EXPECT_EQ(0, b.totalCpuNs());
  EXPECT_EQ(0, b.callCount());
}

TEST_F(HierarchicalTimerTest, clearDestroysTree) {
  tree_.getOrCreateNode("a/b/c");
  tree_.clear();

  EXPECT_TRUE(tree_.root().children().empty());
}

TEST_F(HierarchicalTimerTest, averageTimeIsZeroWhenNoCallsRecorded) {
  auto* node = tree_.getOrCreateNode("empty");
  EXPECT_EQ(0, node->averageTimeNs());
}

TEST_F(HierarchicalTimerTest, emptyTreeToString) {
  const auto output = tree_.toString();
  EXPECT_NE(std::string::npos, output.find("no timers recorded"));
}

// =====================================================================
// Tests - ScopedTimer tests (explicit tree, path-based)
// =====================================================================

TEST_F(HierarchicalTimerTest, basicSingleTimer) {
  {
    ScopedTimer timer(tree_, "op");
    doWork();
  }

  const auto& topChildren = tree_.root().children();
  ASSERT_EQ(1, topChildren.size());

  const auto& node = *topChildren[0];
  EXPECT_EQ("op", node.name());
  EXPECT_EQ(1, node.callCount());
  EXPECT_GT(node.totalTimeNs(), 0);
  EXPECT_GT(node.totalCpuNs(), 0);
  EXPECT_EQ(node.totalTimeNs(), node.averageTimeNs());
}

TEST_F(HierarchicalTimerTest, multipleCallsAccumulate) {
  constexpr int kNumCalls = 5;
  for (int i = 0; i < kNumCalls; ++i) {
    ScopedTimer timer(tree_, "repeated");
    doWork();
  }

  const auto& node = *tree_.root().children()[0];
  EXPECT_EQ(kNumCalls, node.callCount());
  EXPECT_GT(node.totalTimeNs(), 0);
}

TEST_F(HierarchicalTimerTest, nestedScopedTimers) {
  {
    ScopedTimer outer(tree_, "loadStripe");
    doWork();
    {
      ScopedTimer inner(tree_, "loadStripe/readIO");
      doWork();
    }
    {
      ScopedTimer inner(tree_, "loadStripe/decode");
      doWork();
    }
  }

  const auto& roots = tree_.root().children();
  ASSERT_EQ(1, roots.size());

  const auto& loadStripe = *roots[0];
  EXPECT_EQ("loadStripe", loadStripe.name());
  EXPECT_EQ(1, loadStripe.callCount());
  EXPECT_GT(loadStripe.totalTimeNs(), 0);

  const auto& topChildren = loadStripe.children();
  ASSERT_EQ(2, topChildren.size());
  EXPECT_EQ("readIO", topChildren[0]->name());
  EXPECT_EQ("decode", topChildren[1]->name());
  EXPECT_EQ(1, topChildren[0]->callCount());
  EXPECT_EQ(1, topChildren[1]->callCount());

  const uint64_t childrenSum =
      topChildren[0]->totalTimeNs() + topChildren[1]->totalTimeNs();
  EXPECT_GE(loadStripe.totalTimeNs(), childrenSum);
}

TEST_F(HierarchicalTimerTest, deeplyNestedScopedTimers) {
  {
    ScopedTimer t1(tree_, "a");
    {
      ScopedTimer t2(tree_, "a/b");
      {
        ScopedTimer t3(tree_, "a/b/c");
        doWork();
      }
    }
  }

  const auto& a = *tree_.root().children()[0];
  EXPECT_EQ("a", a.name());
  EXPECT_EQ(1, a.callCount());

  const auto& b = *a.children()[0];
  EXPECT_EQ("b", b.name());
  EXPECT_EQ(1, b.callCount());

  const auto& c = *b.children()[0];
  EXPECT_EQ("c", c.name());
  EXPECT_EQ(1, c.callCount());
  EXPECT_GT(c.totalTimeNs(), 0);

  EXPECT_GE(b.totalTimeNs(), c.totalTimeNs());
  EXPECT_GE(a.totalTimeNs(), b.totalTimeNs());
}

TEST_F(HierarchicalTimerTest, nestedScopedTimersMultipleIterations) {
  constexpr int kIterations = 3;
  for (int i = 0; i < kIterations; ++i) {
    ScopedTimer outer(tree_, "loop");
    {
      ScopedTimer inner(tree_, "loop/work");
      doWork();
    }
  }

  const auto& loop = *tree_.root().children()[0];
  EXPECT_EQ(kIterations, loop.callCount());

  const auto& work = *loop.children()[0];
  EXPECT_EQ(kIterations, work.callCount());
  EXPECT_GE(loop.totalTimeNs(), work.totalTimeNs());
}

TEST_F(HierarchicalTimerTest, multipleSameLevelScopedTimers) {
  {
    ScopedTimer outer(tree_, "process");
    {
      ScopedTimer s1(tree_, "process/parse");
      doWork();
    }
    {
      ScopedTimer s2(tree_, "process/validate");
      doWork();
    }
    {
      ScopedTimer s3(tree_, "process/transform");
      doWork();
    }
    {
      ScopedTimer s4(tree_, "process/serialize");
      doWork();
    }
  }

  const auto& process = *tree_.root().children()[0];
  EXPECT_EQ("process", process.name());
  EXPECT_EQ(1, process.callCount());

  const auto& topChildren = process.children();
  ASSERT_EQ(4, topChildren.size());

  const std::vector<std::string> kExpectedNames{
      "parse", "validate", "transform", "serialize"};
  for (size_t i = 0; i < topChildren.size(); ++i) {
    EXPECT_EQ(kExpectedNames[i], topChildren[i]->name());
    EXPECT_EQ(1, topChildren[i]->callCount());
    EXPECT_GT(topChildren[i]->totalTimeNs(), 0);
  }

  uint64_t childrenSum = 0;
  for (const auto& child : topChildren) {
    childrenSum += child->totalTimeNs();
  }
  EXPECT_GE(process.totalTimeNs(), childrenSum);
}

TEST_F(HierarchicalTimerTest, sameLevelScopedTimersWithNesting) {
  {
    ScopedTimer outer(tree_, "root");
    {
      ScopedTimer s1(tree_, "root/io");
      {
        ScopedTimer s1a(tree_, "root/io/read");
        doWork();
      }
      {
        ScopedTimer s1b(tree_, "root/io/write");
        doWork();
      }
    }
    {
      ScopedTimer s2(tree_, "root/compute");
      {
        ScopedTimer s2a(tree_, "root/compute/map");
        doWork();
      }
      {
        ScopedTimer s2b(tree_, "root/compute/reduce");
        doWork();
      }
    }
  }

  const auto& rootNode = *tree_.root().children()[0];
  const auto& topChildren = rootNode.children();
  ASSERT_EQ(2, topChildren.size());

  const auto& io = *topChildren[0];
  EXPECT_EQ("io", io.name());
  ASSERT_EQ(2, io.children().size());
  EXPECT_EQ("read", io.children()[0]->name());
  EXPECT_EQ("write", io.children()[1]->name());

  const auto& compute = *topChildren[1];
  EXPECT_EQ("compute", compute.name());
  ASSERT_EQ(2, compute.children().size());
  EXPECT_EQ("map", compute.children()[0]->name());
  EXPECT_EQ("reduce", compute.children()[1]->name());

  const uint64_t ioChildrenSum =
      io.children()[0]->totalTimeNs() + io.children()[1]->totalTimeNs();
  EXPECT_GE(io.totalTimeNs(), ioChildrenSum);

  const uint64_t computeChildrenSum = compute.children()[0]->totalTimeNs() +
      compute.children()[1]->totalTimeNs();
  EXPECT_GE(compute.totalTimeNs(), computeChildrenSum);
}

TEST_F(HierarchicalTimerTest, scopedTimerIsNonCopyableNonMovable) {
  EXPECT_FALSE(std::is_copy_constructible_v<ScopedTimer>);
  EXPECT_FALSE(std::is_copy_assignable_v<ScopedTimer>);
  EXPECT_FALSE(std::is_move_constructible_v<ScopedTimer>);
  EXPECT_FALSE(std::is_move_assignable_v<ScopedTimer>);
}

// =====================================================================
// Tests - toString exact output tests
// =====================================================================

TEST_F(HierarchicalTimerTest, toStringFormattedOutputSingleNode) {
  auto* node = tree_.getOrCreateNode("op");
  node->addTime(5'000'000); // 5ms
  node->incrementCallCount();

  std::string expected = expectedHeader();
  expected += expectedLine("", "op", "5.00ms", "-", "0ns", "-", "1");
  expected += expectedFooter();

  EXPECT_EQ(expected, tree_.toString());
}

TEST_F(HierarchicalTimerTest, toStringFormattedOutputWithChildren) {
  auto* parent = tree_.getOrCreateNode("loadStripe");
  parent->addTime(100'000'000); // 100ms
  parent->incrementCallCount();

  auto* child1 = tree_.getOrCreateNode("loadStripe/readIO");
  child1->addTime(60'000'000); // 60ms
  child1->incrementCallCount();

  auto* child2 = tree_.getOrCreateNode("loadStripe/decode");
  child2->addTime(30'000'000); // 30ms
  child2->incrementCallCount();

  std::string expected = expectedHeader();
  expected += expectedLine("", "loadStripe", "100.00ms", "-", "0ns", "-", "1");
  expected +=
      expectedLine("├── ", "readIO", "60.00ms", "60.0%", "0ns", "-", "1");
  expected +=
      expectedLine("├── ", "decode", "30.00ms", "30.0%", "0ns", "-", "1");
  expected +=
      expectedLine("└── ", "(other)", "10.00ms", "10.0%", "0ns", "-", "-");
  expected += expectedFooter();

  EXPECT_EQ(expected, tree_.toString());
}

TEST_F(HierarchicalTimerTest, toStringFormattedOutputDeepNesting) {
  auto* a = tree_.getOrCreateNode("a");
  a->addTime(1'000'000'000); // 1s
  a->incrementCallCount();

  auto* b = tree_.getOrCreateNode("a/b");
  b->addTime(500'000'000); // 500ms
  b->incrementCallCount();
  b->incrementCallCount();

  auto* c = tree_.getOrCreateNode("a/b/c");
  c->addTime(200'000'000); // 200ms
  c->incrementCallCount();
  c->incrementCallCount();
  c->incrementCallCount();
  c->incrementCallCount();

  std::string expected = expectedHeader();
  expected += expectedLine("", "a", "1.00s", "-", "0ns", "-", "1");
  expected += expectedLine("├── ", "b", "500.00ms", "50.0%", "0ns", "-", "2");
  expected +=
      expectedLine("│   ├── ", "c", "200.00ms", "40.0%", "0ns", "-", "4");
  expected +=
      expectedLine("│   └── ", "(other)", "300.00ms", "60.0%", "0ns", "-", "-");
  expected +=
      expectedLine("└── ", "(other)", "500.00ms", "50.0%", "0ns", "-", "-");
  expected += expectedFooter();

  EXPECT_EQ(expected, tree_.toString());
}

TEST_F(HierarchicalTimerTest, toStringFormattedOutputMultipleRoots) {
  auto* alpha = tree_.getOrCreateNode("alpha");
  alpha->addTime(10'000'000); // 10ms
  alpha->incrementCallCount();

  auto* beta = tree_.getOrCreateNode("beta");
  beta->addTime(20'000'000); // 20ms
  beta->incrementCallCount();

  std::string expected = expectedHeader();
  expected += expectedLine("", "alpha", "10.00ms", "-", "0ns", "-", "1");
  expected += expectedLine("", "beta", "20.00ms", "-", "0ns", "-", "1");
  expected += expectedFooter();

  EXPECT_EQ(expected, tree_.toString());
}

TEST_F(
    HierarchicalTimerTest,
    toStringFormattedOutputMultipleSameLevelChildren) {
  auto* parent = tree_.getOrCreateNode("pipeline");
  parent->addTime(200'000'000); // 200ms
  parent->incrementCallCount();

  auto* c1 = tree_.getOrCreateNode("pipeline/parse");
  c1->addTime(50'000'000); // 50ms
  c1->incrementCallCount();

  auto* c2 = tree_.getOrCreateNode("pipeline/validate");
  c2->addTime(30'000'000); // 30ms
  c2->incrementCallCount();

  auto* c3 = tree_.getOrCreateNode("pipeline/transform");
  c3->addTime(80'000'000); // 80ms
  c3->incrementCallCount();

  auto* c4 = tree_.getOrCreateNode("pipeline/emit");
  c4->addTime(20'000'000); // 20ms
  c4->incrementCallCount();

  std::string expected = expectedHeader();
  expected += expectedLine("", "pipeline", "200.00ms", "-", "0ns", "-", "1");
  expected +=
      expectedLine("├── ", "parse", "50.00ms", "25.0%", "0ns", "-", "1");
  expected +=
      expectedLine("├── ", "validate", "30.00ms", "15.0%", "0ns", "-", "1");
  expected +=
      expectedLine("├── ", "transform", "80.00ms", "40.0%", "0ns", "-", "1");
  expected += expectedLine("├── ", "emit", "20.00ms", "10.0%", "0ns", "-", "1");
  expected +=
      expectedLine("└── ", "(other)", "20.00ms", "10.0%", "0ns", "-", "-");
  expected += expectedFooter();

  EXPECT_EQ(expected, tree_.toString());
}

// =====================================================================
// Tests - Standalone tests
// =====================================================================

TEST(HierarchicalTimerStandaloneTest, noEntriesNoOutput) {
  LogCapture capture;
  {
    TimerTree tree;
  }
  EXPECT_TRUE(capture.captured().empty()) << capture.captured();
}

TEST(HierarchicalTimerStandaloneTest, destructorPrintsTitleAndResults) {
  LogCapture capture;

  {
    TimerTree tree("MyBenchmarkTitle");
    {
      ScopedTimer t(tree, "sessionTimer");
      doWork();
    }
  } // Destructor prints the table.

  const auto& output = capture.captured();
  EXPECT_NE(output.find("MyBenchmarkTitle"), std::string::npos) << output;
  EXPECT_NE(output.find("sessionTimer"), std::string::npos) << output;
  EXPECT_NE(output.find("HIERARCHICAL TIMING BREAKDOWN"), std::string::npos)
      << output;
}

TEST(HierarchicalTimerStandaloneTest, printStatsClearsEntries) {
  LogCapture capture1;
  TimerTree tree("Session");

  {
    ScopedTimer t(tree, "firstTimer");
    doWork();
  }
  tree.printStats();

  const auto& output1 = capture1.captured();
  EXPECT_NE(output1.find("Session"), std::string::npos) << output1;
  EXPECT_NE(output1.find("firstTimer"), std::string::npos) << output1;

  // After printStats(), entries should be cleared.
  // Destructor should produce no further output.
  LogCapture capture2;
  tree.printStats();
  EXPECT_TRUE(capture2.captured().empty()) << capture2.captured();
}

// =====================================================================
// Tests - min/max tracking and CPU time
// =====================================================================

TEST_F(HierarchicalTimerTest, minMaxTracking) {
  auto* node = tree_.getOrCreateNode("tracked");
  node->addTime(100);
  node->incrementCallCount();
  node->addTime(500);
  node->incrementCallCount();
  node->addTime(200);
  node->incrementCallCount();

  EXPECT_EQ(800, node->totalTimeNs());
  EXPECT_EQ(3, node->callCount());
  EXPECT_EQ(100, node->minTimeNs());
  EXPECT_EQ(500, node->maxTimeNs());
  EXPECT_EQ(266, node->averageTimeNs()); // 800 / 3 = 266
}

TEST_F(HierarchicalTimerTest, toStringIncludesWallCpuColumns) {
  auto* node = tree_.getOrCreateNode("op");
  node->addTime(5'000'000); // 5ms
  node->incrementCallCount();

  const auto output = tree_.toString();
  EXPECT_NE(output.find("Wall"), std::string::npos) << output;
  EXPECT_NE(output.find("CPU"), std::string::npos) << output;
  EXPECT_NE(output.find("Wall %"), std::string::npos) << output;
  EXPECT_NE(output.find("CPU %"), std::string::npos) << output;

  // Verbose columns should not appear in compact mode.
  const auto verboseOutput = tree_.toString(true);
  EXPECT_NE(verboseOutput.find("Min Wall"), std::string::npos) << verboseOutput;
  EXPECT_NE(verboseOutput.find("Max Wall"), std::string::npos) << verboseOutput;
  EXPECT_NE(verboseOutput.find("Avg CPU"), std::string::npos) << verboseOutput;
  EXPECT_NE(verboseOutput.find("Min CPU"), std::string::npos) << verboseOutput;
  EXPECT_NE(verboseOutput.find("Max CPU"), std::string::npos) << verboseOutput;
}

TEST_F(HierarchicalTimerTest, toStringWithTitle) {
  TimerTree namedTree("MyBench");
  auto* node = namedTree.getOrCreateNode("op");
  node->addTime(1'000'000);
  node->incrementCallCount();

  const auto output = namedTree.toString();
  EXPECT_NE(output.find("MyBench"), std::string::npos) << output;
  EXPECT_NE(output.find("HIERARCHICAL TIMING BREAKDOWN"), std::string::npos)
      << output;
}

TEST_F(HierarchicalTimerTest, resetAlsoResetsMinMaxAndCpu) {
  auto* node = tree_.getOrCreateNode("tracked");
  node->addTime(100);
  node->addCpuTime(50);
  node->incrementCallCount();

  EXPECT_EQ(100, node->minTimeNs());
  EXPECT_EQ(100, node->maxTimeNs());
  EXPECT_EQ(50, node->totalCpuNs());
  EXPECT_EQ(50, node->minCpuNs());
  EXPECT_EQ(50, node->maxCpuNs());

  tree_.reset();

  EXPECT_EQ(std::numeric_limits<uint64_t>::max(), node->minTimeNs());
  EXPECT_EQ(0, node->maxTimeNs());
  EXPECT_EQ(0, node->totalCpuNs());
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(), node->minCpuNs());
  EXPECT_EQ(0, node->maxCpuNs());
}

TEST_F(HierarchicalTimerTest, cpuPerCallStats) {
  auto* node = tree_.getOrCreateNode("tracked");
  node->addCpuTime(100);
  node->addTime(200);
  node->incrementCallCount();
  node->addCpuTime(500);
  node->addTime(600);
  node->incrementCallCount();
  node->addCpuTime(300);
  node->addTime(400);
  node->incrementCallCount();

  EXPECT_EQ(900, node->totalCpuNs());
  EXPECT_EQ(3, node->callCount());
  EXPECT_EQ(300, node->averageCpuNs()); // 900 / 3
  EXPECT_EQ(100, node->minCpuNs());
  EXPECT_EQ(500, node->maxCpuNs());
}

TEST_F(HierarchicalTimerTest, toStringFormattedOutputWithCpuPerCall) {
  auto* node = tree_.getOrCreateNode("op");
  node->addTime(10'000'000); // 10ms wall
  node->addCpuTime(5'000'000); // 5ms CPU
  node->incrementCallCount();
  node->addTime(20'000'000); // 20ms wall
  node->addCpuTime(15'000'000); // 15ms CPU
  node->incrementCallCount();

  // Compact mode: no avg/min/max columns.
  std::string expected = expectedHeader();
  expected += expectedLine("", "op", "30.00ms", "-", "20.00ms", "-", "2");
  expected += expectedFooter();

  EXPECT_EQ(expected, tree_.toString());
}

TEST_F(HierarchicalTimerTest, toStringVerboseOutput) {
  auto* node = tree_.getOrCreateNode("op");
  node->addTime(10'000'000); // 10ms wall
  node->addCpuTime(5'000'000); // 5ms CPU
  node->incrementCallCount();
  node->addTime(20'000'000); // 20ms wall
  node->addCpuTime(15'000'000); // 15ms CPU
  node->incrementCallCount();

  std::string expected = expectedHeader("", true);
  expected += expectedVerboseLine(
      "",
      "op",
      "30.00ms", // total wall
      "-", // wall %
      "20.00ms", // total CPU
      "-", // cpu %
      "2",
      "15.00ms", // avg wall: 30/2
      "10.00ms", // min wall
      "20.00ms", // max wall
      "10.00ms", // avg CPU: 20/2
      "5.00ms", // min CPU
      "15.00ms"); // max CPU
  expected += expectedFooter(true);

  EXPECT_EQ(expected, tree_.toString(true));
}

TEST_F(HierarchicalTimerTest, cpuNowReturnsNonZero) {
  EXPECT_GT(tree_.cpuNow(), 0);
}

TEST_F(HierarchicalTimerTest, cpuTimeTrackedByScopedTimer) {
  {
    ScopedTimer t(tree_, "cpuWork");
    doWork();
  }

  const auto& roots = tree_.root().children();
  ASSERT_EQ(1, roots.size());
  EXPECT_GT(roots[0]->totalTimeNs(), 0);
  EXPECT_GT(roots[0]->totalCpuNs(), 0);
}

// =====================================================================
// Thread-local TimerTree tests
// =====================================================================

TEST(HierarchicalTimerThreadLocalTest, scopedTimerUsesThreadInstance) {
  auto& tree = TimerTree::threadInstance();
  tree.clear();

  {
    ScopedTimer t("threadLocalOp");
    doWork();
  }

  const auto& roots = tree.root().children();
  ASSERT_EQ(1, roots.size());
  EXPECT_EQ("threadLocalOp", roots[0]->name());
  EXPECT_EQ(1, roots[0]->callCount());
  EXPECT_GT(roots[0]->totalTimeNs(), 0);
  EXPECT_GT(roots[0]->totalCpuNs(), 0);
  tree.clear();
}

TEST(HierarchicalTimerThreadLocalTest, threadInstanceIsSameAcrossCalls) {
  auto& tree1 = TimerTree::threadInstance();
  auto& tree2 = TimerTree::threadInstance();
  EXPECT_EQ(&tree1, &tree2);
}

// Simulates two different classes recording into the same thread-local tree
// using stack-based auto-nesting.
namespace {
void classAWork() {
  ScopedTimer t1("classA");
  {
    ScopedTimer t2("process");
    doWork();
  }
}

void classBWork() {
  ScopedTimer t1("classB");
  {
    ScopedTimer t2("compute");
    doWork();
  }
}
} // namespace

TEST(HierarchicalTimerThreadLocalTest, sharedAcrossClasses) {
  auto& tree = TimerTree::threadInstance();
  tree.clear();

  classAWork();
  classBWork();
  classAWork();

  const auto& roots = tree.root().children();
  ASSERT_EQ(2, roots.size());
  EXPECT_EQ("classA", roots[0]->name());
  EXPECT_EQ("classB", roots[1]->name());

  EXPECT_EQ(2, roots[0]->callCount());
  const auto& classAChildren = roots[0]->children();
  ASSERT_EQ(1, classAChildren.size());
  EXPECT_EQ("process", classAChildren[0]->name());
  EXPECT_EQ(2, classAChildren[0]->callCount());

  EXPECT_EQ(1, roots[1]->callCount());
  const auto& classBChildren = roots[1]->children();
  ASSERT_EQ(1, classBChildren.size());
  EXPECT_EQ("compute", classBChildren[0]->name());
  EXPECT_EQ(1, classBChildren[0]->callCount());
  tree.clear();
}

TEST(HierarchicalTimerThreadLocalTest, nestedThreadLocalTimers) {
  auto& tree = TimerTree::threadInstance();
  tree.clear();

  {
    ScopedTimer outer("pipeline");
    {
      ScopedTimer inner("step1");
      doWork();
    }
    {
      ScopedTimer inner("step2");
      doWork();
    }
  }

  const auto& roots = tree.root().children();
  ASSERT_EQ(1, roots.size());
  EXPECT_EQ("pipeline", roots[0]->name());
  EXPECT_EQ(1, roots[0]->callCount());
  ASSERT_EQ(2, roots[0]->children().size());
  EXPECT_EQ("step1", roots[0]->children()[0]->name());
  EXPECT_EQ("step2", roots[0]->children()[1]->name());
  tree.clear();
}

// =====================================================================
// Tests - Stack-based auto-nesting
// =====================================================================

TEST(HierarchicalTimerAutoNestingTest, basicAutoNesting) {
  auto& tree = TimerTree::threadInstance();
  tree.clear();

  {
    ScopedTimer t1("benchmark");
    {
      ScopedTimer t2("read");
      {
        ScopedTimer t3("decode");
        doWork();
      }
    }
  }

  const auto& roots = tree.root().children();
  ASSERT_EQ(1, roots.size());
  EXPECT_EQ("benchmark", roots[0]->name());
  EXPECT_EQ(1, roots[0]->callCount());

  const auto& readChildren = roots[0]->children();
  ASSERT_EQ(1, readChildren.size());
  EXPECT_EQ("read", readChildren[0]->name());
  EXPECT_EQ(1, readChildren[0]->callCount());

  const auto& decodeChildren = readChildren[0]->children();
  ASSERT_EQ(1, decodeChildren.size());
  EXPECT_EQ("decode", decodeChildren[0]->name());
  EXPECT_EQ(1, decodeChildren[0]->callCount());
  EXPECT_GT(decodeChildren[0]->totalTimeNs(), 0);

  EXPECT_GE(readChildren[0]->totalTimeNs(), decodeChildren[0]->totalTimeNs());
  EXPECT_GE(roots[0]->totalTimeNs(), readChildren[0]->totalTimeNs());
  tree.clear();
}

TEST(HierarchicalTimerAutoNestingTest, autoNestingRestoresActiveNode) {
  auto& tree = TimerTree::threadInstance();
  tree.clear();

  {
    ScopedTimer t1("root");
    {
      ScopedTimer t2("child1");
      doWork();
    }
    // After child1 is destroyed, active node should be restored to "root".
    {
      ScopedTimer t3("child2");
      doWork();
    }
  }

  const auto& roots = tree.root().children();
  ASSERT_EQ(1, roots.size());
  EXPECT_EQ("root", roots[0]->name());

  // Both children should be siblings under "root", not nested.
  const auto& children = roots[0]->children();
  ASSERT_EQ(2, children.size());
  EXPECT_EQ("child1", children[0]->name());
  EXPECT_EQ("child2", children[1]->name());
  tree.clear();
}

TEST(HierarchicalTimerAutoNestingTest, sequentialTopLevelTimers) {
  auto& tree = TimerTree::threadInstance();
  tree.clear();

  {
    ScopedTimer t1("alpha");
    doWork();
  }
  {
    ScopedTimer t2("beta");
    doWork();
  }

  // Sequential (non-nested) timers should create separate top-level entries.
  const auto& roots = tree.root().children();
  ASSERT_EQ(2, roots.size());
  EXPECT_EQ("alpha", roots[0]->name());
  EXPECT_EQ("beta", roots[1]->name());
  tree.clear();
}

TEST(HierarchicalTimerAutoNestingTest, autoNestingCpuTimeTracked) {
  auto& tree = TimerTree::threadInstance();
  tree.clear();

  {
    ScopedTimer t1("outer");
    {
      ScopedTimer t2("inner");
      doWork();
    }
  }

  const auto& roots = tree.root().children();
  ASSERT_EQ(1, roots.size());
  EXPECT_GT(roots[0]->totalCpuNs(), 0);
  EXPECT_GT(roots[0]->children()[0]->totalCpuNs(), 0);
  tree.clear();
}

} // namespace facebook::velox::test
