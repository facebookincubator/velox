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

#include "velox/common/time/HierarchicalTimer.h"

#include <chrono>
#include <ctime>
#include <iomanip>

#include <fmt/core.h>
#include <glog/logging.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SuccinctPrinter.h"

#ifdef ENABLE_HW_TIMER
#if !defined(__x86_64__)
#error "HierarchicalTimer RDTSCP mode requires x86-64 architecture"
#endif
#include <immintrin.h>
#include "folly/chrono/Hardware.h"
#endif

namespace facebook::velox {

namespace {

#ifdef ENABLE_HW_TIMER
double estimateTscFreqGhz(
    std::chrono::milliseconds window = std::chrono::milliseconds(100)) {
  static const double cached = [&]() {
    using clock = std::chrono::steady_clock;
    // Warm-up (reduces first-use noise).
    (void)folly::hardware_timestamp();
    auto t0 = clock::now();
    uint64_t c0 = folly::hardware_timestamp();
    // Busy-wait until window elapsed (less jitter than sleep).
    while (clock::now() - t0 < window) {
      _mm_pause();
    }
    auto t1 = clock::now();
    uint64_t c1 = folly::hardware_timestamp();
    std::chrono::duration<double> seconds = t1 - t0;
    return (static_cast<double>(c1 - c0) / seconds.count()) / 1e9;
  }();
  return cached;
}
#endif

/// Tracks the currently active TimerNode for stack-based auto-nesting.
thread_local TimerNode* activeNode_ =
    nullptr; // NOLINT(facebook-avoid-non-const-global-variables)

/// Section column width for the formatted tree output.
/// Must accommodate prefix (4 chars per depth level) plus node name.
constexpr int kSectionWidth = 60;

std::string buildPrefix(const std::vector<bool>& ancestorIsLast) {
  if (ancestorIsLast.empty()) {
    return "";
  }
  std::string prefix;
  // Draw continuation lines for ancestors (all but the current node).
  for (size_t i = 0; i + 1 < ancestorIsLast.size(); ++i) {
    prefix += ancestorIsLast[i] ? "    " : "│   ";
  }
  // Draw the branch for the current node.
  prefix += ancestorIsLast.back() ? "└── " : "├── ";
  return prefix;
}

/// Computes display width by counting UTF-8 codepoints (not bytes).
/// Box-drawing chars (│├└─) are 3 bytes each but 1 display column.
int displayWidth(const std::string& s) {
  int width = 0;
  for (unsigned char c : s) {
    // Count only non-continuation bytes (not 0x80-0xBF).
    if ((c & 0xC0) != 0x80) {
      ++width;
    }
  }
  return width;
}

/// Pads the section string to kSectionWidth display columns for alignment.
std::string padSection(const std::string& section) {
  const int dw = displayWidth(section);
  if (dw >= kSectionWidth) {
    return section;
  }
  return section + std::string(kSectionWidth - dw, ' ');
}

} // namespace

// --- TimerNode ---

TimerNode::TimerNode(const std::string& name, TimerNode* parent)
    : name_{name}, parent_{parent} {}

TimerNode* TimerNode::getOrCreateChild(const std::string& childName) {
  auto it = childrenByName_.find(childName);
  if (it != childrenByName_.end()) {
    return it->second;
  }
  auto child = std::make_unique<TimerNode>(childName, this);
  auto* raw = child.get();
  childrenByName_[childName] = raw;
  children_.push_back(std::move(child));
  return raw;
}

void TimerNode::addTime(uint64_t ns) {
  totalTimeNs_ += ns;
  if (ns < minTimeNs_) {
    minTimeNs_ = ns;
  }
  if (ns > maxTimeNs_) {
    maxTimeNs_ = ns;
  }
}

void TimerNode::addCpuTime(uint64_t ns) {
  totalCpuNs_ += ns;
  if (ns < minCpuNs_) {
    minCpuNs_ = ns;
  }
  if (ns > maxCpuNs_) {
    maxCpuNs_ = ns;
  }
}

void TimerNode::incrementCallCount() {
  ++callCount_;
}

void TimerNode::reset() {
  totalTimeNs_ = 0;
  totalCpuNs_ = 0;
  callCount_ = 0;
  minTimeNs_ = std::numeric_limits<uint64_t>::max();
  maxTimeNs_ = 0;
  minCpuNs_ = std::numeric_limits<uint64_t>::max();
  maxCpuNs_ = 0;
  for (auto& child : children_) {
    child->reset();
  }
}

uint64_t TimerNode::totalTimeNs() const {
  return totalTimeNs_;
}

uint64_t TimerNode::totalCpuNs() const {
  return totalCpuNs_;
}

uint64_t TimerNode::callCount() const {
  return callCount_;
}

uint64_t TimerNode::averageTimeNs() const {
  if (callCount_ == 0) {
    return 0;
  }
  return totalTimeNs_ / callCount_;
}

uint64_t TimerNode::minTimeNs() const {
  return minTimeNs_;
}

uint64_t TimerNode::maxTimeNs() const {
  return maxTimeNs_;
}

uint64_t TimerNode::averageCpuNs() const {
  if (callCount_ == 0) {
    return 0;
  }
  return totalCpuNs_ / callCount_;
}

uint64_t TimerNode::minCpuNs() const {
  return minCpuNs_;
}

uint64_t TimerNode::maxCpuNs() const {
  return maxCpuNs_;
}

const std::string& TimerNode::name() const {
  return name_;
}

TimerNode* TimerNode::parent() const {
  return parent_;
}

const std::vector<std::unique_ptr<TimerNode>>& TimerNode::children() const {
  return children_;
}

void TimerNode::format(
    std::string& out,
    int depth,
    bool isLast,
    uint64_t parentTimeNs,
    uint64_t parentCpuTimeNs,
    bool verbose) const {
  std::vector<bool> ancestorTrail;
  formatImpl(
      out,
      depth,
      isLast,
      parentTimeNs,
      parentCpuTimeNs,
      verbose,
      ancestorTrail);
}

void TimerNode::formatImpl(
    std::string& out,
    int depth,
    bool isLast,
    uint64_t parentTimeNs,
    uint64_t parentCpuTimeNs,
    bool verbose,
    std::vector<bool>& ancestorTrail) const {
  if (depth > 0) {
    ancestorTrail.push_back(isLast);
  }
  const auto prefix = buildPrefix(ancestorTrail);
  const auto wall = succinctNanos(totalTimeNs_);
  const auto cpu = succinctNanos(totalCpuNs_);
  const auto calls =
      callCount_ > 0 ? std::to_string(callCount_) : std::string{"-"};

  std::string wallPct;
  if (parentTimeNs > 0 && depth > 0) {
    wallPct = fmt::format(
        "{:.1f}%",
        100.0 * static_cast<double>(totalTimeNs_) /
            static_cast<double>(parentTimeNs));
  } else {
    wallPct = "-";
  }

  std::string cpuPct;
  if (parentCpuTimeNs > 0 && depth > 0) {
    cpuPct = fmt::format(
        "{:.1f}%",
        100.0 * static_cast<double>(totalCpuNs_) /
            static_cast<double>(parentCpuTimeNs));
  } else {
    cpuPct = "-";
  }

  const auto section = padSection(prefix + name_);

  out += fmt::format(
      "{} {:>10s} {:>8s} {:>10s} {:>8s} {:>8s}",
      section,
      wall,
      wallPct,
      cpu,
      cpuPct,
      calls);

  if (verbose) {
    std::string avgWall, minWall, maxWall;
    std::string avgCpu, minCpu, maxCpu;
    if (callCount_ > 0) {
      avgWall = succinctNanos(averageTimeNs());
      minWall = succinctNanos(minTimeNs_);
      maxWall = succinctNanos(maxTimeNs_);
      avgCpu = succinctNanos(averageCpuNs());
      minCpu = succinctNanos(maxCpuNs_ > 0 ? minCpuNs_ : 0);
      maxCpu = succinctNanos(maxCpuNs_);
    } else {
      avgWall = "-";
      minWall = "-";
      maxWall = "-";
      avgCpu = "-";
      minCpu = "-";
      maxCpu = "-";
    }
    out += fmt::format(
        " {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}",
        avgWall,
        minWall,
        maxWall,
        avgCpu,
        minCpu,
        maxCpu);
  }
  out += "\n";

  // Determine if "(other)" row will be shown so we can set isLast correctly
  // for the final real child.
  uint64_t childrenSum = 0;
  uint64_t childrenCpuSum = 0;
  for (const auto& child : children_) {
    childrenSum += child->totalTimeNs();
    childrenCpuSum += child->totalCpuNs();
  }
  const bool hasOther = !children_.empty() && totalTimeNs_ > childrenSum;

  // Format children.
  for (size_t i = 0; i < children_.size(); ++i) {
    const bool lastChild = (i == children_.size() - 1) && !hasOther;
    children_[i]->formatImpl(
        out,
        depth + 1,
        lastChild,
        totalTimeNs_,
        totalCpuNs_,
        verbose,
        ancestorTrail);
  }

  // Show unaccounted time if this node has children and time exceeds children
  // sum.
  if (hasOther) {
    const uint64_t other = totalTimeNs_ - childrenSum;
    const uint64_t cpuOther =
        (totalCpuNs_ > childrenCpuSum) ? totalCpuNs_ - childrenCpuSum : 0;
    ancestorTrail.push_back(true);
    const auto otherPrefix = buildPrefix(ancestorTrail);
    const auto otherSection = padSection(otherPrefix + "(other)");
    const auto otherWallPct = fmt::format(
        "{:.1f}%",
        100.0 * static_cast<double>(other) / static_cast<double>(totalTimeNs_));
    std::string otherCpuPct;
    if (totalCpuNs_ > 0) {
      otherCpuPct = fmt::format(
          "{:.1f}%",
          100.0 * static_cast<double>(cpuOther) /
              static_cast<double>(totalCpuNs_));
    } else {
      otherCpuPct = "-";
    }
    out += fmt::format(
        "{} {:>10s} {:>8s} {:>10s} {:>8s} {:>8s}",
        otherSection,
        succinctNanos(other),
        otherWallPct,
        succinctNanos(cpuOther),
        otherCpuPct,
        "-");
    if (verbose) {
      out += fmt::format(
          " {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}",
          "-",
          "-",
          "-",
          "-",
          "-",
          "-");
    }
    out += "\n";
    ancestorTrail.pop_back();
  }

  if (depth > 0 && !ancestorTrail.empty()) {
    ancestorTrail.pop_back();
  }
}

// --- TimerTree ---

TimerTree::TimerTree(const std::string& name)
    : name_{name}, root_{std::make_unique<TimerNode>("root")} {}

TimerTree::~TimerTree() {
  printStats();
}

uint64_t TimerTree::now() const {
#ifdef ENABLE_HW_TIMER
  return static_cast<uint64_t>(
      static_cast<double>(folly::hardware_timestamp()) / estimateTscFreqGhz());
#else
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
#endif
}

uint64_t TimerTree::cpuNow() const {
  struct timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
      static_cast<uint64_t>(ts.tv_nsec);
}

TimerNode* TimerTree::getOrCreateNode(const std::string& path) {
  TimerNode* current = root_.get();
  size_t start = 0;
  while (start < path.size()) {
    auto pos = path.find('/', start);
    if (pos == std::string::npos) {
      pos = path.size();
    }
    const auto segment = path.substr(start, pos - start);
    if (!segment.empty()) {
      current = current->getOrCreateChild(segment);
    }
    start = pos + 1;
  }
  return current;
}

void TimerTree::reset() {
  root_->reset();
}

void TimerTree::clear() {
  root_ = std::make_unique<TimerNode>("root");
}

std::string TimerTree::toString(bool verbose) const {
  const auto& topChildren = root_->children();
  if (topChildren.empty()) {
    return "(no timers recorded)\n";
  }

  // Compact: Section(60) + Wall(10) + Wall%(8) + CPU(10) + CPU%(8) + Calls(8)
  //          + 6 separating spaces = 110.
  // Verbose adds: AvgWall(10) + MinWall(10) + MaxWall(10) + AvgCPU(10) +
  //               MinCPU(10) + MaxCPU(10) + 6 separating spaces = 66.
  constexpr int kCompactWidth = 110;
  constexpr int kVerboseWidth = 176;
  const int totalWidth = verbose ? kVerboseWidth : kCompactWidth;

  std::string out;
  const std::string separator(totalWidth, '=');
  const std::string dashSeparator(totalWidth, '-');

  out += separator + "\n";
  if (!name_.empty()) {
    out += fmt::format("{:^{}s}\n", name_, totalWidth);
  }
  out += fmt::format("{:^{}s}\n", "HIERARCHICAL TIMING BREAKDOWN", totalWidth);
  out += separator + "\n\n";

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
  out += dashSeparator + "\n";

  for (size_t i = 0; i < topChildren.size(); ++i) {
    topChildren[i]->format(
        out,
        /*depth=*/0,
        /*isLast=*/false,
        /*parentTimeNs=*/0,
        /*parentCpuTimeNs=*/0,
        verbose);
  }

  out += separator + "\n";
  return out;
}

void TimerTree::printStats(bool verbose) {
  const auto& topChildren = root_->children();
  if (topChildren.empty()) {
    return;
  }
  LOG(INFO) << "\n" << toString(verbose);
  clear();
}

const TimerNode& TimerTree::root() const {
  return *root_;
}

const std::string& TimerTree::name() const {
  return name_;
}

TimerTree& TimerTree::threadInstance() {
  static thread_local TimerTree instance;
  return instance;
}

// --- ScopedTimer ---

ScopedTimer::ScopedTimer(const std::string& name)
    : tree_(TimerTree::threadInstance()),
      restoreActive_(true),
      startNs_(tree_.now()),
      cpuStartNs_(tree_.cpuNow()) {
  previousActive_ = activeNode_;
  if (activeNode_) {
    node_ = activeNode_->getOrCreateChild(name);
  } else {
    node_ = tree_.root_->getOrCreateChild(name);
  }
  activeNode_ = node_;
  node_->incrementCallCount();
}

ScopedTimer::ScopedTimer(TimerTree& tree, const std::string& path)
    : tree_(tree),
      node_(tree.getOrCreateNode(path)),
      restoreActive_(false),
      startNs_(tree.now()),
      cpuStartNs_(tree.cpuNow()) {
  node_->incrementCallCount();
}

ScopedTimer::~ScopedTimer() {
  const uint64_t endNs = tree_.now();
  const uint64_t cpuEndNs = tree_.cpuNow();
  VELOX_CHECK_GE(endNs, startNs_);
  node_->addTime(endNs - startNs_);
  if (cpuEndNs >= cpuStartNs_) {
    node_->addCpuTime(cpuEndNs - cpuStartNs_);
  }
  if (restoreActive_) {
    activeNode_ = previousActive_;
  }
}

} // namespace facebook::velox
