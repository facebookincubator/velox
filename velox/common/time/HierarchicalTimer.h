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

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox {

/// A node in the hierarchical timer tree.
///
/// Each node tracks cumulative wall time, CPU time, and call count for a named
/// section. Nodes form a tree: "parent/child" creates a "child" node under
/// "parent". Tracks wall total, CPU total, count, min, and max time per node.
///
/// Not thread-safe. All access must be from a single thread.
class TimerNode {
 public:
  explicit TimerNode(const std::string& name, TimerNode* parent = nullptr);

  /// Returns or creates a direct child with the given name.
  TimerNode* getOrCreateChild(const std::string& childName);

  /// Adds elapsed nanoseconds to this node's wall time total, updates min/max.
  void addTime(uint64_t ns);

  /// Adds elapsed nanoseconds to this node's CPU time total, updates min/max.
  void addCpuTime(uint64_t ns);

  /// Increments the call counter by one.
  void incrementCallCount();

  /// Zeros all counters for this node and all descendants.
  /// Keeps the tree structure intact.
  void reset();

  uint64_t totalTimeNs() const;
  uint64_t totalCpuNs() const;
  uint64_t callCount() const;
  uint64_t averageTimeNs() const;
  uint64_t minTimeNs() const;
  uint64_t maxTimeNs() const;
  uint64_t averageCpuNs() const;
  uint64_t minCpuNs() const;
  uint64_t maxCpuNs() const;
  const std::string& name() const;
  TimerNode* parent() const;
  const std::vector<std::unique_ptr<TimerNode>>& children() const;

  /// Formats this node (and descendants) into a human-readable table row.
  /// `depth` controls indentation; `isLast` controls tree-drawing glyphs.
  void format(
      std::string& out,
      int depth,
      bool isLast,
      uint64_t parentTimeNs,
      uint64_t parentCpuTimeNs,
      bool verbose) const;

 private:
  /// Internal recursive formatter that tracks ancestor isLast flags for
  /// drawing proper │ continuation lines in the tree.
  void formatImpl(
      std::string& out,
      int depth,
      bool isLast,
      uint64_t parentTimeNs,
      uint64_t parentCpuTimeNs,
      bool verbose,
      std::vector<bool>& ancestorTrail) const;
  std::string name_;
  TimerNode* parent_;
  std::vector<std::unique_ptr<TimerNode>> children_;
  std::unordered_map<std::string, TimerNode*> childrenByName_;
  uint64_t totalTimeNs_{0};
  uint64_t totalCpuNs_{0};
  uint64_t callCount_{0};
  uint64_t minTimeNs_{std::numeric_limits<uint64_t>::max()};
  uint64_t maxTimeNs_{0};
  uint64_t minCpuNs_{std::numeric_limits<uint64_t>::max()};
  uint64_t maxCpuNs_{0};
};

/// Owns a forest of TimerNode trees and provides path-based access.
///
/// Paths are `/`-separated: "a/b/c" resolves to root -> a -> b -> c.
/// Multiple top-level names create separate subtrees.
///
/// Uses RDTSCP for high-precision timing when `ENABLE_HW_TIMER` is defined,
/// automatically falls back to `steady_clock` otherwise.
///
/// Uses ScopedTimer for RAII timing. On destruction (or when printStats() is
/// called), prints a hierarchical timing breakdown and clears all data.
///
/// Not thread-safe. All access must be from a single thread.
///
/// ## Usage
///
/// ### Thread-local (recommended) -- auto-nesting via call stack:
///
///   void Reader::loadStripe(int i) {
///     ScopedTimer t("loadStripe");
///     {
///       ScopedTimer t2("readIO");     // auto-nested under loadStripe
///       readIO(i);
///     }
///     {
///       ScopedTimer t3("decode");     // auto-nested under loadStripe
///       decode(i);
///     }
///   }
///
///   void Writer::flush() {
///     ScopedTimer t("flush");
///     doFlush();
///   }
///
///   // Print results at any point:
///   TimerTree::threadInstance().printStats();
///
/// ### Explicit tree -- for isolated profiling sessions:
///
///   TimerTree tree("my benchmark");
///   for (int i = 0; i < numStripes; ++i) {
///     ScopedTimer t(tree, "loadStripe");
///     loadStripe(i);
///   }
///   // Destructor prints results, or call tree.printStats() explicitly.
///
/// ## Example output
///
///   =============================================================...=========
///                        HIERARCHICAL TIMING BREAKDOWN
///   =============================================================...=========
///
///   Section                                                      Wall
///      CPU    Calls  Avg Wall  ...  % Parent
///   -------------------------------------------------------------...---------
///   loadStripe                                               120.50ms
///    45.20ms        5   24.10ms  ...        -
///   ├── readIO                                                80.20ms
///    10.50ms        5   16.04ms  ...    66.6%
///   ├── decode                                                35.10ms
///    30.00ms        5    7.02ms  ...    29.1%
///   └── (other)                                                5.20ms
///     4.70ms        -        -  ...     4.3%
///   flush                                                     15.30ms
///    14.80ms        1   15.30ms  ...        -
///   =============================================================...=========
class TimerTree {
 public:
  explicit TimerTree(const std::string& name = "");

  TimerTree(const TimerTree&) = delete;
  TimerTree& operator=(const TimerTree&) = delete;
  TimerTree(TimerTree&&) = delete;
  TimerTree& operator=(TimerTree&&) = delete;

  /// Prints the results table if any entries were recorded.
  ~TimerTree();

  /// Returns the current wall timestamp in nanoseconds.
  /// Uses RDTSCP when ENABLE_HW_TIMER is defined, steady_clock otherwise.
  uint64_t now() const;

  /// Returns the current thread CPU time in nanoseconds.
  /// Uses clock_gettime(CLOCK_THREAD_CPUTIME_ID) to measure user+system CPU
  /// time for the calling thread.
  uint64_t cpuNow() const;

  /// Walks (or creates) the node addressed by a `/`-separated path.
  TimerNode* getOrCreateNode(const std::string& path);

  /// Zeros all counters but keeps the tree structure.
  void reset();

  /// Destroys the entire tree (removes all nodes).
  void clear();

  /// Returns a formatted hierarchical summary table.
  std::string toString(bool verbose = false) const;

  /// Prints the results table via LOG(INFO) and clears all data.
  void printStats(bool verbose = false);

  /// Access root node for inspection.
  const TimerNode& root() const;

  /// Returns the timer name.
  const std::string& name() const;

  /// Returns the thread-local TimerTree instance. Creates one on first access
  /// per thread. Results are printed when the thread exits (thread_local
  /// destructor). Call printStats() to print and reset intermediate results.
  static TimerTree& threadInstance();

 private:
  friend class ScopedTimer;

  std::string name_;
  std::unique_ptr<TimerNode> root_;
};

/// RAII timer that measures a scoped block and records it into a TimerTree.
///
/// On construction, records the current wall and CPU time and increments the
/// call count for the node. On destruction, computes elapsed wall and CPU
/// nanoseconds and adds them to the node.
///
/// Two constructors:
/// - ScopedTimer(name): records into TimerTree::threadInstance() with
///   stack-based auto-nesting. Nested ScopedTimers automatically form
///   parent-child relationships based on the call stack. The name is treated
///   as a simple child name (no path splitting).
/// - ScopedTimer(tree, path): records into an explicit TimerTree using
///   path-based node resolution (splits on '/').
///
/// Not thread-safe. The referenced TimerTree must outlive this object.
class ScopedTimer {
 public:
  /// Records into the thread-local TimerTree with stack-based auto-nesting.
  /// The name is used as a direct child name under the currently active node
  /// (or root if no timer is active).
  explicit ScopedTimer(const std::string& name);

  /// Records into an explicit TimerTree using path-based node resolution.
  ScopedTimer(TimerTree& tree, const std::string& path);
  ~ScopedTimer();

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
  ScopedTimer(ScopedTimer&&) = delete;
  ScopedTimer& operator=(ScopedTimer&&) = delete;

 private:
  TimerTree& tree_;
  TimerNode* node_;
  TimerNode* previousActive_{nullptr};
  bool restoreActive_;
  uint64_t startNs_;
  uint64_t cpuStartNs_;
};

} // namespace facebook::velox
