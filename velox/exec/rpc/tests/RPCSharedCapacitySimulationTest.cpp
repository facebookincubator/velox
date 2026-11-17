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

// Simulates workers, each with its own RPCRateLimiter, that keep their windows
// full against one backend on a virtual clock. The backend holds 'capacity'
// calls at once and has no queue: a call beyond it is rejected after 1 ms, the
// rest succeed after the latency, jittered by +-20%. Workers resend as soon as
// their windows have room, so workers already holding the backend take most
// freed slots. Random draws use only the generator's raw output and
// simultaneous completions run in the order they were scheduled, so each run
// sees the same random numbers with any standard library.

#include <fmt/format.h>
#include <folly/futures/ManualTimekeeper.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "velox/exec/rpc/RPCRateLimiter.h"

namespace facebook::velox::exec::rpc {
namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr auto kLatency = 1'000ms;

// A limiter whose window is below one call admits on a timer this simulation
// does not run, so when nothing is in flight the workers ask again after this
// long.
constexpr auto kPacingPoll = 100ms;

// Rejections before this are startup overshoot and are not counted.
constexpr auto kWarmup = 60s;

class SharedBackend {
 public:
  SharedBackend(
      int numWorkers,
      int64_t capacity,
      RPCFlowControlPolicy policy,
      uint32_t seed)
      : capacity_(capacity), random_(seed) {
    for (int i = 0; i < numWorkers; ++i) {
      auto limiter = std::make_shared<RPCRateLimiter>(
          "shared." + std::to_string(i),
          [this] { return now_; },
          &timekeeper_,
          [] { return 1.0; });
      limiter->configure({
          .adaptive = true,
          .ceiling = 1'000,
          .floor = 1,
          .flowControlPolicy = policy,
      });
      limiters_.push_back(std::move(limiter));
    }
    started_.resize(numWorkers, false);
    successes_.resize(numWorkers, 0);
  }

  // Lets 'worker' send from now on.
  void start(int worker) {
    started_[worker] = true;
    refill(worker);
  }

  // Runs the backend until 'at' after it was created.
  void runUntil(Clock::duration at) {
    const auto end = Clock::time_point{} + at;
    refillAll();
    for (;;) {
      if (calls_.empty()) {
        if (now_ + kPacingPoll > end) {
          break;
        }
        now_ += kPacingPoll;
        refillAll();
        continue;
      }
      if (calls_.top().done > end) {
        break;
      }
      const auto call = calls_.top();
      calls_.pop();
      now_ = std::max(now_, call.done);
      complete(call);
      refillAll();
    }
    now_ = std::max(now_, end);
  }

  int64_t window(int worker) const {
    return limiters_[worker]->stats().capacity;
  }

  // Successes after the warmup.
  int64_t successes(int worker) const {
    return successes_[worker];
  }

  int64_t totalSuccesses() const {
    int64_t total{0};
    for (const auto count : successes_) {
      total += count;
    }
    return total;
  }

  // Rejections after the warmup.
  int64_t rejections() const {
    return numRejections_;
  }

  // Jain's index of the windows, each capped at the capacity: 1 when all are
  // equal, 1/N when one worker holds them all.
  double jainIndex() const {
    double sum{0};
    double squares{0};
    for (const auto& limiter : limiters_) {
      const double window = std::min<double>(
          static_cast<double>(limiter->stats().capacity),
          static_cast<double>(capacity_));
      sum += window;
      squares += window * window;
    }
    return squares == 0
        ? 0
        : sum * sum / (static_cast<double>(limiters_.size()) * squares);
  }

 private:
  struct Call {
    Clock::time_point done;
    // Orders calls done at the same time by when they were sent.
    uint64_t sequence;
    int worker;
    std::shared_ptr<RPCRateLimiter::AdmissionLease> lease;
    bool rejected;

    bool operator>(const Call& other) const {
      return std::tie(done, sequence) > std::tie(other.done, other.sequence);
    }
  };

  // Rotates the first worker to refill so no worker always sends first.
  void refillAll() {
    const int numWorkers = static_cast<int>(limiters_.size());
    for (int i = 0; i < numWorkers; ++i) {
      refill((nextWorker_ + i) % numWorkers);
    }
    nextWorker_ = (nextWorker_ + 1) % numWorkers;
  }

  void refill(int worker) {
    if (!started_[worker]) {
      return;
    }
    auto& limiter = *limiters_[worker];
    for (;;) {
      auto tokens = limiter.tryAcquireUpTo(1);
      if (tokens.empty()) {
        return;
      }
      const bool rejected = served_ >= capacity_;
      Clock::duration delay = 1ms;
      if (!rejected) {
        ++served_;
        const double unitDraw = static_cast<double>(random_()) /
            (static_cast<double>(std::mt19937::max()) + 1.0);
        delay = std::chrono::duration_cast<Clock::duration>(
            kLatency * (0.8 + 0.4 * unitDraw));
      }
      calls_.push({
          now_ + delay,
          numSent_++,
          worker,
          limiter.makeLease(std::move(tokens)),
          rejected,
      });
    }
  }

  void complete(const Call& call) {
    const bool counted = now_ - Clock::time_point{} >= kWarmup;
    if (call.rejected) {
      numRejections_ += counted ? 1 : 0;
      call.lease->completeAndRelease(
          /*hardOverload=*/true, /*capacityRejection=*/true);
      return;
    }
    --served_;
    successes_[call.worker] += counted ? 1 : 0;
    const auto completion = call.lease->completeAndRelease(
        /*hardOverload=*/false, /*capacityRejection=*/false);
    limiters_[call.worker]->onSuccessForEpoch(
        completion.overloadEpoch,
        1,
        std::chrono::duration_cast<std::chrono::nanoseconds>(kLatency).count());
  }

  const int64_t capacity_;
  std::mt19937 random_;
  folly::ManualTimekeeper timekeeper_;
  Clock::time_point now_{};
  std::vector<std::shared_ptr<RPCRateLimiter>> limiters_;
  std::vector<bool> started_;
  std::vector<int64_t> successes_;
  std::priority_queue<Call, std::vector<Call>, std::greater<Call>> calls_;
  int nextWorker_{0};
  // Calls the backend holds now.
  int64_t served_{0};
  int64_t numRejections_{0};
  uint64_t numSent_{0};
};

struct SteadyState {
  int64_t successes{0};
  int64_t rejections{0};
  // Mean of Jain's index of the windows, sampled every 10 s after 300 s.
  double meanJainIndex{0};
  // The fewest successes of any worker over its fair share, the mean.
  double minShare{0};
};

constexpr auto kDuration = 1'200s;

SteadyState
steadyState(int numWorkers, int64_t capacity, RPCFlowControlPolicy policy) {
  SharedBackend backend(numWorkers, capacity, policy, /*seed=*/1);
  for (int i = 0; i < numWorkers; ++i) {
    backend.start(i);
  }
  double jainSum{0};
  int numSamples{0};
  for (auto at = 10s; at <= kDuration; at += 10s) {
    backend.runUntil(at);
    if (at >= 300s) {
      jainSum += backend.jainIndex();
      ++numSamples;
    }
  }
  SteadyState result{
      .successes = backend.totalSuccesses(),
      .rejections = backend.rejections(),
      .meanJainIndex = jainSum / numSamples,
      .minShare = 1,
  };
  const double fairShare = static_cast<double>(result.successes) / numWorkers;
  for (int i = 0; i < numWorkers; ++i) {
    result.minShare = std::min(
        result.minShare, static_cast<double>(backend.successes(i)) / fairShare);
  }
  return result;
}

// Compares both policies when 1-10 workers keep 60-600 slots busy for 20
// minutes: successes and rejections after the first minute, the mean Jain's
// index of the windows, and the smallest share of successes any worker got.
// CAPACITY_PROBING serves at least as many calls and gives every worker at
// least half its fair share. A round of growth exceeds the backend by at most
// 16 calls and a worker alone exceeds it about once every eight rounds, so its
// rejections stay below 2 per worker per round.
TEST(RPCSharedCapacitySimulationTest, steadyState) {
  fmt::print(
      "{:>5} {:>3} | {:>9} {:>8} {:>6} {:>5} | {:>9} {:>8} {:>6} {:>5}\n",
      "slots",
      "N",
      "W_success",
      "W_reject",
      "W_jain",
      "W_min",
      "P_success",
      "P_reject",
      "P_jain",
      "P_min");
  for (const int64_t capacity : {60, 240, 600}) {
    for (const int numWorkers : {1, 2, 4, 10}) {
      const auto window =
          steadyState(numWorkers, capacity, RPCFlowControlPolicy::kWindowAimd);
      const auto probing = steadyState(
          numWorkers, capacity, RPCFlowControlPolicy::kCapacityProbing);
      fmt::print(
          "{:>5} {:>3} | {:>9} {:>8} {:>6.3f} {:>5.2f} | {:>9} {:>8} {:>6.3f} {:>5.2f}\n",
          capacity,
          numWorkers,
          window.successes,
          window.rejections,
          window.meanJainIndex,
          window.minShare,
          probing.successes,
          probing.rejections,
          probing.meanJainIndex,
          probing.minShare);
      std::fflush(stdout);
      const auto cell =
          fmt::format("{} slots, {} workers", capacity, numWorkers);
      EXPECT_GE(probing.successes, window.successes) << cell;
      EXPECT_GE(probing.minShare, 0.5) << cell;
      EXPECT_LE(
          static_cast<double>(probing.rejections),
          2.0 * numWorkers * ((kDuration - kWarmup) / kLatency))
          << cell;
    }
  }
}

// Three workers fill 240 slots; a fourth arrives after 5 minutes and the run
// continues for 15 more. The incumbents take most freed slots, so the late
// worker's window falls below half its fair share before it grows back into
// room their cuts free; the test times that regrowth. Under CAPACITY_PROBING
// growth heads for the calls the backend held at its last cut, rather than
// adding one call per round.
TEST(RPCSharedCapacitySimulationTest, lateWorker) {
  constexpr int kNumWorkers = 4;
  constexpr int kLateWorker = kNumWorkers - 1;
  constexpr int64_t kCapacity = 240;
  constexpr int64_t kFairShare = kCapacity / kNumWorkers;
  constexpr auto kArrival = 300s;
  fmt::print(
      "{:<16} | {:>12} {:>14} {:>14}\n",
      "policy",
      "below half, s",
      "back to half, s",
      "share, last 5m");
  for (const auto policy : {
           RPCFlowControlPolicy::kWindowAimd,
           RPCFlowControlPolicy::kCapacityProbing,
       }) {
    SharedBackend backend(kNumWorkers, kCapacity, policy, /*seed=*/1);
    for (int i = 0; i < kLateWorker; ++i) {
      backend.start(i);
    }
    backend.runUntil(kArrival);
    backend.start(kLateWorker);
    // When the late worker's window first fell below half its fair share, and
    // how long it then took to get back to it.
    std::optional<std::chrono::seconds> belowHalfAfter;
    std::optional<std::chrono::seconds> backToHalfAfter;
    int64_t lateBefore{0};
    int64_t totalBefore{0};
    for (auto at = kArrival + 1s; at <= kArrival + 900s; at += 1s) {
      backend.runUntil(at);
      const auto window = backend.window(kLateWorker);
      if (!belowHalfAfter.has_value() && window < kFairShare / 2) {
        belowHalfAfter = at - kArrival;
      }
      if (belowHalfAfter.has_value() && !backToHalfAfter.has_value() &&
          window >= kFairShare / 2) {
        backToHalfAfter = at - kArrival - *belowHalfAfter;
      }
      if (at == kArrival + 600s) {
        lateBefore = backend.successes(kLateWorker);
        totalBefore = backend.totalSuccesses();
      }
    }
    const auto seconds = [](const std::optional<std::chrono::seconds>& after) {
      return after.has_value() ? std::to_string(after->count())
                               : std::string("never");
    };
    const double lastShare =
        static_cast<double>(backend.successes(kLateWorker) - lateBefore) /
        static_cast<double>(backend.totalSuccesses() - totalBefore) *
        kNumWorkers;
    fmt::print(
        "{:<16} | {:>12} {:>14} {:>14.2f}\n",
        rpcFlowControlPolicyName(policy),
        seconds(belowHalfAfter),
        seconds(backToHalfAfter),
        lastShare);
    if (policy == RPCFlowControlPolicy::kCapacityProbing) {
      ASSERT_TRUE(backToHalfAfter.has_value());
      EXPECT_LE(*backToHalfAfter, 60s);
      EXPECT_GE(lastShare, 0.5);
    }
  }
}

} // namespace
} // namespace facebook::velox::exec::rpc
