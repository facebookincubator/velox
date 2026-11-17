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

// Simulates jobs of finite rows on a virtual clock through both admission
// layers RPCOperator uses: each driver's CongestionController window and each
// worker's RPCRateLimiter. One backend serves 'numSlots' calls at once, queues
// up to 'queueCapacity' more and rejects the rest. A rejected row retries after
// a jittered exponential backoff and fails after 'maxRetries' retries. Each
// test prints WINDOW_AIMD and CAPACITY_PROBING side by side and checks what
// CAPACITY_PROBING claims for those jobs. Random draws use only the generator's
// raw output and simultaneous events run in the order they were scheduled, so
// each job sees the same random numbers with any standard library.

#include <fmt/format.h>
#include <folly/futures/ManualTimekeeper.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "velox/exec/rpc/CongestionController.h"
#include "velox/exec/rpc/RPCRateLimiter.h"

namespace facebook::velox::exec::rpc {
namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

// A limiter whose window is below one call admits on a timer this simulation
// does not run, so a worker it refuses with nothing in flight asks again after
// this long.
constexpr auto kPacingPoll = 100ms;

// A job still running after this long has stalled.
constexpr auto kMaxSimulatedTime = 2h;

// The PER_ROW driver window RPCOperator starts at.
constexpr int64_t kDefaultDriverWindow{100};

struct Job {
  std::string name;
  int numWorkers;
  int numDrivers;
  // Drivers that receive rows: a narrow job feeds one.
  int numActiveDrivers;
  int64_t rowsPerWorker;
  int64_t numSlots;
  int64_t queueCapacity;
  std::chrono::milliseconds latency;
  int maxRetries;
  // The first retry's nominal wait; later retries double it, up to a minute.
  std::chrono::milliseconds retryAfter{100};
  // Per-worker limiter ceiling the function or session sets.
  int64_t ceiling;
  // Only the safety ceiling of rpc.ratelimiter.hard_limit=0 bounds the limiter,
  // so a CAPACITY_PROBING driver window rises only to 200.
  bool safetyCeilingOnly{false};
  // Rows come from one job-wide pool, as when a source stage hands splits to
  // whichever worker asks, instead of a fixed share per worker.
  bool pooled{false};
  // The last 'numLateWorkers' workers start 'lateStart' after the others, as a
  // second query arriving on a busy backend.
  int numLateWorkers{0};
  std::chrono::seconds lateStart{0};
  // Rows of each late worker, if not 'rowsPerWorker': a smaller query
  // arriving at a busy backend.
  int64_t lateRowsPerWorker{0};
  // The first 'numDefaultWorkers' workers run WINDOW_AIMD whatever the policy,
  // as default queries sharing the backend with opted-in ones.
  int numDefaultWorkers{0};
  // Workers start spread evenly over this interval.
  std::chrono::seconds startSpread{0};
  // The backend rejects every call during this window.
  std::chrono::seconds outageStart{0};
  std::chrono::seconds outageLength{0};
};

struct Policy {
  std::string_view name;
  RPCFlowControlPolicy flowControl;
  // Zero starts each driver window at 100 and raises it to the limiter's
  // ceiling under CAPACITY_PROBING, as RPCOperator does.
  int64_t driverWindow{0};
};

constexpr Policy kWindowAimd{"WINDOW_AIMD", RPCFlowControlPolicy::kWindowAimd};
constexpr Policy kCapacityProbing{
    "CAPACITY_PROBING",
    RPCFlowControlPolicy::kCapacityProbing,
};

// Means over seeds.
struct Result {
  double seconds{0};
  // Busy slot-seconds over slot-seconds available until the job finished.
  double utilization{0};
  double rejections{0};
  // Rows that used up their retries.
  double failedRows{0};
  // Seconds from the late workers' start until the last of them finished.
  double lateDone{0};
  // Seconds from the end of an outage until the backend is 90% busy again
  // over 5 s; infinite if it never was.
  double recoverySeconds{0};
  // Most calls the backend held in its queue at once.
  double peakQueued{0};
  // When the last of the default workers finished.
  double defaultDone{0};
};

class JobSimulation {
 public:
  JobSimulation(const Job& job, const Policy& policy, uint32_t seed)
      : job_(job), random_(seed) {
    for (int i = 0; i < job.numWorkers; ++i) {
      auto worker = std::make_unique<Worker>();
      worker->limiter = std::make_shared<RPCRateLimiter>(
          "job." + std::to_string(i),
          [this] { return now_; },
          &timekeeper_,
          [] { return 1.0; });
      const bool isDefault = i < job.numDefaultWorkers;
      worker->flowControl =
          isDefault ? RPCFlowControlPolicy::kWindowAimd : policy.flowControl;
      worker->limiter->configure({
          .adaptive = true,
          .ceiling = job.ceiling,
          .floor = 50,
          .flowControlPolicy = worker->flowControl,
      });
      int64_t driverWindow{kDefaultDriverWindow};
      if (!isDefault && policy.driverWindow > 0) {
        driverWindow = policy.driverWindow;
      } else if (
          worker->flowControl == RPCFlowControlPolicy::kCapacityProbing) {
        driverWindow = std::max(
            kDefaultDriverWindow,
            job.safetyCeilingOnly ? RPCRateLimiter::kInitialAdaptiveLimit
                                  : job.ceiling);
      }
      const bool isLate = i >= job.numWorkers - job.numLateWorkers;
      const int64_t rows = isLate && job.lateRowsPerWorker > 0
          ? job.lateRowsPerWorker
          : job.rowsPerWorker;
      for (int driverIndex = 0; driverIndex < job.numDrivers; ++driverIndex) {
        Driver driver;
        driver.window = CongestionController{driverWindow, driverWindow};
        if (!job.pooled && driverIndex < job.numActiveDrivers) {
          driver.freshRows = rows / job.numActiveDrivers +
              (driverIndex < rows % job.numActiveDrivers ? 1 : 0);
        }
        worker->drivers.push_back(std::move(driver));
      }
      worker->remainingRows = rows;
      totalRows_ += rows;
      worker->startAt = Clock::time_point{} + (isLate ? job.lateStart : 0s) +
          std::chrono::duration_cast<Clock::duration>(
                            job.startSpread * i / job.numWorkers);
      workers_.push_back(std::move(worker));
    }
    poolRows_ = job.pooled ? totalRows_ : 0;
  }

  Result run() {
    for (int i = 0; i < job_.numWorkers; ++i) {
      push(workers_[i]->startAt, EventKind::kWake, i);
    }
    while (!events_.empty() && doneRows_ < totalRows_ &&
           now_ < Clock::time_point{} + kMaxSimulatedTime) {
      const auto event = events_.top();
      events_.pop();
      advanceTo(event.at);
      switch (event.kind) {
        case EventKind::kArrive:
          arrive(event.id);
          break;
        case EventKind::kServiceDone:
          serviceDone(event.id);
          break;
        case EventKind::kRespond:
          respond(event.id);
          refill(calls_[event.id].worker);
          break;
        case EventKind::kWake: {
          auto& wakeAt = workers_[event.id]->wakeAt;
          if (wakeAt.has_value() && *wakeAt <= now_) {
            wakeAt.reset();
          }
          refill(static_cast<int>(event.id));
          break;
        }
      }
    }
    EXPECT_EQ(doneRows_, totalRows_) << job_.name << ": the simulation stalled";
    const double elapsed = seconds(now_);
    Result result{
        .seconds = elapsed,
        .utilization =
            busyIntegral_ / (elapsed * static_cast<double>(job_.numSlots)),
        .rejections = static_cast<double>(numRejections_),
        .failedRows = static_cast<double>(numFailedRows_),
        .recoverySeconds = recoverySeconds(),
        .peakQueued = static_cast<double>(peakQueued_),
    };
    for (int i = job_.numWorkers - job_.numLateWorkers; i < job_.numWorkers;
         ++i) {
      result.lateDone = std::max(
          result.lateDone,
          workers_[i]->doneAt -
              std::chrono::duration<double>(job_.lateStart).count());
    }
    for (int i = 0; i < job_.numDefaultWorkers; ++i) {
      result.defaultDone = std::max(result.defaultDone, workers_[i]->doneAt);
    }
    return result;
  }

 private:
  enum class EventKind { kArrive, kServiceDone, kRespond, kWake };

  struct Event {
    Clock::time_point at;
    // Orders events due at the same time by when they were scheduled.
    uint64_t sequence;
    EventKind kind;
    // A call index, or a worker index for kWake.
    size_t id;

    bool operator>(const Event& other) const {
      return std::tie(at, sequence) > std::tie(other.at, other.sequence);
    }
  };

  struct Call {
    int worker;
    int driver;
    int retries;
    Clock::time_point sentAt;
    std::shared_ptr<RPCRateLimiter::AdmissionLease> lease;
    bool rejected{false};
  };

  struct RetryRow {
    Clock::time_point notBefore;
    // Orders rows due at the same time by when they were rejected.
    uint64_t sequence;
    int retries;

    bool operator>(const RetryRow& other) const {
      return std::tie(notBefore, sequence) >
          std::tie(other.notBefore, other.sequence);
    }
  };

  struct Driver {
    CongestionController window;
    int64_t inFlight{0};
    int64_t freshRows{0};
    // Set by a rejection that is retried; the next dispatch applies the driver
    // backoff once, as RPCOperator::dispatchDueRetries() does.
    bool overloadPending{false};
    // Set by a row out of retries; the next dispatch halves the driver window
    // once under either policy, as RPCOperator does when it outputs such rows.
    bool exhaustedPending{false};
    std::priority_queue<RetryRow, std::vector<RetryRow>, std::greater<RetryRow>>
        retries;

    bool hasReadyRetry(Clock::time_point now) const {
      return !retries.empty() && retries.top().notBefore <= now;
    }
  };

  struct Worker {
    // The worker's policy: the simulated one, or WINDOW_AIMD for a default
    // worker.
    RPCFlowControlPolicy flowControl{RPCFlowControlPolicy::kWindowAimd};
    Clock::time_point startAt;
    std::shared_ptr<RPCRateLimiter> limiter;
    std::vector<Driver> drivers;
    // Rotates the first driver to refill.
    int nextDriver{0};
    int64_t remainingRows{0};
    double doneAt{0};
    // The earliest pending wake, so each retry time is scheduled once.
    std::optional<Clock::time_point> wakeAt;
  };

  static double seconds(Clock::time_point at) {
    return std::chrono::duration<double>(at - Clock::time_point{}).count();
  }

  void advanceTo(Clock::time_point at) {
    if (at <= now_) {
      return;
    }
    busyIntegral_ += static_cast<double>(busy_) *
        std::chrono::duration<double>(at - now_).count();
    double from = seconds(now_);
    const double to = seconds(at);
    while (from < to) {
      const auto second = static_cast<size_t>(from);
      const double end = std::min(to, static_cast<double>(second + 1));
      if (busyBySecond_.size() <= second) {
        busyBySecond_.resize(second + 1, 0.0);
      }
      busyBySecond_[second] += static_cast<double>(busy_) * (end - from);
      from = end;
    }
    now_ = at;
  }

  double recoverySeconds() const {
    if (job_.outageLength.count() == 0) {
      return 0;
    }
    const auto outageEnd =
        static_cast<size_t>((job_.outageStart + job_.outageLength).count());
    for (size_t second = outageEnd; second + 5 <= busyBySecond_.size();
         ++second) {
      double busy{0};
      for (size_t i = second; i < second + 5; ++i) {
        busy += busyBySecond_[i];
      }
      if (busy >= 0.9 * 5 * static_cast<double>(job_.numSlots)) {
        return static_cast<double>(second - outageEnd);
      }
    }
    return std::numeric_limits<double>::infinity();
  }

  bool inOutage() const {
    const auto elapsed = now_ - Clock::time_point{};
    return elapsed >= job_.outageStart &&
        elapsed < job_.outageStart + job_.outageLength;
  }

  void push(Clock::time_point at, EventKind kind, size_t id) {
    events_.push({at, numScheduled_++, kind, id});
  }

  // Uniform in [0, 1). The standard distributions may draw differently in
  // each standard library; the generator's output is the same everywhere.
  double unitDraw() {
    return static_cast<double>(random_()) /
        (static_cast<double>(std::mt19937::max()) + 1.0);
  }

  Clock::duration serviceTime() {
    return std::chrono::duration_cast<Clock::duration>(
        job_.latency * (0.8 + 0.4 * unitDraw()));
  }

  // One way, 1-3 ms.
  Clock::duration networkDelay() {
    return std::chrono::microseconds{1'000 + random_() % 2'001};
  }

  void applyDriverBackoff(Worker& worker, Driver& driver) {
    if (driver.exhaustedPending) {
      driver.exhaustedPending = false;
      driver.window.onError();
    }
    if (!driver.overloadPending) {
      return;
    }
    driver.overloadPending = false;
    // RPCOperator leaves the backoff of a retried capacity rejection to a
    // CAPACITY_PROBING limiter.
    if (worker.flowControl != RPCFlowControlPolicy::kCapacityProbing) {
      driver.window.onError();
    }
  }

  int64_t freshRows(const Driver& driver) const {
    return job_.pooled ? poolRows_ : driver.freshRows;
  }

  void refill(int workerIndex) {
    auto& worker = *workers_[workerIndex];
    if (now_ < worker.startAt) {
      return;
    }
    const int numDrivers = static_cast<int>(worker.drivers.size());
    bool limiterFull{false};
    for (int i = 0; i < numDrivers && !limiterFull; ++i) {
      const int driverIndex = (worker.nextDriver + i) % numDrivers;
      auto& driver = worker.drivers[driverIndex];
      applyDriverBackoff(worker, driver);
      while (driver.inFlight < driver.window.limit() &&
             (driver.hasReadyRetry(now_) || freshRows(driver) > 0)) {
        auto tokens = worker.limiter->tryAcquireUpTo(1);
        if (tokens.empty()) {
          limiterFull = true;
          break;
        }
        int retries{0};
        if (driver.hasReadyRetry(now_)) {
          retries = driver.retries.top().retries;
          driver.retries.pop();
        } else if (job_.pooled) {
          --poolRows_;
        } else {
          --driver.freshRows;
        }
        ++driver.inFlight;
        const size_t id = calls_.size();
        calls_.push_back({
            workerIndex,
            driverIndex,
            retries,
            now_,
            worker.limiter->makeLease(std::move(tokens)),
        });
        push(now_ + networkDelay(), EventKind::kArrive, id);
      }
    }
    worker.nextDriver = (worker.nextDriver + 1) % numDrivers;
    int64_t inFlight{0};
    for (const auto& driver : worker.drivers) {
      inFlight += driver.inFlight;
    }
    scheduleWake(
        workerIndex,
        limiterFull && inFlight == 0
            ? std::optional<Clock::time_point>(now_ + kPacingPoll)
            : std::nullopt);
  }

  // Wakes the worker at 'earliest', if set, or when its earliest waiting retry
  // is due, whichever comes first.
  void scheduleWake(
      int workerIndex,
      std::optional<Clock::time_point> earliest) {
    auto& worker = *workers_[workerIndex];
    for (const auto& driver : worker.drivers) {
      if (!driver.retries.empty() && driver.retries.top().notBefore > now_ &&
          (!earliest.has_value() ||
           driver.retries.top().notBefore < *earliest)) {
        earliest = driver.retries.top().notBefore;
      }
    }
    if (earliest.has_value() &&
        (!worker.wakeAt.has_value() || *earliest < *worker.wakeAt)) {
      worker.wakeAt = earliest;
      push(*earliest, EventKind::kWake, workerIndex);
    }
  }

  void arrive(size_t id) {
    if (!inOutage() && busy_ < job_.numSlots) {
      ++busy_;
      push(now_ + serviceTime(), EventKind::kServiceDone, id);
    } else if (
        !inOutage() &&
        static_cast<int64_t>(queued_.size()) < job_.queueCapacity) {
      queued_.push_back(id);
      peakQueued_ = std::max(peakQueued_, static_cast<int64_t>(queued_.size()));
    } else {
      calls_[id].rejected = true;
      push(now_ + 1ms + networkDelay(), EventKind::kRespond, id);
    }
  }

  void serviceDone(size_t id) {
    --busy_;
    if (!queued_.empty()) {
      const auto next = queued_.front();
      queued_.pop_front();
      ++busy_;
      push(now_ + serviceTime(), EventKind::kServiceDone, next);
    }
    push(now_ + networkDelay(), EventKind::kRespond, id);
  }

  void respond(size_t id) {
    auto& call = calls_[id];
    auto& worker = *workers_[call.worker];
    auto& driver = worker.drivers[call.driver];
    --driver.inFlight;
    if (call.rejected) {
      ++numRejections_;
      call.lease->completeAndRelease(
          /*hardOverload=*/true, /*capacityRejection=*/true);
      call.lease.reset();
      const int retries = call.retries + 1;
      if (retries > job_.maxRetries) {
        driver.exhaustedPending = true;
        ++numFailedRows_;
        rowDone(worker);
        return;
      }
      driver.overloadPending = true;
      // Exponential from the first retry's wait, capped at a minute, fully
      // jittered.
      const double cap = std::min(
          std::chrono::duration<double>(job_.retryAfter).count() *
              std::pow(2.0, retries - 1),
          60.0);
      driver.retries.push({
          now_ +
              std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>(cap * unitDraw())),
          numScheduled_++,
          retries,
      });
      return;
    }
    const auto responseNanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now_ - call.sentAt)
            .count();
    driver.window.onSample(responseNanos);
    const auto completion = call.lease->completeAndRelease(
        /*hardOverload=*/false, /*capacityRejection=*/false);
    call.lease.reset();
    worker.limiter->onSuccessForEpoch(
        completion.overloadEpoch, 1, responseNanos);
    rowDone(worker);
  }

  void rowDone(Worker& worker) {
    if (!job_.pooled && --worker.remainingRows == 0) {
      worker.doneAt = seconds(now_);
    }
    ++doneRows_;
  }

  const Job job_;
  std::mt19937 random_;
  folly::ManualTimekeeper timekeeper_;
  Clock::time_point now_{};
  std::vector<std::unique_ptr<Worker>> workers_;
  std::vector<Call> calls_;
  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events_;
  std::deque<size_t> queued_;
  int64_t busy_{0};
  double busyIntegral_{0};
  std::vector<double> busyBySecond_;
  int64_t totalRows_{0};
  int64_t poolRows_{0};
  int64_t doneRows_{0};
  int64_t numRejections_{0};
  int64_t numFailedRows_{0};
  int64_t peakQueued_{0};
  uint64_t numScheduled_{0};
};

void printHeader() {
  fmt::print(
      "{:<36} {:<16} | {:>8} {:>6} {:>10} {:>8} | {:>7} {:>7} {:>7} {:>7}\n",
      "job",
      "policy",
      "seconds",
      "busy%",
      "rejections",
      "failed",
      "late",
      "recover",
      "queued",
      "default");
}

// Prints and returns the means over seeds 1 to 'numSeeds'.
Result simulate(const Job& job, const Policy& policy, int numSeeds) {
  Result mean;
  for (int seed = 1; seed <= numSeeds; ++seed) {
    const auto one = JobSimulation(job, policy, seed).run();
    mean.seconds += one.seconds / numSeeds;
    mean.utilization += one.utilization / numSeeds;
    mean.rejections += one.rejections / numSeeds;
    mean.failedRows += one.failedRows / numSeeds;
    mean.lateDone += one.lateDone / numSeeds;
    mean.peakQueued += one.peakQueued / numSeeds;
    mean.defaultDone += one.defaultDone / numSeeds;
    mean.recoverySeconds += one.recoverySeconds / numSeeds;
  }
  fmt::print(
      "{:<36} {:<16} | {:>8.1f} {:>6.1f} {:>10.0f} {:>8.0f} | {:>7.1f} {:>7.1f} {:>7.0f} {:>7.1f}\n",
      job.name,
      policy.name,
      mean.seconds,
      100 * mean.utilization,
      mean.rejections,
      mean.failedRows,
      mean.lateDone,
      mean.recoverySeconds,
      mean.peakQueued,
      mean.defaultDone);
  std::fflush(stdout);
  return mean;
}

struct Comparison {
  Result window;
  Result probing;
};

Comparison compare(const Job& job, int numSeeds) {
  return {
      simulate(job, kWindowAimd, numSeeds),
      simulate(job, kCapacityProbing, numSeeds),
  };
}

// One worker feeding one of its 16 drivers, 20,000 rows of 4 s calls.
Job narrow(int64_t numSlots, int64_t queueCapacity) {
  return {
      .name =
          fmt::format("narrow, {} slots, queue {}", numSlots, queueCapacity),
      .numWorkers = 1,
      .numDrivers = 16,
      .numActiveDrivers = 1,
      .rowsPerWorker = 20'000,
      .numSlots = numSlots,
      .queueCapacity = queueCapacity,
      .latency = 4'000ms,
      .maxRetries = 6,
      .ceiling = 200,
  };
}

// About 150 workers by 16 drivers, 1,000 rows each, with a function ceiling of
// 25 calls per worker.
Job deep(int64_t numSlots, int64_t queueCapacity, bool pooled) {
  return {
      .name = fmt::format(
          "deep{}, {} slots, queue {}",
          pooled ? " pooled" : "",
          numSlots,
          queueCapacity),
      .numWorkers = 150,
      .numDrivers = 16,
      .numActiveDrivers = 16,
      .rowsPerWorker = 1'000,
      .numSlots = numSlots,
      .queueCapacity = queueCapacity,
      .latency = 1'500ms,
      .maxRetries = 6,
      .ceiling = 25,
      .pooled = pooled,
  };
}

// About 300 workers with about 140 rows each.
Job wide(int64_t numSlots, int64_t queueCapacity, bool pooled) {
  return {
      .name = fmt::format(
          "wide{}, {} slots, queue {}",
          pooled ? " pooled" : "",
          numSlots,
          queueCapacity),
      .numWorkers = 299,
      .numDrivers = 16,
      .numActiveDrivers = 16,
      .rowsPerWorker = 140,
      .numSlots = numSlots,
      .queueCapacity = queueCapacity,
      .latency = 2'000ms,
      .maxRetries = 6,
      .ceiling = 200,
      .pooled = pooled,
  };
}

// One worker feeds one driver, whose window of 100 is the limit.
// CAPACITY_PROBING raises it to the limiter's ceiling of 200. WINDOW_AIMD with
// rpc.congestion.max_window=200 gets the same gain when the backend has room
// for 200 calls, and only part of it when the backend has fewer slots.
TEST(RPCJobSimulationTest, narrow) {
  const Policy windowAimd200{
      "WINDOW_AIMD, 200",
      RPCFlowControlPolicy::kWindowAimd,
      200,
  };
  printHeader();
  for (const auto& job :
       {narrow(150, 0), narrow(600, 0), narrow(600, 600), narrow(2'000, 0)}) {
    const auto [window, probing] = compare(job, 3);
    simulate(job, windowAimd200, 3);
    EXPECT_LT(probing.seconds, window.seconds) << job.name;
    EXPECT_EQ(probing.failedRows, 0) << job.name;
  }
}

TEST(RPCJobSimulationTest, smallJobs) {
  printHeader();
  for (const auto& job : std::vector<Job>{
           {
               .name = "small, 100 rows, 200 slots",
               .numWorkers = 1,
               .numDrivers = 15,
               .numActiveDrivers = 15,
               .rowsPerWorker = 100,
               .numSlots = 200,
               .queueCapacity = 0,
               .latency = 2'000ms,
               .maxRetries = 6,
               .ceiling = 200,
           },
           {
               .name = "small, 2,000 rows, 150 slots",
               .numWorkers = 1,
               .numDrivers = 15,
               .numActiveDrivers = 15,
               .rowsPerWorker = 2'000,
               .numSlots = 150,
               .queueCapacity = 0,
               .latency = 2'000ms,
               .maxRetries = 6,
               .ceiling = 200,
           },
       }) {
    const auto [window, probing] = compare(job, 5);
    EXPECT_LE(probing.seconds, window.seconds) << job.name;
    EXPECT_EQ(probing.failedRows, 0) << job.name;
  }
}

// One driver whose window of 100 happens to match a backend of 100 slots
// without a queue, so WINDOW_AIMD keeps it full and never overloads it.
// CAPACITY_PROBING finds the limit by exceeding it: each capacity rejection
// keeps 80% of the calls the backend held, and the climb back leaves about 6%
// of the slots idle. With any other capacity WINDOW_AIMD's fixed window wastes
// more; see narrow.
TEST(RPCJobSimulationTest, oneDriverAtCapacity) {
  printHeader();
  for (const int64_t rows : {800, 4'000}) {
    const Job job{
        .name = fmt::format("one driver, {} rows, 100 slots", rows),
        .numWorkers = 1,
        .numDrivers = 16,
        .numActiveDrivers = 1,
        .rowsPerWorker = rows,
        .numSlots = 100,
        .queueCapacity = 0,
        .latency = 500ms,
        .maxRetries = 6,
        .ceiling = 200,
    };
    const auto [window, probing] = compare(job, 5);
    EXPECT_LE(probing.seconds, 1.1 * window.seconds) << job.name;
    EXPECT_EQ(probing.failedRows, 0) << job.name;
  }
}

// One active driver, 1,000 rows of 500 ms calls against 100 slots with 100 ms
// retry waits, or 60 slots with 2 s retry waits.
TEST(RPCJobSimulationTest, demoBackends) {
  printHeader();
  for (const auto& [numSlots, retryAfter] :
       std::vector<std::pair<int64_t, std::chrono::milliseconds>>{
           {100, 100ms}, {60, 2'000ms}}) {
    const Job job{
        .name = fmt::format(
            "demo, {} slots, {} ms retries", numSlots, retryAfter.count()),
        .numWorkers = 1,
        .numDrivers = 16,
        .numActiveDrivers = 1,
        .rowsPerWorker = 1'000,
        .numSlots = numSlots,
        .queueCapacity = 0,
        .latency = 500ms,
        .maxRetries = 6,
        .retryAfter = retryAfter,
        .ceiling = 200,
    };
    const auto [window, probing] = compare(job, 5);
    EXPECT_EQ(probing.failedRows, 0) << job.name;
  }
}

// Without a queue, WINDOW_AIMD's floor of 50 calls per worker is above each
// worker's share of the backend, so rows fail after their retries.
// CAPACITY_PROBING goes below the floor and fails none, at the cost of a
// longer job.
TEST(RPCJobSimulationTest, deepAndWideJobs) {
  printHeader();
  for (const auto& job : {
           deep(1'500, 0, /*pooled=*/false),
           deep(1'500, 1'500, /*pooled=*/false),
           deep(6'000, 0, /*pooled=*/false),
           deep(1'500, 0, /*pooled=*/true),
           deep(1'500, 1'500, /*pooled=*/true),
       }) {
    const auto [window, probing] = compare(job, 3);
    EXPECT_EQ(probing.failedRows, 0) << job.name;
  }
  for (const auto& job : {
           wide(3'000, 0, /*pooled=*/false),
           wide(3'000, 3'000, /*pooled=*/false),
           wide(10'000, 0, /*pooled=*/false),
           wide(3'000, 0, /*pooled=*/true),
       }) {
    const auto [window, probing] = compare(job, 3);
    EXPECT_EQ(probing.failedRows, 0) << job.name;
  }
}

// 200 workers share 50 slots, a quarter of a call each. WINDOW_AIMD holds at
// least its floor of 50 calls per worker. CAPACITY_PROBING cuts below one call
// and paces admissions.
TEST(RPCJobSimulationTest, manyWorkersFewSlots) {
  printHeader();
  const Job job{
      .name = "200 workers, 50 slots",
      .numWorkers = 200,
      .numDrivers = 16,
      .numActiveDrivers = 16,
      .rowsPerWorker = 100,
      .numSlots = 50,
      .queueCapacity = 0,
      .latency = 1'000ms,
      .maxRetries = 6,
      .ceiling = 200,
  };
  const auto [window, probing] = compare(job, 3);
  EXPECT_LT(probing.failedRows, window.failedRows);
}

// 8 or 64 workers with 5,000 rows each share a backend with or without a queue.
TEST(RPCJobSimulationTest, contendedJobs) {
  printHeader();
  for (const auto& [numWorkers, numSlots, queueCapacity] :
       std::vector<std::tuple<int, int64_t, int64_t>>{
           {8, 500, 0},
           {8, 500, 125},
           {8, 1'000, 0},
           {64, 2'000, 2'000},
           {64, 6'000, 1'500},
       }) {
    const Job job{
        .name = fmt::format(
            "{} workers, {} slots, queue {}",
            numWorkers,
            numSlots,
            queueCapacity),
        .numWorkers = numWorkers,
        .numDrivers = 16,
        .numActiveDrivers = 16,
        .rowsPerWorker = 5'000,
        .numSlots = numSlots,
        .queueCapacity = queueCapacity,
        .latency = 1'000ms,
        .maxRetries = 6,
        .ceiling = 200,
    };
    const auto [window, probing] = compare(job, 3);
    EXPECT_EQ(probing.failedRows, 0) << job.name;
  }
}

// One worker with 15 drivers against a backend that queues whatever it cannot
// serve, so it never rejects, with only the safety ceiling of
// rpc.ratelimiter.hard_limit=0. CAPACITY_PROBING stops doubling once a round's
// responses slow down, one round after the queue forms, and then grows one call
// per round. Without that it would fill the drivers' windows: 15 of 200 calls.
TEST(RPCJobSimulationTest, queueingBackend) {
  printHeader();
  const Job job{
      .name = "queueing backend, 100 slots",
      .numWorkers = 1,
      .numDrivers = 15,
      .numActiveDrivers = 15,
      .rowsPerWorker = 30'000,
      .numSlots = 100,
      .queueCapacity = 1'000'000,
      .latency = 3'000ms,
      .maxRetries = 6,
      .ceiling = 1'000'000,
      .safetyCeilingOnly = true,
  };
  const auto [window, probing] = compare(job, 3);
  EXPECT_LT(probing.peakQueued, 1'000);
  EXPECT_LE(probing.seconds, 1.01 * window.seconds);
}

// 20 workers with 20 s calls start over 30 s and share 1,000 slots, 50 each, at
// WINDOW_AIMD's floor. WINDOW_AIMD fails rows after their retries;
// CAPACITY_PROBING fails none and so takes longer.
TEST(RPCJobSimulationTest, longCallsStaggeredStarts) {
  printHeader();
  const Job job{
      .name = "20 workers, 20 s calls, start over 30 s",
      .numWorkers = 20,
      .numDrivers = 16,
      .numActiveDrivers = 16,
      .rowsPerWorker = 2'250,
      .numSlots = 1'000,
      .queueCapacity = 0,
      .latency = 20'000ms,
      .maxRetries = 6,
      .ceiling = 200,
      .startSpread = 30s,
  };
  const auto [window, probing] = compare(job, 3);
  // Guards the scenario: WINDOW_AIMD fails rows here.
  EXPECT_GT(window.failedRows, 0);
  EXPECT_EQ(probing.failedRows, 0);
}

// Four of eight workers start 30 s late on 400 slots, 50 each.
TEST(RPCJobSimulationTest, lateQuerySmallShares) {
  printHeader();
  const Job job{
      .name = "late by 30 s, 8 workers, 400 slots",
      .numWorkers = 8,
      .numDrivers = 16,
      .numActiveDrivers = 16,
      .rowsPerWorker = 5'000,
      .numSlots = 400,
      .queueCapacity = 0,
      .latency = 3'000ms,
      .maxRetries = 6,
      .ceiling = 200,
      .numLateWorkers = 4,
      .lateStart = 30s,
  };
  const auto [window, probing] = compare(job, 3);
  EXPECT_LE(probing.lateDone, 1.05 * window.lateDone);
  EXPECT_EQ(probing.failedRows, 0);
}

// A query of 400 rows arrives a minute after four workers with 3,000 rows each
// filled 400 slots with 20 s calls. The incumbents give up room 20% at a time,
// more slowly than WINDOW_AIMD's halving, so the small query finishes later,
// while the whole job finishes sooner.
TEST(RPCJobSimulationTest, smallQueryAtBusyBackend) {
  printHeader();
  const Job job{
      .name = "small query at a busy backend, 20 s",
      .numWorkers = 5,
      .numDrivers = 16,
      .numActiveDrivers = 16,
      .rowsPerWorker = 3'000,
      .numSlots = 400,
      .queueCapacity = 0,
      .latency = 20'000ms,
      .maxRetries = 6,
      .ceiling = 200,
      .numLateWorkers = 1,
      .lateStart = 60s,
      .lateRowsPerWorker = 400,
  };
  const auto [window, probing] = compare(job, 3);
  EXPECT_LE(probing.lateDone, 1.2 * window.lateDone);
  EXPECT_LT(probing.seconds, window.seconds);
  EXPECT_EQ(probing.failedRows, 0);
}

// Ten default workers share 2,000 slots with ten opted-in ones. "default" is
// when the default workers finish, against when they finish beside ten more
// default workers: opted-in workers take somewhat more than their share.
TEST(RPCJobSimulationTest, mixedPolicies) {
  printHeader();
  const Job job{
      .name = "10 default + 10 opted-in, 2,000 slots",
      .numWorkers = 20,
      .numDrivers = 16,
      .numActiveDrivers = 16,
      .rowsPerWorker = 5'000,
      .numSlots = 2'000,
      .queueCapacity = 0,
      .latency = 3'000ms,
      .maxRetries = 6,
      .ceiling = 200,
      .numDefaultWorkers = 10,
  };
  const auto [window, probing] = compare(job, 5);
  EXPECT_LE(probing.defaultDone, 1.1 * window.defaultDone);
  EXPECT_EQ(probing.failedRows, 0);
}

// A second query starts on two of four workers after the first query's two
// workers already hold the backend's 400 slots.
TEST(RPCJobSimulationTest, lateQuery) {
  printHeader();
  for (const auto& [ceiling, lateStart] :
       std::vector<std::pair<int64_t, std::chrono::seconds>>{
           {200, 30s},
           {400, 30s},
           {200, 120s},
       }) {
    const Job job{
        .name =
            fmt::format("late by {} s, ceiling {}", lateStart.count(), ceiling),
        .numWorkers = 4,
        .numDrivers = 16,
        .numActiveDrivers = 16,
        .rowsPerWorker = 13'000,
        .numSlots = 400,
        .queueCapacity = 0,
        .latency = 3'000ms,
        .maxRetries = 6,
        .ceiling = ceiling,
        .numLateWorkers = 2,
        .lateStart = lateStart,
    };
    const auto [window, probing] = compare(job, 3);
    EXPECT_LT(probing.lateDone, window.lateDone) << job.name;
    EXPECT_EQ(probing.failedRows, 0) << job.name;
  }
}

// 200 workers with fixed rows start over 10 s on 5,000 slots.
TEST(RPCJobSimulationTest, startSkew) {
  printHeader();
  const Job job{
      .name = "200 workers starting over 10 s",
      .numWorkers = 200,
      .numDrivers = 16,
      .numActiveDrivers = 16,
      .rowsPerWorker = 3'600,
      .numSlots = 5'000,
      .queueCapacity = 0,
      .latency = 3'000ms,
      .maxRetries = 5,
      .ceiling = 200,
      .startSpread = 10s,
  };
  const auto [window, probing] = compare(job, 3);
  // Guards the scenario: WINDOW_AIMD fails rows here.
  EXPECT_GT(window.failedRows, 0);
  EXPECT_EQ(probing.failedRows, 0);
  EXPECT_LE(probing.seconds, 1.1 * window.seconds);
}

// The backend rejects every call for 5-60 s, then comes back. WINDOW_AIMD
// halves to its floor and grows back one call per round; CAPACITY_PROBING
// doubles back to the window it had.
TEST(RPCJobSimulationTest, outage) {
  printHeader();
  for (const auto length : {5s, 10s, 60s}) {
    const Job job{
        .name = fmt::format("outage of {} s", length.count()),
        .numWorkers = 1,
        .numDrivers = 16,
        .numActiveDrivers = 16,
        .rowsPerWorker = 30'000,
        .numSlots = 100,
        .queueCapacity = 0,
        .latency = 3'000ms,
        .maxRetries = 6,
        .ceiling = 200,
        .outageStart = 60s,
        .outageLength = length,
    };
    const auto [window, probing] = compare(job, 3);
    EXPECT_TRUE(std::isfinite(probing.recoverySeconds)) << job.name;
    EXPECT_LT(probing.recoverySeconds, window.recoverySeconds) << job.name;
    EXPECT_LT(probing.failedRows, window.failedRows) << job.name;
  }
}

} // namespace
} // namespace facebook::velox::exec::rpc
