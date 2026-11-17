/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/futures/SharedPromise.h>
#include <functional>
#include <map>
#include <string>

#include "velox/exec/FixedPointOptions.h"

namespace facebook::velox::exec {

/// Holds the latest convergence verdict of each fixed point loop, for an
/// application that carries verdicts from worker 0 to its peers in memory:
/// FixedPointOptions::publishVerdict on worker 0 publishes into it, and
/// awaitVerdict on every other worker gets from it. Both hooks receive worker
/// 0's FixedPointOptions::LoopAddress, which keys the verdict.
///
///   FixedPointVerdicts verdicts;
///   verdicts.publish(workerZero, iteration, converged);  // On worker 0.
///   auto verdict = verdicts.get(workerZero, iteration);  // On a peer.
///
/// The workers of a loop run in lockstep, so once one of them reaches a new
/// iteration's verdict, none needs the previous one: only the latest is kept.
class FixedPointVerdicts {
 public:
  /// Records worker 0's verdict for 'iteration' of its loop.
  void publish(
      const FixedPointOptions::LoopAddress& workerZero,
      int32_t iteration,
      bool converged);

  /// Returns the verdict for 'iteration' of the loop of 'workerZero', which
  /// resolves once worker 0 publishes it.
  folly::SemiFuture<bool> get(
      const FixedPointOptions::LoopAddress& workerZero,
      int32_t iteration);

  /// Drops the verdicts of the loops whose worker 0 is 'worker'.
  void erase(const std::string& worker);

 private:
  struct Verdict {
    int32_t iteration;
    folly::SharedPromise<bool> promise;
  };

  // Runs 'action' on the promise of the verdict for 'iteration' of the loop,
  // under the lock.
  void withVerdict(
      const FixedPointOptions::LoopAddress& workerZero,
      int32_t iteration,
      const std::function<void(folly::SharedPromise<bool>&)>& action);

  // Worker 0's name to the latest verdict of each of its loops, by nesting.
  folly::Synchronized<folly::F14FastMap<
      std::string,
      std::map<std::vector<FixedPointOptions::NestingStep>, Verdict>>>
      verdicts_;
};

} // namespace facebook::velox::exec
