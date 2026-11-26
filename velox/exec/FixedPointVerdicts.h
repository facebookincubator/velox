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
/// Worker 0 can run ahead of its peers, but it starts iteration t+1 only when
/// the verdict for t is "not converged". So keeping each loop's latest
/// iteration answers every request:
///  - an earlier iteration is "not converged";
///  - the latest iteration gets its verdict, once worker 0 publishes it;
///  - a later iteration waits until worker 0 publishes it.
class FixedPointVerdicts {
 public:
  /// Records worker 0's verdict for 'iteration' of its loop.
  void publish(
      const FixedPointOptions::LoopAddress& workerZero,
      int32_t iteration,
      bool converged);

  /// Returns the verdict for 'iteration' of the loop of 'workerZero'. It
  /// resolves to false at once for an iteration before the loop's latest, and
  /// otherwise once worker 0 publishes the verdict.
  folly::SemiFuture<bool> get(
      const FixedPointOptions::LoopAddress& workerZero,
      int32_t iteration);

  /// Drops the verdicts of the loops whose worker 0 is 'worker'.
  void erase(const std::string& worker);

 private:
  // A loop's latest iteration, and the verdict worker 0 published or will
  // publish for it.
  struct Verdict {
    int32_t iteration;
    folly::SharedPromise<bool> promise;
  };

  // Worker 0's name to the latest verdict of each of its loops, by nesting.
  folly::Synchronized<folly::F14FastMap<
      std::string,
      std::map<std::vector<FixedPointOptions::NestingStep>, Verdict>>>
      verdicts_;
};

} // namespace facebook::velox::exec
