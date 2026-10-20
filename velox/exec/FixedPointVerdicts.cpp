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
#include "velox/exec/FixedPointVerdicts.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {

void FixedPointVerdicts::publish(
    const FixedPointOptions::LoopAddress& workerZero,
    int32_t iteration,
    bool converged) {
  withVerdict(workerZero, iteration, [&](folly::SharedPromise<bool>& promise) {
    promise.setValue(converged);
  });
}

folly::SemiFuture<bool> FixedPointVerdicts::get(
    const FixedPointOptions::LoopAddress& workerZero,
    int32_t iteration) {
  auto verdict = folly::SemiFuture<bool>::makeEmpty();
  withVerdict(workerZero, iteration, [&](folly::SharedPromise<bool>& promise) {
    verdict = promise.getSemiFuture();
  });
  return verdict;
}

void FixedPointVerdicts::erase(const std::string& rootTaskId) {
  verdicts_.wlock()->erase(rootTaskId);
}

void FixedPointVerdicts::withVerdict(
    const FixedPointOptions::LoopAddress& workerZero,
    int32_t iteration,
    const std::function<void(folly::SharedPromise<bool>&)>& action) {
  auto locked = verdicts_.wlock();
  auto& loops = (*locked)[workerZero.rootTaskId];
  auto [it, inserted] =
      loops.try_emplace(workerZero.address, Verdict{iteration, {}});
  auto& verdict = it->second;
  VELOX_CHECK_LE(
      verdict.iteration,
      iteration,
      "Fixed point verdict requested for an earlier iteration: {}",
      workerZero.address);
  if (verdict.iteration < iteration) {
    verdict = Verdict{iteration, {}};
  }
  action(verdict.promise);
}

} // namespace facebook::velox::exec
