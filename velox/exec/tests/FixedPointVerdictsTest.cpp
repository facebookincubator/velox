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

#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::exec {
namespace {

// A verdict reaches a peer whether the peer asks before or after worker 0
// publishes it, and each loop of worker 0 has its own verdict.
TEST(FixedPointVerdictsTest, publishAndGet) {
  FixedPointVerdicts verdicts;
  const FixedPointOptions::LoopAddress loop{"w0", {}};
  const FixedPointOptions::LoopAddress nestedLoop{
      "w0", {{.planNodeId = "1", .iteration = 0}}};

  auto early = verdicts.get(loop, 0);
  ASSERT_FALSE(early.isReady());
  verdicts.publish(loop, 0, true);
  ASSERT_TRUE(std::move(early).get());
  ASSERT_TRUE(verdicts.get(loop, 0).get());

  ASSERT_FALSE(verdicts.get(nestedLoop, 0).isReady());
}

// The store keeps each loop's latest iteration. A request for it gets its
// verdict, and a request for an earlier iteration gets "not converged": worker
// 0 starts an iteration only after one that did not converge.
TEST(FixedPointVerdictsTest, latestIteration) {
  FixedPointVerdicts verdicts;
  const FixedPointOptions::LoopAddress loop{"w0", {}};

  verdicts.publish(loop, 0, false);
  auto next = verdicts.get(loop, 1);
  ASSERT_FALSE(next.isReady());
  verdicts.publish(loop, 1, true);
  ASSERT_TRUE(std::move(next).get());

  auto earlier = verdicts.get(loop, 0);
  ASSERT_TRUE(earlier.isReady());
  ASSERT_FALSE(std::move(earlier).get());
}

// Worker 0 runs ahead of a peer that has yet to receive an earlier verdict, as
// when the body moves no rows between workers and worker 0 decides from its own
// rows. The slow peer still gets "not converged" for every iteration worker 0
// passed, and a fast peer's request for the next iteration waits for it.
TEST(FixedPointVerdictsTest, runAhead) {
  FixedPointVerdicts verdicts;
  const FixedPointOptions::LoopAddress loop{"w0", {}};

  verdicts.publish(loop, 0, false);
  verdicts.publish(loop, 1, false);
  verdicts.publish(loop, 2, false);

  auto fast = verdicts.get(loop, 3);
  ASSERT_FALSE(fast.isReady());

  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    auto slow = verdicts.get(loop, iteration);
    ASSERT_TRUE(slow.isReady());
    ASSERT_FALSE(std::move(slow).get());
  }

  verdicts.publish(loop, 3, true);
  ASSERT_TRUE(std::move(fast).get());
}

// A peer asks for an iteration only after it receives the one before, so a
// request that skips an unpublished verdict is an internal error.
TEST(FixedPointVerdictsTest, skippedVerdict) {
  FixedPointVerdicts verdicts;
  const FixedPointOptions::LoopAddress loop{"w0", {}};

  auto pending = verdicts.get(loop, 0);
  VELOX_ASSERT_THROW(
      verdicts.get(loop, 1),
      "Fixed point verdict must be published before the next is requested: w0");
}

// Erasing worker 0 drops the verdicts of all its loops.
TEST(FixedPointVerdictsTest, erase) {
  FixedPointVerdicts verdicts;
  const FixedPointOptions::LoopAddress loop{"w0", {}};
  const FixedPointOptions::LoopAddress otherTask{"w1", {}};

  verdicts.publish(loop, 3, true);
  verdicts.publish(otherTask, 3, true);
  verdicts.erase("w0");

  ASSERT_FALSE(verdicts.get(loop, 0).isReady());
  ASSERT_TRUE(verdicts.get(otherTask, 3).get());
}

} // namespace
} // namespace facebook::velox::exec
