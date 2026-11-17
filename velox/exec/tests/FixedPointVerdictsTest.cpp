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

// Only the latest iteration's verdict is kept: a new iteration replaces it,
// and asking for an earlier one fails.
TEST(FixedPointVerdictsTest, latestIteration) {
  FixedPointVerdicts verdicts;
  const FixedPointOptions::LoopAddress loop{"w0", {}};

  verdicts.publish(loop, 0, false);
  auto next = verdicts.get(loop, 1);
  ASSERT_FALSE(next.isReady());
  verdicts.publish(loop, 1, true);
  ASSERT_TRUE(std::move(next).get());

  VELOX_ASSERT_THROW(
      verdicts.get(loop, 0),
      "Fixed point verdict requested for an earlier iteration: w0");
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
