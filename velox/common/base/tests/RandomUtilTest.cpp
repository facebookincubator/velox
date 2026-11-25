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

#include "velox/common/base/RandomUtil.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <vector>

namespace facebook::velox::random {
namespace {

// forEachHit() reports the same hits as one testOne() per element, including
// when calls split a gap.
TEST(RandomSkipTrackerTest, forEachHitMatchesTestOne) {
  for (const double sampleRate : {0.0, 0.0002, 0.01, 0.5, 0.99}) {
    for (uint32_t seed = 1; seed <= 10; ++seed) {
      SCOPED_TRACE(fmt::format("sampleRate={}, seed={}", sampleRate, seed));
      setSeed(seed);
      RandomSkipTracker expectedSkip(sampleRate);
      setSeed(seed);
      RandomSkipTracker skip(sampleRate);
      uint64_t offset = 0;
      for (const uint64_t numElements : {0, 1, 7, 100, 3, 10'000, 1}) {
        std::vector<uint64_t> expected;
        for (uint64_t i = 0; i < numElements; ++i) {
          if (expectedSkip.testOne()) {
            expected.push_back(offset + i);
          }
        }
        std::vector<uint64_t> actual;
        skip.forEachHit(
            numElements, [&](uint64_t i) { actual.push_back(offset + i); });
        EXPECT_EQ(actual, expected);
        offset += numElements;
      }
    }
  }
}

} // namespace
} // namespace facebook::velox::random
