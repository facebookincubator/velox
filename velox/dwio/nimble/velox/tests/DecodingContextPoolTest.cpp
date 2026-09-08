/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <folly/system/HardwareConcurrency.h>
#include <gtest/gtest.h>
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/FieldWriter.h"

namespace {
using DecodingContext = facebook::nimble::DecodingContextPool::DecodingContext;
using DecodingContextPool = facebook::nimble::DecodingContextPool;

TEST(DeocodingContextPoolTest, vectorDecoderVisitorMissingThrows) {
  try {
    auto pool = DecodingContextPool{/* vectorDecoderVisitor*/ nullptr};
    FAIL();
  } catch (facebook::nimble::NimbleInternalError& e) {
    ASSERT_EQ(e.errorMessage(), "vectorDecoderVisitor must be set");
  }
}

TEST(DecodingContextPoolTest, reserveAddPair) {
  DecodingContextPool pool;
  EXPECT_EQ(pool.size(), 0);
  {
    // inner scope
    auto context = pool.reserveContext();
    EXPECT_EQ(pool.size(), 0);
  }

  EXPECT_EQ(pool.size(), 1);
}

TEST(DecodingContextPoolTest, fillPool) {
  DecodingContextPool pool;
  EXPECT_EQ(pool.size(), 0);

  { // inner scope
    auto context1 = pool.reserveContext();
    auto context2 = pool.reserveContext();
    auto context3 = pool.reserveContext();
    auto context4 = pool.reserveContext();
    EXPECT_EQ(pool.size(), 0);
  }

  EXPECT_EQ(pool.size(), 4);
}

TEST(DecodingContextPoolTest, parallelFillPool) {
  auto parallelismFactor = folly::available_concurrency();
  auto executor = folly::CPUThreadPoolExecutor{parallelismFactor};

  DecodingContextPool pool;
  EXPECT_EQ(pool.size(), 0);

  for (auto i = 0; i < parallelismFactor; ++i) {
    executor.add([&]() {
      for (auto j = 0; j < 100000; ++j) {
        auto context = pool.reserveContext();
      }
    });
  }
  executor.join();
  EXPECT_LE(pool.size(), parallelismFactor);
}
} // namespace
