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

#include "velox/dwio/nimble/velox/DeduplicationUtils.h"
#include <gtest/gtest.h>
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook;

class DeduplicationUtilsTests : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::SharedArbitrator::registerFactory();
    velox::memory::MemoryManager::Options options;
    options.arbitratorKind = "SHARED";
    velox::memory::MemoryManager::testingSetInstance(options);
  }

  static void TearDownTestCase() {
    // registerFactory() throws if the kind is already registered, and every
    // suite in this binary runs SetUpTestCase. Pairing them leaves the
    // process-global registry as each suite found it, matching
    // OperatorTestBase::TearDownTestCase.
    velox::memory::SharedArbitrator::unregisterFactory();
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("default_root");
    leafPool_ = rootPool_->addLeafChild("default_leaf");
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(DeduplicationUtilsTests, testOrdered) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto keys = vectorMaker.flatVector<int32_t>({1, 2, 1, 2});
  auto vals = vectorMaker.flatVector<float>({0.3, 0.4, 0.3, 0.4});
  auto map = vectorMaker.mapVector({{0, 2}}, keys, vals);
  EXPECT_TRUE(nimble::DeduplicationUtils::CompareMapsAtIndex(*map, 0, *map, 1));
}

TEST_F(DeduplicationUtilsTests, testDedupUnordered) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto keys = vectorMaker.flatVector<int32_t>({1, 2, 2, 1});
  auto vals = vectorMaker.flatVector<float>({0.3, 0.4, 0.4, 0.3});
  auto map = vectorMaker.mapVector({{0, 2}}, keys, vals);
  EXPECT_TRUE(nimble::DeduplicationUtils::CompareMapsAtIndex(*map, 0, *map, 1));
}

TEST_F(DeduplicationUtilsTests, testUnorderedNotMatched) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto keys = vectorMaker.flatVector<int32_t>({1, 2, 2, 1});
  auto vals = vectorMaker.flatVector<float>({0.3, 0.4, 0.3, 0.4});
  auto map = vectorMaker.mapVector({{0, 2}}, keys, vals);
  EXPECT_FALSE(
      nimble::DeduplicationUtils::CompareMapsAtIndex(*map, 0, *map, 1));
}

TEST_F(DeduplicationUtilsTests, testLengthNotMatched) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto keys = vectorMaker.flatVector<int32_t>({1, 2, 2});
  auto vals = vectorMaker.flatVector<float>({0.3, 0.4, 0.3});
  auto map = vectorMaker.mapVector({{0, 2}}, keys, vals);
  EXPECT_FALSE(
      nimble::DeduplicationUtils::CompareMapsAtIndex(*map, 0, *map, 1));
}

TEST_F(DeduplicationUtilsTests, testNullNotMatched) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto keys = vectorMaker.flatVector<int32_t>({1, 2});
  auto vals = vectorMaker.flatVector<float>({0.3, 0.4});
  auto map = vectorMaker.mapVector({{0, 2}}, keys, vals, {1});
  EXPECT_FALSE(
      nimble::DeduplicationUtils::CompareMapsAtIndex(*map, 0, *map, 1));
}
