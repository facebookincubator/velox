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

#include "velox/experimental/cxl/CxlMemoryResource.h"

#include <numa.h>

#include "velox/common/memory/SharedArbitrator.h"

#include <gtest/gtest.h>

namespace facebook::velox::cxl {
namespace {

class CxlMemoryResourceTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    // makeCxlMemoryResource creates a SHARED arbitrator, whose factory must be
    // registered first.
    memory::SharedArbitrator::registerFactory();
  }

  static void TearDownTestCase() {
    memory::SharedArbitrator::unregisterFactory();
  }

  void SetUp() override {
    if (numa_available() < 0) {
      GTEST_SKIP() << "NUMA is unavailable on this host.";
    }
  }
};

TEST_F(CxlMemoryResourceTest, buildsResource) {
  constexpr int64_t kMaxCapacity = 64 << 20; // 64MB
  auto resource = makeCxlMemoryResource(0, kMaxCapacity);
  ASSERT_NE(resource, nullptr);
  EXPECT_EQ(resource->tag(), std::string{kCxlResourceTag});
  EXPECT_NE(resource->allocator(), nullptr);
  EXPECT_NE(resource->arbitrator(), nullptr);
}

} // namespace
} // namespace facebook::velox::cxl
