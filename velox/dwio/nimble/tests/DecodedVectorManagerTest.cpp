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
#include <gtest/gtest.h>

#include "velox/dwio/nimble/writer/DecodedVectorManager.h"

using namespace facebook;

TEST(DecodedVectorManagerTest, firstAcquisitionCreatesNewVector) {
  nimble::DecodedVectorManager manager;
  nimble::DecodedVectorManager::LocalDecodedVector local{manager};
  auto& vec = local.get();
  // Simply verify the returned reference is valid by checking it's addressable.
  EXPECT_NE(&vec, nullptr);
}

TEST(DecodedVectorManagerTest, poolReusesReleasedVector) {
  nimble::DecodedVectorManager manager;
  velox::DecodedVector* firstAddr;
  {
    nimble::DecodedVectorManager::LocalDecodedVector local{manager};
    firstAddr = &local.get();
  }
  // After the first LocalDecodedVector is destroyed, the vector should be
  // returned to the pool. The next acquisition should reuse it.
  nimble::DecodedVectorManager::LocalDecodedVector local2{manager};
  EXPECT_EQ(&local2.get(), firstAddr);
}

TEST(DecodedVectorManagerTest, multipleConcurrentBorrowsGetDistinctVectors) {
  nimble::DecodedVectorManager manager;
  nimble::DecodedVectorManager::LocalDecodedVector a{manager};
  nimble::DecodedVectorManager::LocalDecodedVector b{manager};
  EXPECT_NE(&a.get(), &b.get());
}

TEST(DecodedVectorManagerTest, moveConstructorTransfersOwnership) {
  nimble::DecodedVectorManager manager;
  velox::DecodedVector* addr;
  {
    nimble::DecodedVectorManager::LocalDecodedVector original{manager};
    addr = &original.get();
    nimble::DecodedVectorManager::LocalDecodedVector moved{std::move(original)};
    // The moved-to instance should hold the same vector.
    EXPECT_EQ(&moved.get(), addr);
    // After move, destroying the original should not return the vector to the
    // pool (it's null). The moved-to instance is destroyed here and returns the
    // vector to the pool.
  }
  // Verify the vector was returned to the pool exactly once.
  nimble::DecodedVectorManager::LocalDecodedVector reused{manager};
  EXPECT_EQ(&reused.get(), addr);
}

TEST(DecodedVectorManagerTest, poolGrowsWithMultipleReturns) {
  nimble::DecodedVectorManager manager;
  velox::DecodedVector* addr1;
  velox::DecodedVector* addr2;
  {
    nimble::DecodedVectorManager::LocalDecodedVector a{manager};
    nimble::DecodedVectorManager::LocalDecodedVector b{manager};
    addr1 = &a.get();
    addr2 = &b.get();
  }
  // Both vectors are now in the pool. C++ destroys locals in reverse order,
  // so b is destroyed first (pushing addr2), then a (pushing addr1).
  // Pool is LIFO (vector pop_back), so addr1 is acquired first.
  nimble::DecodedVectorManager::LocalDecodedVector c{manager};
  nimble::DecodedVectorManager::LocalDecodedVector d{manager};
  EXPECT_EQ(&c.get(), addr1);
  EXPECT_EQ(&d.get(), addr2);
}
