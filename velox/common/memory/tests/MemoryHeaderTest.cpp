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

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"

using namespace ::testing;
using namespace facebook::velox::memory;

TEST(MemoryHeaderTest, GetProcessDefaultMemoryManager) {
  auto& managerA = getProcessDefaultMemoryManager();
  ASSERT_EQ(managerA.numPools(), 0);
  auto& managerB = getProcessDefaultMemoryManager();
  ASSERT_EQ(managerB.numPools(), 0);

  auto poolFromA = managerA.getPool("child_1");
  auto poolFromB = managerB.getPool("child_2");
  ASSERT_EQ(2, managerA.numPools());
  ASSERT_EQ(2, managerB.numPools());
  poolFromA.reset();
  ASSERT_EQ(managerA.numPools(), 1);
  ASSERT_EQ(managerB.numPools(), 1);
  poolFromA.reset();
  ASSERT_EQ(managerA.numPools(), 0);
  ASSERT_EQ(managerB.numPools(), 0);
}

TEST(MemoryHeaderTest, getDefaultMemoryPool) {
  auto& manager = getProcessDefaultMemoryManager();
  ASSERT_EQ(manager.numPools(), 0);
  {
    auto poolA = getDefaultMemoryPool();
    auto poolB = getDefaultMemoryPool();
    ASSERT_EQ(manager.numPools(), 2);
    {
      auto poolC = getDefaultMemoryPool();
      ASSERT_EQ(manager.numPools(), 3);
      {
        auto poolD = getDefaultMemoryPool();
        ASSERT_EQ(manager.numPools(), 4);
      }
      ASSERT_EQ(manager.numPools(), 3);
    }
    ASSERT_EQ(manager.numPools(), 2);
  }
  ASSERT_EQ(manager.numPools(), 0);
}