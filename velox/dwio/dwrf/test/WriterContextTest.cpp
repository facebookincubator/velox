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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/dwrf/writer/WriterContext.h"
#include "velox/exec/MemoryReclaimer.h"

using namespace ::testing;

namespace facebook::velox::dwrf {
class WriterContextTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(WriterContextTest, getIntDictionaryEncoder) {
  auto config = std::make_shared<Config>();
  WriterContext context{
      config, memory::memoryManager()->addRootPool("GetIntDictionaryEncoder")};

  auto& dictionaryPool = context.getMemoryPool(MemoryUsageCategory::DICTIONARY);
  auto& generalPool = context.getMemoryPool(MemoryUsageCategory::GENERAL);
  auto& intEncoder_1_0 = context.getIntDictionaryEncoder<int32_t>(
      {1, 0}, dictionaryPool, generalPool);

  auto& duplicateCallResult_1_0 = context.getIntDictionaryEncoder<int32_t>(
      {1, 0}, dictionaryPool, generalPool);
  EXPECT_EQ(&intEncoder_1_0, &duplicateCallResult_1_0);

  auto& intEncoder_2_0 = context.getIntDictionaryEncoder<int32_t>(
      {2, 0}, dictionaryPool, generalPool);
  EXPECT_NE(&intEncoder_1_0, &intEncoder_2_0);
  EXPECT_EQ(0, intEncoder_2_0.size());
}

TEST_F(WriterContextTest, removeIntDictionaryEncoderForNode) {
  auto config = std::make_shared<Config>();
  config->set(Config::MAP_FLAT_DICT_SHARE, false);
  WriterContext context{
      config,
      memory::memoryManager()->addRootPool(
          "RemoveIntDictionaryEncoderForNode")};

  auto& dictionaryPool = context.getMemoryPool(MemoryUsageCategory::DICTIONARY);
  auto& generalPool = context.getMemoryPool(MemoryUsageCategory::GENERAL);
  const std::vector<EncodingKey> keys{
      {1, 1}, {1, 2}, {1, 4}, {1, 5}, {2, 0}, {3, 1}, {3, 3}};
  for (const auto& key : keys) {
    context.getIntDictionaryEncoder<int32_t>(key, dictionaryPool, generalPool);
  }

  const auto removeAndCollectNodes = [&](uint32_t nodeToRemove) {
    std::vector<uint32_t> visitedNodes;
    context.removeAllIntDictionaryEncodersOnNode([&](uint32_t nodeId) {
      visitedNodes.push_back(nodeId);
      return nodeId == nodeToRemove;
    });
    std::sort(visitedNodes.begin(), visitedNodes.end());
    return visitedNodes;
  };

  EXPECT_EQ(
      (std::vector<uint32_t>{1, 1, 1, 1, 2, 3, 3}), removeAndCollectNodes(1));
  EXPECT_EQ((std::vector<uint32_t>{2, 3, 3}), removeAndCollectNodes(3));
  EXPECT_EQ((std::vector<uint32_t>{2}), removeAndCollectNodes(2));
  EXPECT_TRUE(removeAndCollectNodes(0).empty());
}

TEST_F(WriterContextTest, buildPhysicalSizeAggregators) {
  auto config = std::make_shared<Config>();
  WriterContext context{
      config,
      memory::memoryManager()->addRootPool("BuildPhysicalSizeAggregators")};
  auto type = ROW({
      {"array", ARRAY(REAL())},
      {"map", MAP(INTEGER(), DOUBLE())},
      {"row",
       ROW({
           {"a", REAL()},
           {"b", INTEGER()},
       })},
      {"nested",
       ARRAY(ROW({
           {"a", INTEGER()},
           {"b", MAP(REAL(), REAL())},
       }))},
  });
  auto typeWithId = velox::dwio::common::TypeWithId::create(type);
  context.buildPhysicalSizeAggregators(*typeWithId);
  std::vector<uint32_t> mapNodes{3, 12};
  for (size_t i = 0; i < 14; ++i) {
    EXPECT_NO_THROW(context.getPhysicalSizeAggregator(i));
  }
  for (const auto nodeId : mapNodes) {
    EXPECT_NO_THROW(
        std::ignore = dynamic_cast<MapPhysicalSizeAggregator&>(
            context.getPhysicalSizeAggregator(nodeId)));
  }
}

TEST_F(WriterContextTest, memory) {
  auto writerRoot = memory::memoryManager()->addRootPool(
      "memory", 1L << 30, exec::MemoryReclaimer::create());
  WriterContext context{std::make_shared<Config>(), writerRoot};
  ASSERT_EQ(context.getTotalMemoryUsage(), 0);
  context.initBuffer();
  VELOX_ASSERT_THROW(context.initBuffer(), "");
  // The writer context has some initial memory allocation on construction.
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208);
  ASSERT_EQ(context.availableMemoryReservation(), 786368);

  auto& generalPool = context.getMemoryPool(MemoryUsageCategory::GENERAL);
  auto& dictPool = context.getMemoryPool(MemoryUsageCategory::DICTIONARY);
  auto& outputPool = context.getMemoryPool(MemoryUsageCategory::OUTPUT_STREAM);
  ASSERT_TRUE(generalPool.reclaimer() == nullptr);
  ASSERT_TRUE(dictPool.reclaimer() == nullptr);
  ASSERT_TRUE(outputPool.reclaimer() == nullptr);
  const int bufferSize{1024};
  void* generalBuf = generalPool.allocate(bufferSize);
  void* dictBuf = dictPool.allocate(bufferSize);
  void* outputBuf = outputPool.allocate(bufferSize);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208 + bufferSize * 3);
  ASSERT_EQ(context.availableMemoryReservation(), 2880448);

  ASSERT_EQ(generalPool.usedBytes(), 262208 + bufferSize);
  ASSERT_EQ(generalPool.reservedBytes(), 1048576);
  ASSERT_EQ(dictPool.usedBytes(), bufferSize);
  ASSERT_EQ(dictPool.reservedBytes(), 1048576);
  ASSERT_EQ(outputPool.usedBytes(), bufferSize);
  ASSERT_EQ(outputPool.reservedBytes(), 1048576);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208 + bufferSize * 3);
  ASSERT_EQ(context.availableMemoryReservation(), 2880448);

  ASSERT_TRUE(generalPool.maybeReserve(4L << 20));
  ASSERT_TRUE(dictPool.maybeReserve(4L << 20));
  ASSERT_TRUE(outputPool.maybeReserve(4L << 20));
  ASSERT_EQ(generalPool.usedBytes(), 262208 + bufferSize);
  ASSERT_EQ(generalPool.reservedBytes(), 9437184);
  ASSERT_EQ(dictPool.usedBytes(), bufferSize);
  ASSERT_EQ(dictPool.reservedBytes(), 9437184);
  ASSERT_EQ(outputPool.usedBytes(), bufferSize);
  ASSERT_EQ(outputPool.reservedBytes(), 9437184);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208 + bufferSize * 3);
  ASSERT_EQ(context.availableMemoryReservation(), 28046272);

  context.releaseMemoryReservation();
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208 + bufferSize * 3);
  ASSERT_EQ(generalPool.usedBytes(), 262208 + bufferSize);
  ASSERT_EQ(generalPool.reservedBytes(), 1048576);
  ASSERT_EQ(dictPool.usedBytes(), bufferSize);
  ASSERT_EQ(dictPool.reservedBytes(), 1048576);
  ASSERT_EQ(outputPool.usedBytes(), bufferSize);
  ASSERT_EQ(outputPool.reservedBytes(), 1048576);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208 + bufferSize * 3);
  ASSERT_EQ(context.availableMemoryReservation(), 2880448);

  generalPool.free(generalBuf, bufferSize);
  dictPool.free(dictBuf, bufferSize);
  outputPool.free(outputBuf, bufferSize);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208);
  ASSERT_EQ(generalPool.usedBytes(), 262208);
  ASSERT_EQ(generalPool.reservedBytes(), 1048576);
  ASSERT_EQ(dictPool.usedBytes(), 0);
  ASSERT_EQ(dictPool.reservedBytes(), 0);
  ASSERT_EQ(outputPool.usedBytes(), 0);
  ASSERT_EQ(outputPool.reservedBytes(), 0);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208);
  ASSERT_EQ(context.availableMemoryReservation(), 786368);
}

TEST_F(WriterContextTest, memoryBudgetDefault) {
  auto pool = memory::memoryManager()->addRootPool("memoryBudgetDefault");
  WriterContext context{std::make_shared<Config>(), pool};
  ASSERT_EQ(context.getMemoryBudget(), pool->maxCapacity());
}

TEST_F(WriterContextTest, memoryBudgetLessThanPoolCapacity) {
  const int64_t poolCapacity = 1L << 30;
  const int64_t budget = 256L << 20;
  auto pool = memory::memoryManager()->addRootPool(
      "memoryBudgetLessThanPoolCapacity", poolCapacity);
  WriterContext context{
      std::make_shared<Config>(),
      pool,
      dwio::common::MetricsLog::voidLog(),
      nullptr,
      false,
      nullptr,
      budget};
  ASSERT_EQ(context.getMemoryBudget(), budget);
}

TEST_F(WriterContextTest, memoryBudgetGreaterThanPoolCapacity) {
  const int64_t poolCapacity = 256L << 20;
  const int64_t budget = 1L << 30;
  auto pool = memory::memoryManager()->addRootPool(
      "memoryBudgetGreaterThanPoolCapacity", poolCapacity);
  WriterContext context{
      std::make_shared<Config>(),
      pool,
      dwio::common::MetricsLog::voidLog(),
      nullptr,
      false,
      nullptr,
      budget};
  ASSERT_EQ(context.getMemoryBudget(), poolCapacity);
}

TEST_F(WriterContextTest, abort) {
  auto writerRoot = memory::memoryManager()->addRootPool(
      "abort", 1L << 30, exec::MemoryReclaimer::create());
  WriterContext context{std::make_shared<Config>(), writerRoot};
  ASSERT_EQ(context.getTotalMemoryUsage(), 0);
  context.initBuffer();
  // The writer context has some initial memory allocation on construction.
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208);
  ASSERT_EQ(context.availableMemoryReservation(), 786368);

  auto& generalPool = context.getMemoryPool(MemoryUsageCategory::GENERAL);
  auto& dictPool = context.getMemoryPool(MemoryUsageCategory::DICTIONARY);
  auto& outputPool = context.getMemoryPool(MemoryUsageCategory::OUTPUT_STREAM);

  const int bufferSize{1024};
  void* generalBuf = generalPool.allocate(bufferSize);
  void* dictBuf = dictPool.allocate(bufferSize);
  void* outputBuf = outputPool.allocate(bufferSize);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208 + bufferSize * 3);
  ASSERT_EQ(context.availableMemoryReservation(), 2880448);

  ASSERT_EQ(generalPool.usedBytes(), 262208 + bufferSize);
  ASSERT_EQ(generalPool.reservedBytes(), 1048576);
  ASSERT_EQ(dictPool.usedBytes(), bufferSize);
  ASSERT_EQ(dictPool.reservedBytes(), 1048576);
  ASSERT_EQ(outputPool.usedBytes(), bufferSize);
  ASSERT_EQ(outputPool.reservedBytes(), 1048576);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208 + bufferSize * 3);
  ASSERT_EQ(context.availableMemoryReservation(), 2880448);

  ASSERT_TRUE(generalPool.maybeReserve(4L << 20));
  ASSERT_TRUE(dictPool.maybeReserve(4L << 20));
  ASSERT_TRUE(outputPool.maybeReserve(4L << 20));
  ASSERT_EQ(generalPool.usedBytes(), 262208 + bufferSize);
  ASSERT_EQ(generalPool.reservedBytes(), 9437184);
  ASSERT_EQ(dictPool.usedBytes(), bufferSize);
  ASSERT_EQ(dictPool.reservedBytes(), 9437184);
  ASSERT_EQ(outputPool.usedBytes(), bufferSize);
  ASSERT_EQ(outputPool.reservedBytes(), 9437184);
  ASSERT_EQ(context.getTotalMemoryUsage(), 262208 + bufferSize * 3);
  ASSERT_EQ(context.availableMemoryReservation(), 28046272);

  context.abort();

  ASSERT_EQ(context.getTotalMemoryUsage(), bufferSize * 3);
  ASSERT_EQ(generalPool.usedBytes(), bufferSize);
  ASSERT_EQ(generalPool.reservedBytes(), 1048576);
  ASSERT_EQ(dictPool.usedBytes(), bufferSize);
  ASSERT_EQ(dictPool.reservedBytes(), 1048576);
  ASSERT_EQ(outputPool.usedBytes(), bufferSize);
  ASSERT_EQ(outputPool.reservedBytes(), 1048576);
  ASSERT_EQ(context.availableMemoryReservation(), 3142656);

  generalPool.free(generalBuf, bufferSize);
  dictPool.free(dictBuf, bufferSize);
  outputPool.free(outputBuf, bufferSize);
  ASSERT_EQ(context.getTotalMemoryUsage(), 0);
  ASSERT_EQ(generalPool.usedBytes(), 0);
  ASSERT_EQ(generalPool.reservedBytes(), 0);
  ASSERT_EQ(dictPool.usedBytes(), 0);
  ASSERT_EQ(dictPool.reservedBytes(), 0);
  ASSERT_EQ(outputPool.usedBytes(), 0);
  ASSERT_EQ(outputPool.reservedBytes(), 0);
  ASSERT_EQ(context.getTotalMemoryUsage(), 0);
  ASSERT_EQ(context.availableMemoryReservation(), 0);
}
} // namespace facebook::velox::dwrf
