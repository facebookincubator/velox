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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/velox/HybridFlatMap.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble {
namespace {

class SerializerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("serializer_test");
    pool_ = rootPool_->addLeafChild("leaf");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(pool_.get());
  }

  static velox::RowTypePtr mapRowType() {
    return velox::ROW(
        {{"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())}});
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

TEST_F(SerializerTest, hybridFlatMapRequiresSerializationFormat) {
  const SerializerOptions options{
      .version = SerializationVersion::kProjection,
      .hybridFlatMapColumns =
          {{"features",
            HybridFlatMap{
                .groups =
                    {{.groupId = 0, .groupKeys = {"1"}},
                     {.groupId = HybridFlatMap::kDefaultGroupId,
                      .groupKeys = {}}}}}},
  };

  NIMBLE_ASSERT_THROW(
      std::make_unique<Serializer>(options, mapRowType(), pool_.get()),
      "Serializer writes must use kSerialization");
}

TEST_F(SerializerTest, validatesHybridFlatMapColumnConfiguration) {
  const auto type = mapRowType();

  NIMBLE_ASSERT_USER_THROW(
      std::make_unique<Serializer>(
          SerializerOptions{
              .version = SerializationVersion::kSerialization,
              .hybridFlatMapColumns =
                  {{"missing",
                    HybridFlatMap{
                        .groups =
                            {{.groupId = 0, .groupKeys = {"1"}},
                             {.groupId = HybridFlatMap::kDefaultGroupId,
                              .groupKeys = {}}}}}},
          },
          type,
          pool_.get()),
      "Hybrid FlatMap column 'missing' does not exist");

  NIMBLE_ASSERT_USER_THROW(
      std::make_unique<Serializer>(
          SerializerOptions{
              .version = SerializationVersion::kSerialization,
              .flatMapColumns = {{"features", {}}},
              .hybridFlatMapColumns =
                  {{"features",
                    HybridFlatMap{
                        .groups =
                            {{.groupId = 0, .groupKeys = {"1"}},
                             {.groupId = HybridFlatMap::kDefaultGroupId,
                              .groupKeys = {}}}}}},
          },
          type,
          pool_.get()),
      "cannot use both FlatMap and hybrid FlatMap options");

  const auto scalarType = velox::ROW({{"features", velox::BIGINT()}});
  NIMBLE_ASSERT_USER_THROW(
      std::make_unique<Serializer>(
          SerializerOptions{
              .version = SerializationVersion::kSerialization,
              .hybridFlatMapColumns =
                  {{"features",
                    HybridFlatMap{
                        .groups =
                            {{.groupId = 0, .groupKeys = {"1"}},
                             {.groupId = HybridFlatMap::kDefaultGroupId,
                              .groupKeys = {}}}}}},
          },
          scalarType,
          pool_.get()),
      "Hybrid FlatMap column 'features' must be a MAP");
}

TEST_F(SerializerTest, hybridFlatMapSerializesDefaultFeature) {
  const SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .hybridFlatMapColumns =
          {{"features",
            HybridFlatMap{
                .groups =
                    {{.groupId = 0, .groupKeys = {"1"}},
                     {.groupId = HybridFlatMap::kDefaultGroupId,
                      .groupKeys = {}}}}}},
  };

  Serializer serializer{options, mapRowType(), pool_.get()};
  const auto input = vectorMaker_->rowVector(
      {"features"},
      {vectorMaker_->mapVector<int32_t, double>({{{1, 1.0}, {99, 99.0}}})});
  EXPECT_NO_THROW(
      serializer.serialize(input, OrderedRanges::of(0, input->size())));
}

} // namespace
} // namespace facebook::nimble
