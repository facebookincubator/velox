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

#include "velox/common/memory/ByteStream.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/LazyVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using facebook::velox::exec::ContainerRowSerde;
using facebook::velox::test::TestingLoader;

class ContainerRowSerdeTest : public testing::Test,
                              public test::VectorTestBase {};

TEST_F(ContainerRowSerdeTest, nestedSerde) {
  auto data = makeFlatVector<int32_t>(10, [](auto row) { return row; });
  auto columnType = ROW({"a", "b"}, {INTEGER(), INTEGER()});

  auto loaderA = std::make_unique<TestingLoader>(data);
  auto loaderB = std::make_unique<TestingLoader>(data);
  auto lazyVectorA = std::make_shared<LazyVector>(
      pool_.get(), INTEGER(), 10, std::move(loaderA));
  auto lazyVectorB = std::make_shared<LazyVector>(
      pool_.get(), INTEGER(), 10, std::move(loaderB));
  std::vector<VectorPtr> children{lazyVectorA, lazyVectorB};
  auto rowVector = std::make_shared<RowVector>(
      pool_.get(), columnType, BufferPtr(nullptr), 10, children);

  HashStringAllocator allocator(pool_.get());

  auto begin = allocator.allocate(1024);
  ByteStream stream(&allocator);
  allocator.extendWrite({begin, begin->begin()}, stream);
  ContainerRowSerde::instance().serialize(*rowVector, 0, stream);

  HashStringAllocator::prepareRead(begin, stream);
  VectorPtr result = BaseVector::create(columnType, 1, pool());
  exec::ContainerRowSerde::instance().deserialize(stream, 0, result.get());
  ASSERT_TRUE(result->equalValueAt(rowVector.get(), 0, 0));
}
