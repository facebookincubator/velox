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

#include "gtest/gtest.h"

#include "velox/expression/DecodedArgs.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

class DecodedArgsTest : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  VectorPtr makeAscending() {
    return makeFlatVector<int64_t>(
        kSize, [](vector_size_t row) { return row; });
  }

  static constexpr vector_size_t kSize{10};

  core::ExecCtx execCtx_{pool_.get(), nullptr};
  EvalCtx evalCtx_{&execCtx_};
};

TEST_F(DecodedArgsTest, argumentsMapToTheirBaseVectors) {
  SelectivityVector rows(kSize);
  auto flat = makeAscending();
  auto constant = makeConstant<int64_t>(42, kSize);
  auto dictionaryBase = makeAscending();
  auto dictionary =
      wrapInDictionary(makeIndicesInReverse(kSize), dictionaryBase);
  std::vector<VectorPtr> args = {flat, constant, dictionary};

  DecodedArgs decodedArgs(rows, args, evalCtx_);

  EXPECT_EQ(3, decodedArgs.size());
  EXPECT_EQ(flat.get(), decodedArgs.at(0)->base());
  EXPECT_EQ(constant.get(), decodedArgs.at(1)->base());
  // A dictionary decodes to the vector it wraps.
  EXPECT_EQ(dictionaryBase.get(), decodedArgs.at(2)->base());
}

TEST_F(DecodedArgsTest, iteration) {
  SelectivityVector rows(kSize);
  std::vector<VectorPtr> args = {
      makeAscending(),
      makeConstant<int64_t>(42, kSize),
      makeAscending(),
  };

  DecodedArgs decodedArgs(rows, args, evalCtx_);

  // Iteration visits the same DecodedVectors as at(), in argument order.
  const std::vector<DecodedVector*> iterated(
      decodedArgs.begin(), decodedArgs.end());
  const std::vector<DecodedVector*> expected{
      decodedArgs.at(0),
      decodedArgs.at(1),
      decodedArgs.at(2),
  };
  EXPECT_EQ(expected, iterated);
}

TEST_F(DecodedArgsTest, withoutArguments) {
  SelectivityVector rows(kSize);

  DecodedArgs noArgs(rows, {}, evalCtx_);

  EXPECT_EQ(0, noArgs.size());
  EXPECT_EQ(noArgs.begin(), noArgs.end());
}
