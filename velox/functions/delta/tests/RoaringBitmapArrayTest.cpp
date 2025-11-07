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

#include "velox/functions/delta/RoaringBitmapArray.h"
#include "velox/core/Expressions.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::functions::delta::test {

namespace {

class RoaringBitmapArrayTest : public functions::test::FunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    registerFunction<RoaringBitmapArrayContains, bool, Varbinary, int64_t>(
        {"bitmap_array_contains"});
  }

  void testBitmapContain(
      const std::string& serialized,
      const VectorPtr& value,
      const VectorPtr& expected) {
    std::vector<core::TypedExprPtr> args;
    args.push_back(
        std::make_shared<core::ConstantTypedExpr>(
            VARBINARY(), variant::binary(serialized)));
    args.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0"));
    auto expr = exec::ExprSet(
        {std::make_shared<core::CallTypedExpr>(
            BOOLEAN(), args, "bitmap_array_contains")},
        &execCtx_);
    auto data = makeRowVector({value});
    exec::EvalCtx evalCtx(&execCtx_, &expr, data.get());
    std::vector<VectorPtr> results(1);
    auto allSelected = SelectivityVector(value->size());
    expr.eval(allSelected, evalCtx, results);
    velox::test::assertEqualVectors(expected, results[0]);
  }
};

TEST_F(RoaringBitmapArrayTest, contains) {
  RoaringBitmapArray array{};
  array.addSafe(206LL);
  array.addSafe(10LL << 32 | 10LL);
  EXPECT_TRUE(array.containsSafe(206LL));
  EXPECT_FALSE(array.containsSafe(207LL));
  EXPECT_TRUE(array.containsSafe(10LL << 32 | 10LL));
  EXPECT_FALSE(array.containsSafe(11LL << 32 | 10LL));
  EXPECT_FALSE(array.containsSafe(10LL << 32 | 11LL));
}

TEST_F(RoaringBitmapArrayTest, serde) {
  RoaringBitmapArray array{};
  array.addSafe(206LL);
  array.addSafe(10LL << 32 | 10LL);
  std::string data;
  data.resize(array.serializedSizeInBytes());
  array.serialize(data.data());
  RoaringBitmapArray deserialized{};
  deserialized.deserialize(data.data());
  EXPECT_TRUE(deserialized.containsSafe(206LL));
  EXPECT_FALSE(deserialized.containsSafe(207LL));
  EXPECT_TRUE(deserialized.containsSafe(10LL << 32 | 10LL));
  EXPECT_FALSE(deserialized.containsSafe(11LL << 32 | 10LL));
  EXPECT_FALSE(deserialized.containsSafe(10LL << 32 | 11LL));
}

TEST_F(RoaringBitmapArrayTest, bitmapContainsFunction) {
  RoaringBitmapArray array{};
  array.addSafe(206LL);
  array.addSafe(10LL << 32 | 10LL);
  std::string data;
  data.resize(array.serializedSizeInBytes());
  array.serialize(data.data());
  auto value = makeFlatVector<int64_t>(std::vector<int64_t>{
      0,
      206LL,
      207LL,
      10LL << 32 | 10LL,
      11LL << 32 | 10LL,
      10LL << 32 | 11LL});
  auto expected = makeFlatVector<bool>(
      std::vector<bool>{false, true, false, true, false, false});
  testBitmapContain(data, value, expected);
}

} // namespace
} // namespace facebook::velox::functions::delta::test
