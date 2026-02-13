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
#include "velox/core/Expressions.h"
#include "velox/type/Variant.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::core::test {

/// Tests for expression hash consistency.
///
/// These tests verify that hash functions are deterministic and stable:
/// 1. Same expression hashed multiple times produces same result
/// 2. Semantically equivalent expressions have same hash
/// 3. Hash survives serialization roundtrip
/// 4. Different expressions produce different hashes
///
/// Note: We do NOT use hardcoded expected hash values as that makes tests
/// brittle. Instead we test hash properties.
class TypedExprHashConsistencyTest : public ::testing::Test,
                                     public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    Type::registerSerDe();
    ITypedExpr::registerSerDe();
  }

  std::shared_ptr<ConstantTypedExpr> makeConstantExpr(
      const TypePtr& type,
      const Variant& value) {
    return std::make_shared<ConstantTypedExpr>(type, value);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

// Test that hashing the same expression multiple times gives same result
TEST_F(TypedExprHashConsistencyTest, idempotency) {
  // InputTypedExpr
  auto inputExpr = std::make_shared<InputTypedExpr>(INTEGER());
  EXPECT_EQ(inputExpr->hash(), inputExpr->hash());

  // ConstantTypedExpr with various types
  auto boolExpr = makeConstantExpr(BOOLEAN(), variant(true));
  EXPECT_EQ(boolExpr->hash(), boolExpr->hash());

  auto intExpr = makeConstantExpr(INTEGER(), variant(int32_t(42)));
  EXPECT_EQ(intExpr->hash(), intExpr->hash());

  auto strExpr = makeConstantExpr(VARCHAR(), variant("hello"));
  EXPECT_EQ(strExpr->hash(), strExpr->hash());

  // CallTypedExpr
  auto callExpr = std::make_shared<CallTypedExpr>(
      BIGINT(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(BIGINT(), "a")},
      "plus");
  EXPECT_EQ(callExpr->hash(), callExpr->hash());

  // FieldAccessTypedExpr
  auto fieldExpr = std::make_shared<FieldAccessTypedExpr>(BIGINT(), "column");
  EXPECT_EQ(fieldExpr->hash(), fieldExpr->hash());

  // CastTypedExpr
  auto castExpr = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x")},
      false);
  EXPECT_EQ(castExpr->hash(), castExpr->hash());
}

// Test that equivalent expressions created separately have same hash
TEST_F(TypedExprHashConsistencyTest, equivalentExpressions) {
  // Create two equivalent InputTypedExpr
  auto input1 = std::make_shared<InputTypedExpr>(INTEGER());
  auto input2 = std::make_shared<InputTypedExpr>(INTEGER());
  EXPECT_EQ(input1->hash(), input2->hash());

  // Create two equivalent ConstantTypedExpr
  auto const1 = makeConstantExpr(INTEGER(), variant(int32_t(42)));
  auto const2 = makeConstantExpr(INTEGER(), variant(int32_t(42)));
  EXPECT_EQ(const1->hash(), const2->hash());

  // Create two equivalent CallTypedExpr
  auto call1 = std::make_shared<CallTypedExpr>(
      BIGINT(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(BIGINT(), "x")},
      "negate");
  auto call2 = std::make_shared<CallTypedExpr>(
      BIGINT(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(BIGINT(), "x")},
      "negate");
  EXPECT_EQ(call1->hash(), call2->hash());

  // Create two equivalent FieldAccessTypedExpr
  auto field1 = std::make_shared<FieldAccessTypedExpr>(VARCHAR(), "name");
  auto field2 = std::make_shared<FieldAccessTypedExpr>(VARCHAR(), "name");
  EXPECT_EQ(field1->hash(), field2->hash());

  // Create two equivalent CastTypedExpr (cast)
  auto cast1 = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "y")},
      false);
  auto cast2 = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "y")},
      false);
  EXPECT_EQ(cast1->hash(), cast2->hash());

  // Create two equivalent CastTypedExpr (try_cast)
  auto tryCast1 = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "z")},
      true);
  auto tryCast2 = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "z")},
      true);
  EXPECT_EQ(tryCast1->hash(), tryCast2->hash());
}

// Test that hash survives serialization roundtrip
TEST_F(TypedExprHashConsistencyTest, serializationRoundtrip) {
  auto testRoundtrip = [this](const TypedExprPtr& expr) {
    auto originalHash = expr->hash();

    // Serialize
    auto serialized = expr->serialize();

    // Deserialize
    auto deserialized =
        ISerializable::deserialize<ITypedExpr>(serialized, pool_.get());

    // Hash should be the same
    EXPECT_EQ(originalHash, deserialized->hash())
        << "Hash changed after serialization roundtrip for: "
        << expr->toString();
  };

  // Test various expression types
  testRoundtrip(std::make_shared<InputTypedExpr>(INTEGER()));
  testRoundtrip(makeConstantExpr(INTEGER(), variant(int32_t(42))));
  testRoundtrip(makeConstantExpr(VARCHAR(), variant("test")));
  testRoundtrip(std::make_shared<FieldAccessTypedExpr>(BIGINT(), "col"));
  testRoundtrip(
      std::make_shared<CallTypedExpr>(
          BIGINT(),
          std::vector<TypedExprPtr>{
              std::make_shared<FieldAccessTypedExpr>(BIGINT(), "a")},
          "abs"));
  testRoundtrip(
      std::make_shared<CastTypedExpr>(
          VARCHAR(),
          std::vector<TypedExprPtr>{
              std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x")},
          false));
}

// Test that different expressions produce different hashes
TEST_F(TypedExprHashConsistencyTest, distinctness) {
  // Different constant values should have different hashes
  auto int1 = makeConstantExpr(INTEGER(), variant(int32_t(1)));
  auto int2 = makeConstantExpr(INTEGER(), variant(int32_t(2)));
  EXPECT_NE(int1->hash(), int2->hash());

  // Different types should have different hashes
  auto intExpr = makeConstantExpr(INTEGER(), variant(int32_t(42)));
  auto bigintExpr = makeConstantExpr(BIGINT(), variant(int64_t(42)));
  EXPECT_NE(intExpr->hash(), bigintExpr->hash());

  // Different field names should have different hashes
  auto field1 = std::make_shared<FieldAccessTypedExpr>(INTEGER(), "a");
  auto field2 = std::make_shared<FieldAccessTypedExpr>(INTEGER(), "b");
  EXPECT_NE(field1->hash(), field2->hash());

  // Different function names should have different hashes
  auto call1 = std::make_shared<CallTypedExpr>(
      BIGINT(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(BIGINT(), "x")},
      "abs");
  auto call2 = std::make_shared<CallTypedExpr>(
      BIGINT(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(BIGINT(), "x")},
      "negate");
  EXPECT_NE(call1->hash(), call2->hash());

  // cast vs try_cast should have different hashes
  auto cast = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x")},
      false);
  auto tryCast = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x")},
      true);
  EXPECT_NE(cast->hash(), tryCast->hash());
}

// Test complex nested expressions
TEST_F(TypedExprHashConsistencyTest, nestedExpressions) {
  // Create a complex nested expression: cast(plus(a, b) as varchar)
  auto fieldA = std::make_shared<FieldAccessTypedExpr>(BIGINT(), "a");
  auto fieldB = std::make_shared<FieldAccessTypedExpr>(BIGINT(), "b");
  auto plusExpr = std::make_shared<CallTypedExpr>(
      BIGINT(), std::vector<TypedExprPtr>{fieldA, fieldB}, "plus");
  auto castExpr = std::make_shared<CastTypedExpr>(
      VARCHAR(), std::vector<TypedExprPtr>{plusExpr}, false);

  // Create an equivalent expression
  auto fieldA2 = std::make_shared<FieldAccessTypedExpr>(BIGINT(), "a");
  auto fieldB2 = std::make_shared<FieldAccessTypedExpr>(BIGINT(), "b");
  auto plusExpr2 = std::make_shared<CallTypedExpr>(
      BIGINT(), std::vector<TypedExprPtr>{fieldA2, fieldB2}, "plus");
  auto castExpr2 = std::make_shared<CastTypedExpr>(
      VARCHAR(), std::vector<TypedExprPtr>{plusExpr2}, false);

  // Equivalent nested expressions should have same hash
  EXPECT_EQ(castExpr->hash(), castExpr2->hash());

  // Idempotency
  EXPECT_EQ(castExpr->hash(), castExpr->hash());
}

// Test lambda expressions
TEST_F(TypedExprHashConsistencyTest, lambdaExpressions) {
  auto signature = ROW({"x"}, {INTEGER()});
  auto body = std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x");
  auto lambda1 = std::make_shared<LambdaTypedExpr>(signature, body);
  auto lambda2 = std::make_shared<LambdaTypedExpr>(signature, body);

  EXPECT_EQ(lambda1->hash(), lambda2->hash());
  EXPECT_EQ(lambda1->hash(), lambda1->hash());
}

// Test concat expressions
TEST_F(TypedExprHashConsistencyTest, concatExpressions) {
  auto expr1 = std::make_shared<ConcatTypedExpr>(
      std::vector<std::string>{"a", "b"},
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x"),
          std::make_shared<FieldAccessTypedExpr>(VARCHAR(), "y")});
  auto expr2 = std::make_shared<ConcatTypedExpr>(
      std::vector<std::string>{"a", "b"},
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x"),
          std::make_shared<FieldAccessTypedExpr>(VARCHAR(), "y")});

  EXPECT_EQ(expr1->hash(), expr2->hash());
  EXPECT_EQ(expr1->hash(), expr1->hash());
}

} // namespace facebook::velox::core::test
