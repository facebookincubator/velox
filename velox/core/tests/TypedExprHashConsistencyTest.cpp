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
#include "velox/core/Expressions.h"
#include "velox/type/Variant.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::core::test {

/// This test verifies that hash() functions of ITypedExpr classes are
/// deterministic across different processes and machines.
///
/// For cross-process/cross-machine consistency, the hash function must:
/// 1. Be deterministic (same input -> same output)
/// 2. Not depend on memory addresses or pointers
/// 3. Not depend on process-specific state (PIDs, random seeds, etc.)
///
/// These tests use hardcoded expected hash values to verify that the hash
/// values remain stable across different processes and machines. If these
/// tests fail, it indicates that the hash values have changed, which may
/// break systems that rely on hash consistency (e.g., distributed caching,
/// query plan comparison across nodes).
class TypedExprHashConsistencyTest : public ::testing::Test,
                                     public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
  }

  // Helper to create a ConstantTypedExpr from a variant
  std::shared_ptr<ConstantTypedExpr> makeConstantExpr(
      const TypePtr& type,
      const Variant& value) {
    return std::make_shared<ConstantTypedExpr>(type, value);
  }

  // Helper to create a null ConstantTypedExpr
  std::shared_ptr<ConstantTypedExpr> makeNullConstantExpr(const TypePtr& type) {
    return std::make_shared<ConstantTypedExpr>(
        type, variant::null(type->kind()));
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(TypedExprHashConsistencyTest, inputTypedExpr) {
  auto expr = std::make_shared<InputTypedExpr>(INTEGER());
  EXPECT_EQ(expr->hash(), 4049903845529765328ULL)
      << "InputTypedExpr hash changed. Actual: " << expr->hash();

  auto expr2 = std::make_shared<InputTypedExpr>(VARCHAR());
  EXPECT_EQ(expr2->hash(), 4643169249201813153ULL)
      << "InputTypedExpr(VARCHAR) hash changed. Actual: " << expr2->hash();
}

TEST_F(TypedExprHashConsistencyTest, constantTypedExpr) {
  // Boolean
  auto boolExpr = makeConstantExpr(BOOLEAN(), variant(true));
  EXPECT_EQ(boolExpr->hash(), 8856849273132034487ULL)
      << "BOOLEAN(true) hash changed. Actual: " << boolExpr->hash();

  // Integer
  auto intExpr = makeConstantExpr(INTEGER(), variant(int32_t(42)));
  EXPECT_EQ(intExpr->hash(), 18252912680933339002ULL)
      << "INTEGER(42) hash changed. Actual: " << intExpr->hash();

  // Null BIGINT
  auto nullExpr = makeNullConstantExpr(BIGINT());
  EXPECT_EQ(nullExpr->hash(), 13436946328467595667ULL)
      << "BIGINT(null) hash changed. Actual: " << nullExpr->hash();

  // String
  auto strExpr = makeConstantExpr(VARCHAR(), variant("hello"));
  EXPECT_EQ(strExpr->hash(), 14416578975068601291ULL)
      << "VARCHAR(hello) hash changed. Actual: " << strExpr->hash();

  // Double
  auto doubleExpr = makeConstantExpr(DOUBLE(), variant(3.14159));
  EXPECT_EQ(doubleExpr->hash(), 12939170893996246636ULL)
      << "DOUBLE hash changed. Actual: " << doubleExpr->hash();

  // Array
  auto arrayExpr =
      makeConstantExpr(ARRAY(INTEGER()), Variant::array({1, 2, 3}));
  EXPECT_EQ(arrayExpr->hash(), 9490324506690155522ULL)
      << "ARRAY hash changed. Actual: " << arrayExpr->hash();

  // Row
  auto rowType = ROW({{"a", INTEGER()}, {"b", VARCHAR()}});
  auto rowExpr = makeConstantExpr(rowType, Variant::row({42, "hello"}));
  EXPECT_EQ(rowExpr->hash(), 3198785565592166050ULL)
      << "ROW hash changed. Actual: " << rowExpr->hash();
}

TEST_F(TypedExprHashConsistencyTest, callTypedExpr) {
  auto expr = std::make_shared<CallTypedExpr>(
      BIGINT(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(BIGINT(), "a")},
      "plus");
  EXPECT_EQ(expr->hash(), 9673001606852849486ULL)
      << "CallTypedExpr hash changed. Actual: " << expr->hash();
}

TEST_F(TypedExprHashConsistencyTest, fieldAccessTypedExpr) {
  auto expr = std::make_shared<FieldAccessTypedExpr>(BIGINT(), "column_a");
  EXPECT_EQ(expr->hash(), 11777945155662407419ULL)
      << "FieldAccessTypedExpr hash changed. Actual: " << expr->hash();
}

TEST_F(TypedExprHashConsistencyTest, dereferenceTypedExpr) {
  auto expr = std::make_shared<DereferenceTypedExpr>(
      VARCHAR(),
      std::make_shared<FieldAccessTypedExpr>(
          ROW({"a", "b"}, {VARCHAR(), BOOLEAN()}), "ab"),
      0);
  EXPECT_EQ(expr->hash(), 13224256824664123067ULL)
      << "DereferenceTypedExpr hash changed. Actual: " << expr->hash();
}

TEST_F(TypedExprHashConsistencyTest, castTypedExpr) {
  // CAST
  auto expr = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x")},
      false);
  EXPECT_EQ(expr->hash(), 10329268476666944746ULL)
      << "CastTypedExpr hash changed. Actual: " << expr->hash();

  // TRY_CAST
  auto tryCastExpr = std::make_shared<CastTypedExpr>(
      VARCHAR(),
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x")},
      true);
  EXPECT_EQ(tryCastExpr->hash(), 3445540181233255608ULL)
      << "CastTypedExpr(try) hash changed. Actual: " << tryCastExpr->hash();
}

TEST_F(TypedExprHashConsistencyTest, concatTypedExpr) {
  auto expr = std::make_shared<ConcatTypedExpr>(
      std::vector<std::string>{"a", "b"},
      std::vector<TypedExprPtr>{
          std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x"),
          std::make_shared<FieldAccessTypedExpr>(VARCHAR(), "y")});
  EXPECT_EQ(expr->hash(), 17944415127129017832ULL)
      << "ConcatTypedExpr hash changed. Actual: " << expr->hash();
}

TEST_F(TypedExprHashConsistencyTest, lambdaTypedExpr) {
  auto signature = ROW({"x"}, {INTEGER()});
  auto body = std::make_shared<FieldAccessTypedExpr>(INTEGER(), "x");
  auto expr = std::make_shared<LambdaTypedExpr>(signature, body);
  EXPECT_EQ(expr->hash(), 9214329465187572972ULL)
      << "LambdaTypedExpr hash changed. Actual: " << expr->hash();
}

} // namespace facebook::velox::core::test
