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

#include "velox/expression/FunctionSignature.h"
#include "velox/type/Type.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {
std::string roundTrip(const std::string& typeSignature) {
  return parseTypeSignature(typeSignature).toString();
}

void testScalarType(const std::string& typeSignature) {
  auto signature = parseTypeSignature(typeSignature);
  ASSERT_EQ(signature.baseName(), typeSignature);
  ASSERT_EQ(signature.parameters().size(), 0);
}
} // namespace

TEST(ParseTypeSignatureTest, scalar) {
  testScalarType("tinyint");
  testScalarType("smallint");
  testScalarType("integer");
  testScalarType("bigint");
  testScalarType("real");
  testScalarType("double");
  testScalarType("timestamp");
}

TEST(ParseTypeSignatureTest, array) {
  auto signature = parseTypeSignature("array(bigint)");
  ASSERT_EQ(signature.baseName(), "array");
  ASSERT_EQ(signature.parameters().size(), 1);

  auto param = signature.parameters()[0];
  ASSERT_EQ(param.baseName(), "bigint");
  ASSERT_EQ(param.parameters().size(), 0);
}

TEST(ParseTypeSignatureTest, map) {
  auto signature = parseTypeSignature("map(bigint, double)");
  ASSERT_EQ(signature.baseName(), "map");
  ASSERT_EQ(signature.parameters().size(), 2);

  auto key = signature.parameters()[0];
  ASSERT_EQ(key.baseName(), "bigint");
  ASSERT_EQ(key.parameters().size(), 0);

  auto value = signature.parameters()[1];
  ASSERT_EQ(value.baseName(), "double");
  ASSERT_EQ(value.parameters().size(), 0);
}

TEST(ParseTypeSignatureTest, roundTrip) {
  ASSERT_EQ(roundTrip("bigint"), "bigint");

  ASSERT_EQ(roundTrip("array(T)"), "array(T)");
  ASSERT_EQ(roundTrip("array(array(T))"), "array(array(T))");
  ASSERT_EQ(roundTrip("array(row(K,V))"), "array(row(K,V))");

  ASSERT_EQ(roundTrip("map(K,V)"), "map(K,V)");

  ASSERT_EQ(roundTrip("function(S,R)"), "function(S,R)");

  // Test a complex type as the second field in a row
  ASSERT_EQ(
      roundTrip("row(map(K,V),map(bigint,array(double)))"),
      "row(map(K,V),map(bigint,array(double)))");
}

TEST(ParseTypeSignatureTest, invalidSignatures) {
  EXPECT_THROW(parseTypeSignature("array(varchar"), VeloxRuntimeError);
  EXPECT_THROW(parseTypeSignature("array(array(T)"), VeloxRuntimeError);
}

TEST(ParseTypeSignatureTest, toVeloxType) {
  ASSERT_NE(INTEGER(), signatureToVeloxType(parseTypeSignature("bigint")));
  ASSERT_EQ(TINYINT(), signatureToVeloxType(parseTypeSignature("tinyint")));
  ASSERT_EQ(SMALLINT(), signatureToVeloxType(parseTypeSignature("smallint")));
  ASSERT_EQ(INTEGER(), signatureToVeloxType(parseTypeSignature("integer")));
  ASSERT_EQ(BIGINT(), signatureToVeloxType(parseTypeSignature("bigint")));
  ASSERT_EQ(DOUBLE(), signatureToVeloxType(parseTypeSignature("double")));
  ASSERT_EQ(REAL(), signatureToVeloxType(parseTypeSignature("float")));
  ASSERT_TRUE(DECIMAL(32, 8)->equivalent(
      *signatureToVeloxType(parseTypeSignature("decimal(32, 8)"))));
  ASSERT_TRUE(MAP(BIGINT(), DOUBLE())
                  ->equivalent(*signatureToVeloxType(
                      parseTypeSignature("map(bigint, double)"))));
  ASSERT_TRUE(MAP(BIGINT(), ARRAY(DOUBLE()))
                  ->equivalent(*signatureToVeloxType(
                      parseTypeSignature("map(bigint, array(double))"))));
  ASSERT_TRUE(ROW({BIGINT(), MAP(BIGINT(), ARRAY(DOUBLE()))})
                  ->equivalent(*signatureToVeloxType(parseTypeSignature(
                      "row(bigint,map(bigint, array(double)))"))));
}
