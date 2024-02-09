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

#include "velox/expression/TypeConverter.h"
#include <gtest/gtest.h>

using namespace ::testing;

namespace facebook::velox::exec::test {
namespace {

void generateAndValidateBasicSignature(
    velox::TypePtr type,
    const std::string& name) {
  auto signature = exec::toTypeSignature(type);
  EXPECT_EQ(signature.baseName(), name);
  EXPECT_EQ(signature.parameters().size(), 0);
  EXPECT_FALSE(signature.rowFieldName().has_value());
}

TEST(TypeConverterTest, physicalTypes) {
  auto integer = INTEGER();
  generateAndValidateBasicSignature(integer, "integer");

  auto timestamp = TIMESTAMP();
  generateAndValidateBasicSignature(timestamp, "timestamp");

  auto doubletype = DOUBLE();
  generateAndValidateBasicSignature(doubletype, "double");

  auto opaque = OPAQUE<std::string>();
  generateAndValidateBasicSignature(opaque, "opaque");
}

TEST(TypeConverterTest, complexTypes) {
  auto type = velox::MAP(INTEGER(), ARRAY(REAL()));
  auto signature = toTypeSignature(type);
  EXPECT_EQ(signature.baseName(), "map");
  EXPECT_EQ(signature.parameters().size(), 2);

  EXPECT_EQ(signature.parameters()[0].baseName(), "integer");

  EXPECT_EQ(signature.parameters()[1].baseName(), "array");
  EXPECT_EQ(signature.parameters()[1].parameters()[0].baseName(), "real");

  auto rowType =
      velox::ROW({"a", "b"}, {MAP(INTEGER(), REAL()), ARRAY(DOUBLE())});
  auto rowSignature = toTypeSignature(rowType);
  ASSERT_EQ(rowSignature.baseName(), "row");
  EXPECT_EQ(rowSignature.parameters().size(), 2);

  EXPECT_EQ(rowSignature.parameters()[0].baseName(), "map");
  EXPECT_EQ(rowSignature.parameters()[1].baseName(), "array");

  EXPECT_EQ(rowSignature.parameters()[0].parameters()[0].baseName(), "integer");
  EXPECT_EQ(rowSignature.parameters()[0].parameters()[1].baseName(), "real");
  EXPECT_EQ(rowSignature.parameters()[1].parameters()[0].baseName(), "double");
}

TEST(TypeConverterTest, logicalTypes) {
  auto decimal = velox::DECIMAL(11, 5);
  generateAndValidateBasicSignature(decimal, "decimal");

  auto date = velox::DATE();
  generateAndValidateBasicSignature(date, "date");

  auto interval = velox::INTERVAL_DAY_TIME();
  generateAndValidateBasicSignature(interval, "interval day to second");

  auto intervalYearToMonth = velox::INTERVAL_YEAR_MONTH();
  generateAndValidateBasicSignature(
      intervalYearToMonth, "interval year to month");
}
} // namespace
} // namespace facebook::velox::exec::test
