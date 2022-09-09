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

#include "velox/substrait/TypeUtils.h"
#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox;
using namespace facebook::velox::substrait;

namespace facebook::velox::substrait::test {

class TypeUtilsTest : public ::testing::Test {
 protected:
  static void testToSubstraitType(
      const TypePtr& type,
      const std::string& expectedType) {
    SCOPED_TRACE(type->toString());
    auto substraitType = substraitSignature(type);
    ASSERT_EQ(substraitType, expectedType);
  }
};

TEST_F(TypeUtilsTest, basic) {
  testToSubstraitType(BOOLEAN(), "bool");

  testToSubstraitType(TINYINT(), "i8");
  testToSubstraitType(SMALLINT(), "i16");
  testToSubstraitType(INTEGER(), "i32");
  testToSubstraitType(BIGINT(), "i64");

  testToSubstraitType(REAL(), "fp32");
  testToSubstraitType(DOUBLE(), "fp64");

  testToSubstraitType(VARCHAR(), "str");
  testToSubstraitType(VARBINARY(), "vbin");

  testToSubstraitType(TIMESTAMP(), "ts");
  testToSubstraitType(DATE(), "date");

  testToSubstraitType(SHORT_DECIMAL(18, 2), "dec");
  testToSubstraitType(LONG_DECIMAL(18, 2), "dec");

  testToSubstraitType(ARRAY(BOOLEAN()), "list");
  testToSubstraitType(ARRAY(INTEGER()), "list");

  testToSubstraitType(MAP(INTEGER(), BIGINT()), "map");

  testToSubstraitType(ROW({INTEGER(), BIGINT()}), "struct");
  testToSubstraitType(ROW({ARRAY(INTEGER())}), "struct");
  testToSubstraitType(ROW({MAP(INTEGER(), INTEGER())}), "struct");
  testToSubstraitType(ROW({ROW({INTEGER()})}), "struct");

  testToSubstraitType(UNKNOWN(), "u!name");

  ASSERT_ANY_THROW(testToSubstraitType(INTERVAL_DAY_TIME(), "iday"));
}

} // namespace facebook::velox::substrait::test
