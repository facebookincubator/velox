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

#include "velox/common/base/tests/GTestUtils.h"

#include "velox/substrait/SubstraitFunctionLookup.h"
#include "velox/substrait/VeloxToSubstraitMappings.h"

using namespace facebook::velox;
using namespace facebook::velox::substrait;

class SubstraitFunctionLookupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    extension = SubstraitExtension::loadExtension();
    mappings = std::make_shared<const VeloxToSubstraitFunctionMappings>();
    scalarFunctionLookup =
        std::make_shared<SubstraitScalarFunctionLookup>(extension, mappings);
  }

 public:
  SubstraitExtensionPtr extension;
  SubstraitFunctionMappingsPtr mappings;
  SubstraitScalarFunctionLookupPtr scalarFunctionLookup;
};

TEST_F(SubstraitFunctionLookupTest, lt_Any_Any) {
  auto functionOption = scalarFunctionLookup->lookupFunction(
      "lt",
      {
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kBool),
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kBool),
      });
  // it should match with any type
  ASSERT_TRUE(functionOption.has_value());
  ASSERT_TRUE(functionOption.value()->name == "lt");
  ASSERT_EQ(functionOption.value()->anchor().key, "lt:any_any");
}

TEST_F(SubstraitFunctionLookupTest, add_i8_i8) {
  auto functionOption = scalarFunctionLookup->lookupFunction(
      "add",
      {
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8),
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8),
      });
  // it should match with I8 type
  ASSERT_TRUE(functionOption.has_value());
  ASSERT_EQ(functionOption.value()->anchor().key, "add:opt_i8_i8");
}

TEST_F(SubstraitFunctionLookupTest, plus_i8_i8) {
  auto functionOption = scalarFunctionLookup->lookupFunction(
      "plus",
      {SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8),
       SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8)});
  // it should match with I8 type
  ASSERT_TRUE(functionOption.has_value());
  ASSERT_EQ(functionOption.value()->anchor().key, "add:opt_i8_i8");
}

TEST_F(SubstraitFunctionLookupTest, plus_i8_i8_i8) {
  auto functionOption = scalarFunctionLookup->lookupFunction(
      "plus",
      {
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8),
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8),
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8),
      });
  // it should match with I8 type
  ASSERT_FALSE(functionOption.has_value());
}

TEST_F(SubstraitFunctionLookupTest, add_i8) {
  auto functionOption = scalarFunctionLookup->lookupFunction(
      "add",
      {
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8),
      });
  // it should match with I8 type
  ASSERT_FALSE(functionOption.has_value());
}

TEST_F(SubstraitFunctionLookupTest, devide_fp32_fp32_with_rounding) {
  auto functionOption = scalarFunctionLookup->lookupFunction(
      "divide",
      {
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kFp32),
          SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kFp32),
      });
  // it should match with I8 type
  ASSERT_TRUE(functionOption.has_value());
  ASSERT_EQ(functionOption.value()->anchor().key, "divide:opt_opt_fp32_fp32");
}
