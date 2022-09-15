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

#include "velox/substrait/SubstraitExtensionCollector.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::substrait;

namespace facebook::velox::substrait::test {

class SubstraitExtensionCollectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Test::SetUp();
    functions::prestosql::registerAllScalarFunctions();
  }
  int getReferenceNumber(
      const std::string& functionName,
      std::vector<TypePtr>&& arguments) {
    int functionReferenceId1 =
        extensionCollector_->getReferenceNumber(functionName, arguments);
    // Repeat the call to make sure properly de-duplicated.
    int functionReferenceId2 =
        extensionCollector_->getReferenceNumber(functionName, arguments);
    EXPECT_EQ(functionReferenceId1, functionReferenceId2);
    return functionReferenceId2;
  }

  SubstraitExtensionCollectorPtr extensionCollector_ =
      std::make_shared<SubstraitExtensionCollector>();
};

TEST_F(SubstraitExtensionCollectorTest, getReferenceNumber) {
  ASSERT_EQ(getReferenceNumber("plus", {INTEGER(), INTEGER()}), 0);
  ASSERT_EQ(getReferenceNumber("divide", {INTEGER(), INTEGER()}), 1);
  ASSERT_EQ(getReferenceNumber("cardinality", {ARRAY(INTEGER())}), 2);
  ASSERT_EQ(getReferenceNumber("array_sum", {ARRAY(INTEGER())}), 3);
  ASSERT_EQ(getReferenceNumber("sum", {INTEGER()}), 4);
  ASSERT_EQ(getReferenceNumber("avg", {INTEGER()}), 5);
  ASSERT_EQ(getReferenceNumber("avg", {ROW({DOUBLE(), BIGINT()})}), 6);
  ASSERT_EQ(getReferenceNumber("count", {INTEGER()}), 7);

  auto functionType = std::make_shared<const FunctionType>(
      std::vector<TypePtr>{INTEGER(), VARCHAR()}, BIGINT());
  std::vector<TypePtr> types = {MAP(INTEGER(), VARCHAR()), functionType};
  ASSERT_ANY_THROW(getReferenceNumber("transform_keys", std::move(types)));
}

TEST_F(SubstraitExtensionCollectorTest, addExtensionsToPlan) {
  // Arrange
  getReferenceNumber("plus", {INTEGER(), INTEGER()});
  getReferenceNumber("divide", {INTEGER(), INTEGER()});
  getReferenceNumber("cardinality", {ARRAY(INTEGER())});
  getReferenceNumber("array_sum", {ARRAY(INTEGER())});
  getReferenceNumber("sum", {INTEGER()});
  getReferenceNumber("avg", {INTEGER()});
  getReferenceNumber("avg", {ROW({DOUBLE(), BIGINT()})});
  getReferenceNumber("count", {INTEGER()});

  google::protobuf::Arena arena;
  auto* substraitPlan =
      google::protobuf::Arena::CreateMessage<::substrait::Plan>(&arena);

  // Act
  extensionCollector_->addExtensionsToPlan(substraitPlan);

  // Assert
  ASSERT_EQ(substraitPlan->extensions().size(), 8);
  ASSERT_EQ(
      substraitPlan->extensions(0).extension_function().name(), "plus:i32_i32");
  ASSERT_EQ(
      substraitPlan->extensions(1).extension_function().name(),
      "divide:i32_i32");
  ASSERT_EQ(
      substraitPlan->extensions(2).extension_function().name(),
      "cardinality:list");
  ASSERT_EQ(
      substraitPlan->extensions(3).extension_function().name(),
      "array_sum:list");
  ASSERT_EQ(
      substraitPlan->extensions(4).extension_function().name(), "sum:i32");
  ASSERT_EQ(
      substraitPlan->extensions(5).extension_function().name(), "avg:i32");
  ASSERT_EQ(
      substraitPlan->extensions(6).extension_function().name(), "avg:i16");
  ASSERT_EQ(
      substraitPlan->extensions(7).extension_function().name(), "count:any");
}

} // namespace facebook::velox::substrait::test
