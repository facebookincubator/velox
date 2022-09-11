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

using namespace facebook::velox;
using namespace facebook::velox::substrait;

namespace facebook::velox::substrait::test {

class SubstraitExtensionCollectorTest : public ::testing::Test {
 protected:
  void assertGetFunctionReference(
      const std::string& functionName,
      const std::vector<TypePtr>& arguments,
      const int& expectedReferenceId) {
    int actualReferenceId =
        extensionCollector_->getFunctionReference(functionName, arguments);
    ASSERT_EQ(actualReferenceId, expectedReferenceId);
  }

  SubstraitExtensionCollectorPtr extensionCollector_ =
      std::make_shared<SubstraitExtensionCollector>();
};

TEST_F(SubstraitExtensionCollectorTest, getFunctionReferenceTest) {
  assertGetFunctionReference("plus", {INTEGER(), INTEGER()}, 0);
  assertGetFunctionReference("plus", {INTEGER(), INTEGER()}, 0);
  assertGetFunctionReference("divide", {INTEGER(), INTEGER()}, 1);
}

TEST_F(SubstraitExtensionCollectorTest, addExtensionsToPlanTest) {
  extensionCollector_->getFunctionReference("plus", {INTEGER(), INTEGER()});
  google::protobuf::Arena arena;
  auto* substraitPlan =
      google::protobuf::Arena::CreateMessage<::substrait::Plan>(&arena);
  extensionCollector_->addExtensionsToPlan(substraitPlan);

  ASSERT_EQ(substraitPlan->extensions().size(), 1);
  ASSERT_EQ(
      substraitPlan->extensions(0).extension_function().name(), "plus:i32_i32");
}

} // namespace facebook::velox::substrait::test
