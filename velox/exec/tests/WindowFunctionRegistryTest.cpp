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

#include "velox/exec/WindowFunction.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/Type.h"

namespace facebook::velox::exec::test {

namespace {
void registerWindowFunction(const std::string& name) {
  std::vector<exec::FunctionSignaturePtr> signatures{
      exec::FunctionSignatureBuilder()
          .argumentType("BIGINT")
          .argumentType("DOUBLE")
          .returnType("BIGINT")
          .build(),
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .argumentType("T")
          .argumentType("T")
          .returnType("T")
          .build(),
      exec::FunctionSignatureBuilder().returnType("DATE").build(),
  };
  exec::registerWindowFunction(name, std::move(signatures), nullptr);
}
} // namespace

class WindowFunctionRegistryTest : public testing::Test {
 public:
  WindowFunctionRegistryTest() {
    registerWindowFunction("window_func");
    registerWindowFunction("window_Func_Alias");
  }

  void checkEqual(const TypePtr& actual, const TypePtr& expected) {
    if (expected) {
      EXPECT_EQ(*actual, *expected);
    } else {
      EXPECT_EQ(actual, nullptr);
    }
  }

  TypePtr resolveWindowFunction(
      const std::string& functionName,
      const std::vector<TypePtr>& argTypes) {
    if (auto windowFunctionSignatures =
            getWindowFunctionSignatures(functionName)) {
      for (const auto& signature : windowFunctionSignatures.value()) {
        SignatureBinder binder(*signature, argTypes);
        if (binder.tryBind()) {
          return binder.tryResolveReturnType();
        }
      }
    }

    return nullptr;
  }

  void testResolveWindowFunction(
      const std::string& functionName,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& expectedReturn) {
    auto result = resolveWindowFunction(functionName, argTypes);
    checkEqual(result, expectedReturn);
  }
};

TEST_F(WindowFunctionRegistryTest, hasWindowFunctionSignature) {
  testResolveWindowFunction("window_func", {BIGINT(), DOUBLE()}, BIGINT());
  testResolveWindowFunction("window_func", {DOUBLE(), DOUBLE()}, DOUBLE());
  testResolveWindowFunction(
      "window_func", {ARRAY(BOOLEAN()), ARRAY(BOOLEAN())}, ARRAY(BOOLEAN()));
  testResolveWindowFunction("window_func", {}, DATE());
}

TEST_F(WindowFunctionRegistryTest, windowFunctionWrongName) {
  testResolveWindowFunction(
      "window_func_not_exist", {BIGINT(), DOUBLE()}, nullptr);
  testResolveWindowFunction(
      "window_func_not_exist", {DOUBLE(), DOUBLE()}, nullptr);
  testResolveWindowFunction(
      "window_func_not_exist", {ARRAY(BOOLEAN()), ARRAY(BOOLEAN())}, nullptr);
  testResolveWindowFunction("window_func_not_exist", {}, nullptr);
}

TEST_F(WindowFunctionRegistryTest, windowFunctionSignatureWrongArgType) {
  testResolveWindowFunction("window_func", {DOUBLE(), BIGINT()}, nullptr);
  testResolveWindowFunction("window_func", {BIGINT()}, nullptr);
  testResolveWindowFunction(
      "window_func", {BIGINT(), BIGINT(), BIGINT()}, nullptr);
}

TEST_F(WindowFunctionRegistryTest, windowFunctionNameMixedCase) {
  testResolveWindowFunction("window_FUNC", {BIGINT(), DOUBLE()}, BIGINT());
  testResolveWindowFunction(
      "window_fUNC_alias", {BIGINT(), DOUBLE()}, BIGINT());
}

} // namespace facebook::velox::exec::test
