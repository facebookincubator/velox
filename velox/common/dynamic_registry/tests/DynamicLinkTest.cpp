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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/dynamic_registry/DynamicLibraryLoader.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h" 

namespace facebook::velox::functions::test {

class DynamicLinkTest : public FunctionBaseTest {
};

std::string getLibraryPath(std::string filename){
  return fmt::format("{}/{}{}",VELOX_TEST_DYNAMIC_LIBRARY_PATH, filename, VELOX_TEST_DYNAMIC_LIBRARY_PATH_SUFFIX);
}

TEST_F(DynamicLinkTest, dynamicLoadFunc) {
  const auto dynamicFunction = [&](std::optional<double> a) {
    return evaluateOnce<int64_t>("dynamic_1()", a);
  };

  auto signaturesBefore = getFunctionSignatures().size();

  VELOX_ASSERT_THROW(
      dynamicFunction(0), "Scalar function doesn't exist: dynamic_1.");

  std::string libraryPath = getLibraryPath("libvelox_function_my_dynamic");
  loadDynamicLibrary(libraryPath.data());
  auto signaturesAfter = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfter, signaturesBefore + 1);
  EXPECT_EQ(123, dynamicFunction(0));

  auto& registry = exec::simpleFunctions();
  auto resolved = registry.resolveFunction("dynamic_1", {});
  EXPECT_EQ(TypeKind::BIGINT, resolved->type()->kind());
}

TEST_F(DynamicLinkTest, dynamicLoadSameFuncTwice) {
  const auto dynamicFunction = [&](std::optional<double> a) {
    return evaluateOnce<int64_t>("dynamic_2()", a);
  };
  auto& registry = exec::simpleFunctions();
  auto signaturesBefore = getFunctionSignatures().size();

  VELOX_ASSERT_THROW(
      dynamicFunction(0), "Scalar function doesn't exist: dynamic_2.");
  
  std::string libraryPath = getLibraryPath("libvelox_function_same_twice_my_dynamic");
  loadDynamicLibrary(libraryPath.data());
  auto signaturesAfterFirst = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfterFirst, signaturesBefore + 1);
  EXPECT_EQ(123, dynamicFunction(0));
  auto resolvedAfterFirst = registry.resolveFunction("dynamic_2", {});
  EXPECT_EQ(TypeKind::BIGINT, resolvedAfterFirst->type()->kind());

  loadDynamicLibrary(libraryPath.data());
  auto signaturesAfterSecond = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfterSecond, signaturesAfterFirst);
  auto resolvedAfterSecond = registry.resolveFunction("dynamic_2", {});
  EXPECT_EQ(TypeKind::BIGINT, resolvedAfterSecond->type()->kind());
}

TEST_F(DynamicLinkTest, dynamicLoadTwoOfTheSameName) {
  const auto dynamicFunctionInt = [&](std::optional<double> a) {
    return evaluateOnce<int64_t>("dynamic_3()", a);
  };
  const auto dynamicFunctionStr = [&](std::optional<std::string> a) {
    return evaluateOnce<std::string>("dynamic_3()", a);
  };

  auto& registry = exec::simpleFunctions();
  auto signaturesBefore = getFunctionSignatures().size();

  VELOX_ASSERT_THROW(
      dynamicFunctionStr("0"), "Scalar function doesn't exist: dynamic_3.");

  std::string libraryPath = getLibraryPath("libvelox_str_function_my_dynamic");
  loadDynamicLibrary(libraryPath.data());
  auto signaturesAfterFirst = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfterFirst, signaturesBefore + 1);
  EXPECT_EQ("123", dynamicFunctionStr("0"));
  auto resolved = registry.resolveFunction("dynamic_3", {});
  EXPECT_EQ(TypeKind::VARCHAR, resolved->type()->kind());

  VELOX_ASSERT_THROW(
    dynamicFunctionInt(0), 
    "Expression evaluation result is not of expected type: dynamic_3() -> CONSTANT vector of type VARCHAR");

  std::string libraryPathInt = getLibraryPath("libvelox_int_function_my_dynamic");
  loadDynamicLibrary(libraryPathInt.data());

  // The first function loaded should be rewritten.
  VELOX_ASSERT_THROW(
    dynamicFunctionStr("0"), 
    "Expression evaluation result is not of expected type: dynamic_3() -> CONSTANT vector of type BIGINT");
  EXPECT_EQ(123, dynamicFunctionInt(0));
  auto signaturesAfterSecond = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfterSecond, signaturesAfterFirst);
  auto resolvedAfterSecond = registry.resolveFunction("dynamic_3", {});
  EXPECT_EQ(TypeKind::BIGINT, resolvedAfterSecond->type()->kind());

}

TEST_F(DynamicLinkTest, dynamicLoadErrFunc) {
  const auto dynamicFunctionErr = [&](const std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<int64_t>("dynamic_4(c0)", a, b);
  };

  const auto dynamicFunction = [&](const facebook::velox::RowVectorPtr& arr) {
    return evaluateOnce<int64_t>("dynamic_4(c0)", arr);
  };

  auto signaturesBefore = getFunctionSignatures().size();
  VELOX_ASSERT_THROW(
      dynamicFunctionErr(0,0), "Scalar function doesn't exist: dynamic_4.");

  std::string libraryPath = getLibraryPath("libvelox_function_err_my_dynamic");
  loadDynamicLibrary(libraryPath.data());

  auto signaturesAfter = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfter, signaturesBefore + 1);

  // Expecting a fail because we are not passing in an array.
  VELOX_ASSERT_THROW(
    dynamicFunctionErr(0,0), 
    "Scalar function signature is not supported: dynamic_4(BIGINT). Supported signatures: (array(bigint)) -> bigint.");

  auto check = makeRowVector({ makeNullableArrayVector(std::vector<std::vector<std::optional<int64_t>>>{{0, 1, 3, 4, 5, 6, 7, 8, 9}})});

  // Expecting a success because we are passing in an array.
  EXPECT_EQ(123, dynamicFunction(check));
}

} // namespace facebook::velox::functions::test
