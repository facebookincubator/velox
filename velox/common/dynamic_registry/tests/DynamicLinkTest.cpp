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
#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions::test {

class DynamicLinkTest : public FunctionBaseTest {};

TEST_F(DynamicLinkTest, dynamicLoadOneFunc) {
  const auto dynamicFunction = [&](std::optional<double> a) {
    return evaluateOnce<int64_t>("dynamic_1()", a);
  };

  auto signaturesBefore = getFunctionSignatures().size();

  // Function does not exist yet.
  VELOX_ASSERT_THROW(
      dynamicFunction(0), "Scalar function doesn't exist: dynamic_1.");

  // Dynamically load the library.
  std::string libraryPath = VELOX_TEST_DYNAMIC_LIBRARY_PATH;
  libraryPath.append("/libvelox_function_my_dynamic")
  .append(VELOX_TEST_DYNAMIC_LIBRARY_PATH_SUFFIX);

  loadDynamicLibrary(libraryPath.data());

  auto signaturesAfter = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfter, signaturesBefore + 1);

  // Make sure the function exists now.
  EXPECT_EQ(123, dynamicFunction(0));
}

TEST_F(DynamicLinkTest, dynamicLoadSameFuncTwice) {
  const auto dynamicFunction = [&](std::optional<double> a) {
    return evaluateOnce<int64_t>("dynamic_2()", a);
  };
  auto signaturesBefore = getFunctionSignatures().size();

  // Function does not exist yet.
  VELOX_ASSERT_THROW(
      dynamicFunction(0), "Scalar function doesn't exist: dynamic_2.");
  
  // Dynamically load the library.
  std::string libraryPath = VELOX_TEST_DYNAMIC_LIBRARY_PATH;
  libraryPath.append("/libvelox_function_same_twice_my_dynamic")
  .append(VELOX_TEST_DYNAMIC_LIBRARY_PATH_SUFFIX);

  loadDynamicLibrary(libraryPath.data());

  auto signaturesAfterFirst = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfterFirst, signaturesBefore + 1);

  // Make sure the function exists now.
  EXPECT_EQ(123, dynamicFunction(0));
  // load same shared library again
  loadDynamicLibrary(libraryPath.data());
  auto signaturesAfterSecond = getFunctionSignatures().size();
  // should have no change from the second attempt
  EXPECT_EQ(signaturesAfterSecond, signaturesAfterFirst);
}

TEST_F(DynamicLinkTest, dynamicLoadTwoOfTheSameName) {
  const auto dynamicFunctionInt = [&](std::optional<double> a) {
    return evaluateOnce<int64_t>("dynamic_3()", a);
  };
  const auto dynamicFunctionStr = [&](std::optional<std::string> a) {
    return evaluateOnce<std::string>("dynamic_3()", a);
  };

  auto signaturesBefore = getFunctionSignatures().size();

  // Function does not exist yet.
  VELOX_ASSERT_THROW(
      dynamicFunctionStr("0"), "Scalar function doesn't exist: dynamic_3.");

  // Dynamically load the library.
  std::string libraryPathStr = VELOX_TEST_DYNAMIC_LIBRARY_PATH;
 libraryPathStr.append("/libvelox_str_function_my_dynamic")
 .append(VELOX_TEST_DYNAMIC_LIBRARY_PATH_SUFFIX);

  loadDynamicLibrary(libraryPathStr.data());

  auto signaturesAfterFirst = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfterFirst, signaturesBefore + 1);

  // Make sure the function exists now.
  EXPECT_EQ("123", dynamicFunctionStr("0"));

  // Function does not exist yet.
  VELOX_ASSERT_THROW(
    dynamicFunctionInt(0), 
    "Expression evaluation result is not of expected type: dynamic_3() -> CONSTANT vector of type VARCHAR");

  // Dynamically load the library.
  std::string libraryPathInt = VELOX_TEST_DYNAMIC_LIBRARY_PATH;
  libraryPathInt.append("/libvelox_int_function_my_dynamic")
  .append(VELOX_TEST_DYNAMIC_LIBRARY_PATH_SUFFIX);

  loadDynamicLibrary(libraryPathInt.data());
  // confirm the first function loaded got rewritten
  VELOX_ASSERT_THROW(
    dynamicFunctionStr("0"), 
    "Expression evaluation result is not of expected type: dynamic_3() -> CONSTANT vector of type BIGINT");

  // confirm the second function got loaded
  EXPECT_EQ(123, dynamicFunctionInt(0));
  auto signaturesAfterSecond = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfterSecond, signaturesAfterFirst);

}

//TODO fix this testcase. currently not finding the case giving us an error
TEST_F(DynamicLinkTest, dynamicLoadErrFunc) {
  // This shouldnt work as we're trying to register a function with an Array not an int64_t
  const auto dynamicFunction = [&](const std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<int64_t>("dynamic_4(c0)", a, b);
  };
  // This doesnt work either
  const auto dynamicFunctionErr = [&](const facebook::velox::RowVectorPtr& arr) {
    return evaluateOnce<int64_t>("dynamic_4(c0)", arr);
  };

  auto signaturesBefore = getFunctionSignatures().size();

  // Function does not exist yet.
  VELOX_ASSERT_THROW(
      dynamicFunction(0,0), "Scalar function doesn't exist: dynamic_4.");

  // Dynamically load the library.
  std::string libraryPath = VELOX_TEST_DYNAMIC_LIBRARY_PATH;
  libraryPath.append("/libvelox_function_err_my_dynamic")
  .append(VELOX_TEST_DYNAMIC_LIBRARY_PATH_SUFFIX);

  loadDynamicLibrary(libraryPath.data());

  auto signaturesAfter = getFunctionSignatures().size();
  EXPECT_EQ(signaturesAfter, signaturesBefore + 1);
  VELOX_ASSERT_THROW(
    dynamicFunction(0,0), 
    "Scalar function signature is not supported: dynamic_4(BIGINT). Supported signatures: (array(bigint)) -> bigint.");
  auto check =  makeRowVector({
      makeFlatVector<int64_t>({0, 1, 3, 4, 5, 6, 7, 8, 9}),
  });
  // Not sure why this fails
  VELOX_ASSERT_THROW(
    dynamicFunctionErr(check), 
    "Scalar function signature is not supported: dynamic_4(BIGINT). Supported signatures: (array(bigint)) -> bigint.");
}

} // namespace facebook::velox::functions::test
