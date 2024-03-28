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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class InputFileNameTest : public SparkFunctionBaseTest {
 protected:
  void testInputFileName(
      std::string fileName,
      std::string expected,
      bool available = true) {
    if (available) {
      facebook::velox::core::inputFileName = fileName;
    }

    EXPECT_EQ(
        evaluateOnce<std::string>(
            "input_file_name()", makeRowVector(ROW({}), 1)),
        std::string(expected));
    facebook::velox::core::inputFileName = "";
  }
};

TEST_F(InputFileNameTest, basic) {
  testInputFileName("file:///tmp/text.txt", "file:///tmp/text.txt");
  testInputFileName(
      "file:///tmp/test dir/a=b/main#text.txt",
      "file:///tmp/test%20dir/a=b/main%23text.txt");
  testInputFileName("file:///xxx", "", false);
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
