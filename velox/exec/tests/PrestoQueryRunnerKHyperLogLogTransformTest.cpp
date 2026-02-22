/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/fuzzer/PrestoQueryRunnerIntermediateTypeTransforms.h"
#include "velox/exec/tests/PrestoQueryRunnerIntermediateTypeTransformTestBase.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"

namespace facebook::velox::exec::test {
namespace {

class PrestoQueryRunnerKHyperLogLogTransformTest
    : public PrestoQueryRunnerIntermediateTypeTransformTestBase {};

TEST_F(PrestoQueryRunnerKHyperLogLogTransformTest, isIntermediateOnlyType) {
  ASSERT_TRUE(isIntermediateOnlyType(KHYPERLOGLOG()));
  ASSERT_TRUE(isIntermediateOnlyType(ARRAY(KHYPERLOGLOG())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(KHYPERLOGLOG(), SMALLINT())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(VARBINARY(), KHYPERLOGLOG())));
  ASSERT_TRUE(isIntermediateOnlyType(ROW({KHYPERLOGLOG(), SMALLINT()})));
  ASSERT_TRUE(isIntermediateOnlyType(ROW(
      {SMALLINT(),
       TIMESTAMP(),
       ARRAY(ROW({MAP(VARCHAR(), KHYPERLOGLOG())}))})));
}

TEST_F(PrestoQueryRunnerKHyperLogLogTransformTest, transform) {
  test(KHYPERLOGLOG());
}

TEST_F(PrestoQueryRunnerKHyperLogLogTransformTest, transformArray) {
  testArray(KHYPERLOGLOG());
}

TEST_F(PrestoQueryRunnerKHyperLogLogTransformTest, transformMap) {
  testMap(KHYPERLOGLOG());
}

TEST_F(PrestoQueryRunnerKHyperLogLogTransformTest, transformRow) {
  testRow(KHYPERLOGLOG());
}

} // namespace
} // namespace facebook::velox::exec::test
