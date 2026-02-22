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
#include "velox/functions/prestosql/types/SetDigestType.h"

namespace facebook::velox::exec::test {
namespace {

class PrestoQueryRunnerSetDigestTransformTest
    : public PrestoQueryRunnerIntermediateTypeTransformTestBase {};

TEST_F(PrestoQueryRunnerSetDigestTransformTest, isIntermediateOnlyType) {
  ASSERT_TRUE(isIntermediateOnlyType(SETDIGEST()));
  ASSERT_TRUE(isIntermediateOnlyType(ARRAY(SETDIGEST())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(SETDIGEST(), SMALLINT())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(VARBINARY(), SETDIGEST())));
  ASSERT_TRUE(isIntermediateOnlyType(ROW({SETDIGEST(), SMALLINT()})));
  ASSERT_TRUE(isIntermediateOnlyType(ROW(
      {SMALLINT(), TIMESTAMP(), ARRAY(ROW({MAP(VARCHAR(), SETDIGEST())}))})));
}

TEST_F(PrestoQueryRunnerSetDigestTransformTest, transform) {
  test(SETDIGEST());
}

TEST_F(PrestoQueryRunnerSetDigestTransformTest, transformArray) {
  testArray(SETDIGEST());
}

TEST_F(PrestoQueryRunnerSetDigestTransformTest, transformMap) {
  testMap(SETDIGEST());
}

TEST_F(PrestoQueryRunnerSetDigestTransformTest, transformRow) {
  testRow(SETDIGEST());
}

} // namespace
} // namespace facebook::velox::exec::test
