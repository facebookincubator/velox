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

#include "velox/functions/prestosql/types/fuzzer_utils/P4HyperLogLogInputGenerator.h"

#include <gtest/gtest.h>

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/type/Variant.h"

namespace facebook::velox::fuzzer::test {

class P4HyperLogLogInputGeneratorTest
    : public functions::test::FunctionBaseTest {};

TEST_F(P4HyperLogLogInputGeneratorTest, generate) {
  P4HyperLogLogInputGenerator generator(123, 0.1, pool());

  size_t numTrials = 100;
  for (size_t i = 0; i < numTrials; ++i) {
    variant generated = generator.generate();
    ASSERT_TRUE(generated.hasValue());
    if (!generated.isNull()) {
      ASSERT_EQ(generated.kind(), TypeKind::VARBINARY);
      // Verify that the generated data is not empty (dense HLL should have
      // data)
      auto binaryData = generated.value<TypeKind::VARBINARY>();
      EXPECT_GT(binaryData.size(), 0);
    }
  }
}

} // namespace facebook::velox::fuzzer::test
