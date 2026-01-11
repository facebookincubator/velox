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
#include <stdint.h>
#include <string.h>
#include <velox/exec/tests/utils/HiveConnectorTestBase.h>
#include <velox/exec/tests/utils/PlanBuilder.h>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "velox4j/conf/Config.h"
#include "velox4j/query/Query.h"
#include "velox4j/test/Init.h"

namespace facebook::velox4j::test {
using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class QuerySerdeTest : public testing::Test,
                       public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    testingEnsureInitializedForSpark();
  }

  QuerySerdeTest() {
    data_ = {makeRowVector({
        makeFlatVector<int64_t>({1, 2, 3}),
        makeFlatVector<int32_t>({10, 20, 30}),
        makeConstant(true, 3),
    })};
  }

  void testSerde(const Query* query) {
    auto serialized = query->serialize();
    auto copy = ISerializable::deserialize<Query>(serialized, pool());
    ASSERT_EQ(query->toString(), copy->toString());
  }

  std::vector<RowVectorPtr> data_;
};

TEST_F(QuerySerdeTest, sanity) {
  auto plan = PlanBuilder()
                  .values({data_})
                  .partialAggregation({"c0"}, {"count(1)", "sum(c1)"})
                  .finalAggregation()
                  .planNode();
  auto query = std::make_shared<Query>(
      plan,
      std::make_shared<const ConfigArray>(
          std::vector<std::pair<std::string, std::string>>({})),
      std::make_shared<const ConnectorConfigArray>(
          std::vector<
              std::pair<std::string, std::shared_ptr<const ConfigArray>>>({})));
  testSerde(query.get());
}
} // namespace facebook::velox4j::test
