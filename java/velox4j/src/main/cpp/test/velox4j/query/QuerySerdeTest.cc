#include <gtest/gtest.h>
#include <velox/exec/tests/utils/HiveConnectorTestBase.h>
#include <velox/exec/tests/utils/PlanBuilder.h>
#include "velox4j/query/Query.h"
#include "velox4j/test/Init.h"

namespace velox4j {
using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class QuerySerdeTest : public testing::Test, public test::VectorTestBase {
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
} // namespace velox4j
