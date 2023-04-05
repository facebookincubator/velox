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

#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/aggregates/tests/AggregationTestBase.h"
#include "velox/vector/VectorSaver.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::aggregate::test {

class AggregationFailedTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
    facebook::velox::aggregate::prestosql::registerAllAggregateFunctions();
  }

  auto execute(const core::PlanNodePtr& plan) {
    LOG(INFO) << "Executing query plan: " << std::endl
              << plan->toString(true, true);

    AssertQueryBuilder builder(plan);
    auto result = builder.maxDrivers(2).copyResults(pool());
    LOG(INFO) << result->toString();

    return result;
  }

  std::string makeDuckWindowSql(
      const std::vector<std::string>& partitionKeys,
      const std::vector<std::string>& sortingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::string>& inputs) {
    std::stringstream sql;
    sql << "SELECT " << folly::join(", ", inputs) << ", "
        << folly::join(", ", aggregates) << " OVER (";

    if (!partitionKeys.empty()) {
      sql << "partition by " << folly::join(", ", partitionKeys);
    }
    if (!sortingKeys.empty()) {
      sql << " order by " << folly::join(", ", sortingKeys);
    }

    sql << ") FROM tmp";

    return sql.str();
  }

  std::optional<MaterializedRowMultiset> computeDuckResults(
      const std ::string& sql,
      const std::vector<RowVectorPtr>& input,
      const RowTypePtr& resultType) {
    try {
      DuckDbQueryRunner queryRunner;
      queryRunner.createTable("tmp", input);
      return queryRunner.execute(sql, resultType);
    } catch (std::exception& e) {
      LOG(WARNING) << "Couldn't get results from DuckDB";
      return std::nullopt;
    }
  }

  std::optional<MaterializedRowMultiset> computeDuckWindow(
      const std::vector<std::string>& partitionKeys,
      const std::vector<std::string>& sortingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<RowVectorPtr>& input,
      const core::PlanNodePtr& plan) {
    // Check if DuckDB supports specified aggregate functions.
    auto windowNode = dynamic_cast<const core::WindowNode*>(plan.get());
    VELOX_CHECK_NOT_NULL(windowNode);

    const auto& outputType = plan->outputType();

    return computeDuckResults(
        makeDuckWindowSql(
            partitionKeys,
            sortingKeys,
            aggregates,
            asRowType(input[0]->type())->names()),
        input,
        outputType);
  }
};

TEST_F(AggregationFailedTest, varPopFloatingPointPrecision) {
  std::vector<std::string> inputFiles{
      "velox_vector_jNeHLY",
      "velox_vector_slaPuy",
      "velox_vector_1FSZkT",
      "velox_vector_KJBzyJ",
      "velox_vector_y5Wuzk",
      "velox_vector_2HmCbt",
      "velox_vector_lMxsDL",
      "velox_vector_ZXDLHI",
      "velox_vector_GV6jNV",
      "velox_vector_rDw7Ox"};
  std::vector<RowVectorPtr> inputs;
  for (const auto& file : inputFiles) {
    inputs.push_back(std::dynamic_pointer_cast<RowVector>(
        facebook::velox::restoreVectorFromFile(
            ("/home/weihe/fuzzer_repro/varpop/" + file).c_str(), pool())));
  }

  auto plan =
      PlanBuilder()
          .values(inputs)
          .window(
              {"var_pop(c0) over (partition by p0 order by s0, s1, s2, s3)"})
          .planNode();
  auto result = execute(plan);
  auto expectedResult = computeDuckWindow(
      {"p0"}, {"s0", "s1", "s2", "s3"}, {"var_pop(c0)"}, inputs, plan);
  assertEqualResults(expectedResult.value(), {result});
}

} // namespace facebook::velox::aggregate::test
