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

#include "velox/exec/fuzzer/DuckQueryRunner.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook;
using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::exec::test {

class DuckQueryRunnerTest : public ::testing::Test,
                            public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    velox::functions::prestosql::registerAllScalarFunctions();
    velox::aggregate::prestosql::registerAllAggregateFunctions();
    velox::window::prestosql::registerAllWindowFunctions();
    velox::parse::registerTypeResolver();
  }

  // Checks for the malformed shape rather than the correct one: several
  // visitors emit a "FROM (SELECT" of their own, so a positive check would pass
  // even with the source left bare.
  void expectNoBareSelectSource(const std::optional<std::string>& sql) {
    ASSERT_TRUE(sql.has_value());
    EXPECT_EQ(sql.value().find("FROM SELECT"), std::string::npos)
        << "a SELECT source must be parenthesized: " << sql.value();
  }

  // The mirror image: a source rendering as a table name must not be wrapped,
  // because DuckDB rejects "FROM (tmp)".
  void expectBareTableNameSource(const std::optional<std::string>& sql) {
    ASSERT_TRUE(sql.has_value());
    EXPECT_EQ(sql.value().find("FROM ("), std::string::npos)
        << "a table name source must stay bare: " << sql.value();
  }
};

// Shaping the FROM operand wrong does not fail loudly: the reference query
// just errors out and the iteration is reported as unverified.
TEST_F(DuckQueryRunnerTest, fromOperandParenthesization) {
  auto aggregatePool = rootPool_->addAggregateChild("fromOperand");
  DuckQueryRunner queryRunner{aggregatePool.get()};
  const auto dataType = ROW({"c0", "c1"}, BIGINT());

  // TableScan and Values sources both render as a table name, which must stay
  // bare because DuckDB rejects it in parentheses.
  {
    expectBareTableNameSource(queryRunner.toSql(
        PlanBuilder()
            .tableScan("tmp", dataType)
            .singleAggregation({"c0"}, {"sum(c1)"})
            .planNode()));

    auto values = makeRowVector(
        {"c0", "c1"},
        {
            makeFlatVector<int64_t>({1, 2, 3}),
            makeFlatVector<int64_t>({10, 20, 30}),
        });
    expectBareTableNameSource(queryRunner.toSql(
        PlanBuilder()
            .values({values})
            .singleAggregation({"c0"}, {"sum(c1)"})
            .planNode()));
  }

  // Every visitor that takes a source, over a Project so the source renders as
  // a SELECT rather than a table name.
  {
    auto project = [&] {
      return PlanBuilder()
          .tableScan("tmp", dataType)
          .project({"c0", "c1 + 1 AS c1"});
    };

    expectNoBareSelectSource(queryRunner.toSql(
        project().singleAggregation({"c0"}, {"sum(c1)"}).planNode()));
    expectNoBareSelectSource(queryRunner.toSql(
        project().project({"c0", "c1 + 2 AS c1"}).planNode()));
    expectNoBareSelectSource(
        queryRunner.toSql(project().rowNumber({"c0"}).planNode()));
    expectNoBareSelectSource(queryRunner.toSql(
        project().topNRowNumber({"c0"}, {"c1"}, 10, false).planNode()));
    expectNoBareSelectSource(queryRunner.toSql(
        project()
            .window({"row_number() over (partition by c0 order by c1)"})
            .planNode()));
  }
}

} // namespace facebook::velox::exec::test
