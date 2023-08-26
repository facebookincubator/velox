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

#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/VeloxHistory.h"

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/experimental/query/tests/Tpch.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

DEFINE_string(data_path, "", "Path to directory for TPC-H files");

DEFINE_int32(trace, 0, "Enable trace 1=retained plans, 2=abandoned, 3=both");

using namespace facebook::velox;
using namespace facebook::verax;

std::string nodeString(core::PlanNode* node) {
  return node->toString(true, true);
}

class PlanTest : public testing::Test {
 protected:
  void SetUp() override {
    rootPool_ = memory::defaultMemoryManager().addRootPool("velox_sql");
    pool_ = rootPool_->addLeafChild("optimizer");
    allocator_ = std::make_unique<HashStringAllocator>(pool_.get());
    context_ = std::make_unique<QueryGraphContext>(*allocator_);
    queryCtx() = context_.get();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    parse::registerTypeResolver();
    filesystems::registerLocalFileSystem();
    if (!registered) {
      registered = true;
      parquet::registerParquetReaderFactory();
    }
    builder_ = std::make_unique<exec::test::TpchQueryBuilder>(
        dwio::common::FileFormat::PARQUET);
    builder_->initialize(FLAGS_data_path);
    history_ = std::make_unique<VeloxHistory>();
    makeCheats();
    queryCtx_ = std::make_shared<core::QueryCtx>();

    evaluator_ = std::make_unique<exec::SimpleExpressionEvaluator>(
        queryCtx_.get(), pool_.get());
  }

  void makeCheats() {
    history_->recordLeafSelectivity(
        "table: lineitem, range filters: [(l_shipdate, BigintRange: [9205, 9223372036854775807] no nulls)]",
        0.5);
    history_->recordLeafSelectivity(
        "table: orders, range filters: [(o_orderdate, BigintRange: [-9223372036854775808, 9203] no nulls)]",
        0.5);
    history_->recordLeafSelectivity(
        "table: customer, range filters: [(c_mktsegment, Filter(BytesValues, deterministic, null not allowed))]",
        0.2);
    history_->recordLeafSelectivity(
        "table: part, remaining filter: (like(ROW[\"p_name\"],\"%green%\"))",
        1.0 / 17);
  }

  std::string makePlan(
      std::shared_ptr<const core::PlanNode> plan,
      bool partitioned,
      bool ordered,
      int numRepeats = 1) {
    auto schema = tpchSchema(100, partitioned, ordered, false);
    std::string string;
    for (auto counter = 0; counter < numRepeats; ++counter) {
      Optimization opt(*plan, *schema, *history_, *evaluator_, FLAGS_trace);
      auto result = opt.bestPlan();
      if (counter == numRepeats - 1) {
        string = result->toString(true);
      }
    }
    return fmt::format(
        "=== {} {}:\n{}\n",
        partitioned ? "Partitioned on PK" : "Not partitioned",
        ordered ? "sorted on PK" : "not sorted",
        string);
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;

  std::unique_ptr<HashStringAllocator> allocator_;

  std::unique_ptr<QueryGraphContext> context_;
  std::unique_ptr<VeloxHistory> history_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExpressionEvaluator> evaluator_;
  std::unique_ptr<exec::test::TpchQueryBuilder> builder_;
  static bool registered;
};

bool PlanTest::registered;

TEST_F(PlanTest, q3) {
  auto q = builder_->getQueryPlan(3).plan;
  auto result = makePlan(q, true, true);
  std::cout << result;
  result = makePlan(q, true, false);
  std::cout << result;
}

TEST_F(PlanTest, q9) {
  auto q = builder_->getQueryPlan(9).plan;
  auto result = makePlan(q, true, true);
  std::cout << result;
  result = makePlan(q, true, false);
  std::cout << result;
}

TEST_F(PlanTest, q17) {
  auto q = builder_->getQueryPlan(17).plan;
  auto result = makePlan(q, true, true);
  std::cout << result;
  result = makePlan(q, true, false);
  std::cout << result;
}

void printPlan(core::PlanNode* plan, bool r, bool d) {
  std::cout << plan->toString(r, d) << std::endl;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
