#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;

class CudfBatchConcatTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    // Enable cuDF debug logs to help trace operator adaptation and runtime
    // behavior during tests.
    facebook::velox::cudf_velox::CudfConfig::getInstance().debugEnabled = true;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void updateCudfConfig(int32_t min, std::optional<int32_t> max) {
    auto& config = CudfConfig::getInstance();
    config.batchSizeMinThreshold = min;
    config.batchSizeMaxThreshold = max;
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatSequence(T start, vector_size_t size) {
    return makeFlatVector<T>(size, [start](auto row) { return start + row; });
  }

  // Forces fragmented input that Values won't coalesce
  core::PlanNodePtr createFragmentedPlan(
      const std::vector<RowVectorPtr>& vectors,
      std::shared_ptr<core::PlanNodeIdGenerator> generator) {
    std::vector<core::PlanNodePtr> sources;
    for (const auto& vec : vectors) {
      sources.push_back(PlanBuilder(generator).values({vec}).planNode());
    }
    return PlanBuilder(generator).localPartition({}, sources).planNode();
  }
};

// Tests that concat works when input batches are smaller than the target
// batch size.
TEST_F(CudfBatchConcatTest, lesserThanTargetRowsPassesAtEnd) {
  int32_t minTarget = 100;
  updateCudfConfig(minTarget, std::nullopt);

  auto data = makeRowVector({makeFlatSequence<int64_t>(0, 50)});
  auto generator = std::make_shared<core::PlanNodeIdGenerator>();

  auto plan =
      PlanBuilder(generator)
          .values({data})
          .addNode([&](auto id, auto source) {
            return std::make_shared<core::CudfBatchConcatNode>(id, source);
          })
          .planNode();

  exec::CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  auto [cursor, batches] = exec::test::readCursor(
      params, [](auto* c) { c->setNoMoreSplits(); }, 3000);

  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0]->size(), 50);
}

// Tests that concat works when input batches sum to more than the target
// and the output is drained in pieces.
TEST_F(CudfBatchConcatTest, fragmentationAndDraining) {
  int32_t minTarget = 30;
  int32_t maxBatch = 20;
  updateCudfConfig(minTarget, maxBatch);

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  auto sourcePlan = createFragmentedPlan(vectors, generator);

  // Correct argument order: (sourcePlan, generator)
  auto plan =
      PlanBuilder(sourcePlan, generator)
          .addNode([&](auto id, auto source) {
            return std::make_shared<core::CudfBatchConcatNode>(id, source);
          })
          .planNode();

  exec::CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  auto [cursor, batches] = exec::test::readCursor(
      params, [](auto* c) { c->setNoMoreSplits(); }, 3000);

  ASSERT_EQ(batches.size(), 3);
  for (const auto& b : batches) {
    ASSERT_EQ(b->size(), 20);
  }
}

// Tests if the last fragmented batch smaller than targetRows_ is
// rebuffered and concatenated with upcoming rows correctly.
TEST_F(CudfBatchConcatTest, fragmentationRebuffersSmallTrailingBatch) {
  int32_t minTarget = 50;
  int32_t maxBatch = 40;
  updateCudfConfig(minTarget, maxBatch);

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  auto sourcePlan = createFragmentedPlan(vectors, generator);

  // Correct argument order: (sourcePlan, generator)
  auto plan =
      PlanBuilder(sourcePlan, generator)
          .addNode([&](auto id, auto source) {
            return std::make_shared<core::CudfBatchConcatNode>(id, source);
          })
          .planNode();

  exec::CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  auto [cursor, batches] = exec::test::readCursor(
      params, [](auto* c) { c->setNoMoreSplits(); }, 3000);

  ASSERT_EQ(batches.size(), 2);
  ASSERT_EQ(batches[0]->size(), 40);
  ASSERT_EQ(batches[1]->size(), 20);
}
