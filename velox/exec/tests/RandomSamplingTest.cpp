//#include <core_systems/if/common/gen-cpp2/queries_types.h>
#include <velox/type/Timestamp.h>
#include <cstdint>
#include "velox/common/base/tests/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/exec/PartitionedOutputBufferManager.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/TempFilePath.h"
#include "velox/functions/Registerer.h"

using namespace facebook::velox;

int64_t count = 0;
const int64_t kCountWrap = 4;

template <typename T>
struct TestRandFunction {
  static constexpr bool is_deterministic = false;

  FOLLY_ALWAYS_INLINE bool call(double& result) {
    result = (count % kCountWrap == 0) ? 0.0 : 1.0;
    count++;

    return true;
  }
};

class RandomSamplingTest : public virtual exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();

    registerFunction<TestRandFunction, double>({"testrand", "testrandom"});
  }

  static void SetUpTestCase() {
    HiveConnectorTestBase::SetUpTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return HiveConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  std::unique_ptr<exec::test::TaskCursor> createTableScanTask(
      std::string query,
      std::shared_ptr<exec::test::TempFilePath> filePath) {
    exec::test::CursorParameters params;
    params.planNode =
        exec::test::PlanBuilder().tableScan(rowType_, {}, query).planNode();
    auto cursor = std::make_unique<exec::test::TaskCursor>(params);
    cursor->task()->addSplit("0", makeHiveSplit(filePath->path));
    cursor->task()->noMoreSplits("0");
    return cursor;
  }

  std::unique_ptr<exec::test::TaskCursor> createFilterTask(
      std::string query,
      std::shared_ptr<exec::test::TempFilePath> filePath) {
    exec::test::CursorParameters params;
    params.planNode =
        exec::test::PlanBuilder().tableScan(rowType_).filter(query).planNode();
    auto cursor = std::make_unique<exec::test::TaskCursor>(params);
    cursor->task()->addSplit("0", makeHiveSplit(filePath->path));
    cursor->task()->noMoreSplits("0");
    return cursor;
  }

  exec::Split makeHiveSplitWithGroup(std::string path, int32_t group) {
    return exec::Split(makeHiveConnectorSplit(std::move(path)), group);
  }

  exec::Split makeHiveSplit(std::string path) {
    return exec::Split(makeHiveConnectorSplit(std::move(path)));
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           REAL(),
           DOUBLE(),
           VARCHAR(),
           TINYINT()})};
};

TEST_F(RandomSamplingTest, aggregationPushdown) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = exec::test::TempFilePath::create();
  writeToFile(filePath->path, vectors);

  auto rowsRead = [&](std::unique_ptr<exec::test::TaskCursor>& cursor) {
    int32_t numRead = 0;
    while (cursor->moveNext()) {
      auto vector = cursor->current();
      numRead += vector->size();
    }
    return numRead;
  };

  auto cursor = createTableScanTask("testrandom() < 0.1", filePath);
  exec::test::waitForTaskCompletion(cursor->task().get());
  int32_t numRead = rowsRead(cursor);
  EXPECT_EQ(numRead, 10000 / kCountWrap);

  cursor = createFilterTask("testrandom() <= 0.1", filePath);
  exec::test::waitForTaskCompletion(cursor->task().get());
  numRead = rowsRead(cursor);
  EXPECT_EQ(numRead, 10000 / kCountWrap);
}
