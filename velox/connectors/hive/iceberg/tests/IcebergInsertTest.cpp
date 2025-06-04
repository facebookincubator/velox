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

#include <folly/init/Init.h>

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg::test {
class IcebergInsertTest
    : public testing::WithParamInterface<dwio::common::FileFormat>,
      public IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    rowType_ =
        ROW({"c1", "c2", "c3", "c4", "c5", "c6", "c7"},
            {BIGINT(),
             INTEGER(),
             SMALLINT(),
             REAL(),
             DOUBLE(),
             VARCHAR(),
             BOOLEAN()});
  }
};

TEST_P(IcebergInsertTest, testIcebergTableWrite) {
  const auto& format = GetParam();
  fileFormat_ = format;
  const auto outputDirectory = exec::test::TempDirectoryPath::create();
  const auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
  constexpr int32_t numBatches = 10;
  constexpr int32_t vectorSize = 5'000;
  const auto vectors = createTestData(numBatches, vectorSize, false, 0.0);
  auto dataSink =
      createIcebergDataSink(rowType_, outputDirectory->getPath(), {});

  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  const auto commitTasks = dataSink->close();
  createDuckDbTable(vectors);
  auto splits = createSplitsForDirectory(dataPath);
  ASSERT_EQ(splits.size(), commitTasks.size());
  auto plan = exec::test::PlanBuilder().tableScan(rowType_).planNode();
  assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
}

INSTANTIATE_TEST_SUITE_P(
    IcebergInsertTest,
    IcebergInsertTest,
    testing::Values(
        dwio::common::FileFormat::DWRF
#ifdef VELOX_ENABLE_PARQUET
        ,
        dwio::common::FileFormat::PARQUET
#endif
        ));

} // namespace facebook::velox::connector::hive::iceberg::test

// This main is needed for some tests on linux.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Signal handler required for ThreadDebugInfoTest
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
