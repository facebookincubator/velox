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

// IMPORTANT: Includes that reference `exec::Driver` / `exec::DriverFactory`
// from inside `namespace facebook::velox::cudf_velox` (e.g. ToCudf.h) must be
// included before `CudfAbfsTest.h`, which opens `cudf_velox::exec::test` and
// would otherwise shadow the upstream `velox::exec` namespace during qualified
// name lookup. This mirrors the pattern already used by `S3ReadTest.cpp`.
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/tests/CudfAbfsTest.h"

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;

namespace {

constexpr const char* kCudfHiveConnectorId{"test-cudf-hive"};

class AbfsReadTest : public CudfAbfsTest, public ::test::VectorTestBase {};

} // namespace

TEST_F(AbfsReadTest, readIntParquet) {
  const auto sourceFile = test::getDataFilePath(
      "velox/experimental/cudf/tests",
      "../../../dwio/parquet/tests/examples/int.parquet");

  const auto abfsFilePath = uploadFile(sourceFile);

  auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfHiveConnectorId,
          "int_table",
          common::SubfieldFilters{},
          nullptr);

  auto plan = PlanBuilder(pool())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto split =
      facebook::velox::connector::hive::HiveConnectorSplitBuilder(abfsFilePath)
          .connectorId(kCudfHiveConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET)
          .build();

  auto actual = AssertQueryBuilder(plan).split(split).copyResults(pool());

  constexpr int64_t kExpectedRows = 10;
  auto expected = makeRowVector(
      {makeFlatVector<int32_t>(
           kExpectedRows, [](auto row) { return row + 100; }),
       makeFlatVector<int64_t>(
           kExpectedRows, [](auto row) { return row + 1000; })});
  assertEqualResults({expected}, {actual});
}
