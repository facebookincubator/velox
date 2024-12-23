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

#pragma once

#include <vector>

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

DECLARE_string(data_path);
DECLARE_bool(create_dataset);

namespace facebook::velox::optimizer::test {

class ParquetTpchTest : public testing::Test {
 protected:
  static void SetUpTestSuite();

  static void TearDownTestSuite();

  static void saveTpchTablesAsParquet();

  void assertQuery(
      int queryId,
      const std::optional<std::vector<uint32_t>>& sortingKeys = {}) {
    auto tpchPlan = tpchBuilder_->getQueryPlan(queryId);
    auto duckDbSql = tpch::getQuery(queryId);
    assertQuery(tpchPlan, duckDbSql, sortingKeys);
  }

  std::shared_ptr<exec::Task> assertQuery(
      const exec::test::TpchPlan& tpchPlan,
      const std::string& duckQuery,
      const std::optional<std::vector<uint32_t>>& sortingKeys) const;

  static std::shared_ptr<exec::test::DuckDbQueryRunner> duckDb_;
  static std::string createPath_;
  static std::string path_;
  static std::shared_ptr<exec::test::TempDirectoryPath> tempDirectory_;
  static std::shared_ptr<exec::test::TpchQueryBuilder> tpchBuilder_;

  static constexpr char const* kTpchConnectorId{"test-tpch"};
};

} // namespace facebook::velox::optimizer::test
