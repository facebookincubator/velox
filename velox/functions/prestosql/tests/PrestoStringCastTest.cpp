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

#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/exec/fuzzer/PrestoQueryRunner.h"
#include "velox/exec/fuzzer/PrestoSql.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/tests/CastBaseTest.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox;

namespace facebook::velox::functions::test {

class PrestoStringCastTest : public functions::test::CastBaseTest {
 public:
  void SetUp() override {
    velox::functions::prestosql::registerAllScalarFunctions();
    dwrf::registerDwrfReaderFactory();
    dwrf::registerDwrfWriterFactory();
    queryRunner_ = std::make_unique<PrestoQueryRunner>(
        rootPool_.get(),
        "http://127.0.0.1:8080",
        "hive",
        static_cast<std::chrono::milliseconds>(5000));
  }

  void TearDown() override {
    dwrf::unregisterDwrfReaderFactory();
    dwrf::unregisterDwrfWriterFactory();
  }

  void evalCastTypedExpression(
      const VectorPtr& data,
      const TypePtr& outputType) {
    const auto kOutputColName = "p0";

    auto rows = makeRowVector({data});

    auto inputType = rows->childAt(0)->type();

    auto typedExpr = buildCastExpr(inputType, outputType, false);
    auto plan = velox::exec::test::PlanBuilder()
                    .values({rows})
                    .projectTypedExpressions({kOutputColName}, {typedExpr})
                    .planNode();

    auto sql = queryRunner_->toSql(plan);
    ASSERT_TRUE(sql.has_value());
    SCOPED_TRACE(fmt::format("SQL: {}", sql.value()));
    ASSERT_EQ(
        sql.value(),
        fmt::format(
            "SELECT cast(c0 as {}) as {} FROM (tmp)",
            exec::test::toTypeSql(outputType),
            kOutputColName));

    // auto outputRowType = ROW({kOutputColName}, {outputType});
    auto [prestoResults, errorCode] = queryRunner_->execute(plan);

    auto veloxResults =
        velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
    ASSERT_TRUE(prestoResults.has_value());
    velox::exec::test::assertEqualResults(
        prestoResults.value(), plan->outputType(), {veloxResults});
  }

  const TypePtr kTargetType_ = VARCHAR();

 private:
  std::unique_ptr<PrestoQueryRunner> queryRunner_;
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
};

TEST_F(PrestoStringCastTest, DISABLED_varchar) {
  auto data = makeNullableFlatVector<std::string>(
      {std::nullopt,
       "ABCDEFSFFFFFFF",
       "ABCDEFSDDDDDDDDDDD"
       "ABCDEFSEEEEEEEEEEEEEEE"});

  evalCastTypedExpression(data, kTargetType_);
}

TEST_F(PrestoStringCastTest, DISABLED_boolean) {
  auto data = makeNullableFlatVector<bool>({std::nullopt, true, false});

  evalCastTypedExpression(data, kTargetType_);
}

TEST_F(PrestoStringCastTest, DISABLED_smallint) {
  auto data = makeNullableFlatVector<int16_t>(
      {std::nullopt,
       12345,
       -12345,
       std::numeric_limits<int16_t>::min(),
       std::numeric_limits<int16_t>::max()});

  evalCastTypedExpression(data, kTargetType_);
}

TEST_F(PrestoStringCastTest, DISABLED_integer) {
  auto data = makeNullableFlatVector<int32_t>(
      {std::nullopt,
       12345,
       -12345,
       12345678,
       -12345678,
       std::numeric_limits<int32_t>::min(),
       std::numeric_limits<int32_t>::max()});

  evalCastTypedExpression(data, kTargetType_);
}

TEST_F(PrestoStringCastTest, DISABLED_bigint) {
  auto data = makeNullableFlatVector<int64_t>(
      {std::nullopt,
       12345,
       -12345,
       12345678,
       -12345678,
       12345678901234,
       -12345678901234,
       std::numeric_limits<int64_t>::min(),
       std::numeric_limits<int64_t>::max()});

  evalCastTypedExpression(data, kTargetType_);
}

TEST_F(PrestoStringCastTest, DISABLED_real) {
  // Real rounding discrepancy and precision.
  // With rounding of the last digit the results match.
  // 1 of extra rows:
  //      "1.1754944E-38"
  // 1 of missing rows:
  //      "1.17549435E-38"
  auto data = makeNullableFlatVector<float>(
      {std::nullopt,
       12345.0,
       -12345.0,
       12345678,
       -12345678,
       std::numeric_limits<float>::min(),
       std::numeric_limits<float>::max()});

  evalCastTypedExpression(data, kTargetType_);
}

TEST_F(PrestoStringCastTest, DISABLED_double) {
  // One less precision for the Presto result
  // vs the Velox result. With rounding
  // Rounding the result match.
  // 2 extra rows, 2 missing rows
  // 2 of extra rows:
  //        "-9.2233720368547758E18"
  //        "9.2233720368547758E18"
  // 2 of missing rows:
  //        "-9.223372036854776E18"
  //        "9.223372036854776E18"
  auto data = makeNullableFlatVector<double>(
      {std::nullopt,
       12345678,
       -12345678,
       12345678901234,
       -12345678901234,
       std::numeric_limits<double>::min(),
       std::numeric_limits<double>::max()});

  evalCastTypedExpression(data, kTargetType_);
}

TEST_F(PrestoStringCastTest, DISABLED_timestamp) {
  // TODO the DRWF data written to file is as if it was written
  // in GMT+3 session timezone instead of UTC.
  // As a result the X-Presto-Time-Zone needs to be set accordingly to be able
  // to match the values.
  auto data = makeNullableFlatVector<Timestamp>(
      {std::nullopt,
       Timestamp(0, 0),
       Timestamp(-1'000'000, 0),
       Timestamp(9'000'000, 500)});

  evalCastTypedExpression(data, kTargetType_);
}

} // namespace facebook::velox::functions::test
