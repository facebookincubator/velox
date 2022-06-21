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
#include <fmt/format.h>

#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/functions/prestosql/aggregates/tests/AggregationTestBase.h"

using facebook::velox::exec::test::PlanBuilder;
using namespace facebook::velox::exec::test;

namespace facebook::velox::aggregate::test {

namespace {

using facebook::velox::exec::test::MaterializedRow;

class MinMaxByAggregationTest : public AggregationTestBase {
 protected:
  void testGlobalAggregation(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& aggName,
      const std::string& colName) {
    const std::string funcName = aggName == kMaxBy ? "max" : "min";
    for (int i = 0; i < rowType_->size(); ++i) {
      const auto valColName = fmt::format("c{}", i);
      const std::string verifyDuckDbSql = "SELECT " + valColName +
          " FROM tmp WHERE " + colName + " = ( SELECT " + funcName + "(" +
          colName + ") FROM tmp) LIMIT 1";
      SCOPED_TRACE(fmt::format(
          "{}({}, {})\nverifyDuckDbSql: {}",
          aggName,
          valColName,
          colName,
          verifyDuckDbSql));
      auto op =
          PlanBuilder()
              .values(vectors)
              .partialAggregation(
                  {}, {fmt::format("{}({}, {})", aggName, valColName, colName)})
              .finalAggregation()
              .planNode();
      assertQuery(op, verifyDuckDbSql);
    }
  }

  void testGroupByAggregation(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& aggName,
      const std::string& colName) {
    const std::string funcName = aggName == kMaxBy ? "max" : "min";
    for (int volColIdx = 0; volColIdx < rowType_->size(); ++volColIdx) {
      const int groupByColIdx = folly::Random::rand32(0, rowType_->size());
      const std::string groupByColName = rowType_->nameOf(groupByColIdx);
      const std::string valColName = rowType_->nameOf(volColIdx);
      const std::string verifyDuckDbSql = "SELECT " + groupByColName + ", " +
          aggName + "(" + valColName + ", " + colName + ") FROM tmp GROUP BY " +
          groupByColName;
      SCOPED_TRACE(fmt::format(
          "{}({}, {}) GROUP BY {}\nverifyDuckDbSql: {}",
          funcName,
          valColName,
          colName,
          groupByColName,
          verifyDuckDbSql));
      auto op =
          PlanBuilder()
              .values(vectors)
              .partialAggregation(
                  {groupByColName},
                  {fmt::format("{}({}, {})", aggName, valColName, colName)})
              .finalAggregation()
              .planNode();
      assertQuery(op, verifyDuckDbSql);
    }
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"},
          {TINYINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           REAL(),
           DOUBLE(),
           BIGINT(),
           VARCHAR()})};
};

TEST_F(MinMaxByAggregationTest, minMaxByPartialConst) {
  for (bool isMin : {false, true}) {
    struct {
      const RowVectorPtr inputRowVector;
      const std::string verifyDuckDbSql;
      const bool expectedBuildFailure = false;

      const std::string debugString() const {
        return fmt::format(
            "\ninputRowVector: {}\nverifyDuckDbSql: {}\nexpectedBuildFailure: {}",
            inputRowVector->toString(),
            verifyDuckDbSql,
            expectedBuildFailure);
      }
    } testSettings[] = {
        // Aggregate by both numeric value and comparison types.
        {makeRowVector({makeConstant(5.0, 10), makeConstant(10, 10)}),
         "SELECT struct_pack(x => 5, y => 10)"},
        // Aggregate by non-numeric value type.
        {makeRowVector(
             {makeConstant<std::string>("abc", 10), makeConstant(10, 10)}),
         "SELECT struct_pack(x => 'abc', y => 10)"},
        // Aggregate by non-numeric comparison type.
        {makeRowVector(
             {makeConstant(5, 10), makeConstant<std::string>("abc", 10)}),
         "SELECT struct_pack(x => 5, y => 'abc')"},
        // Aggregate by both non-numeric value and comparison types.
        {makeRowVector(
             {makeConstant<std::string>("abc", 10),
              makeConstant<std::string>("def", 10)}),
         "SELECT struct_pack(x => 'abc', y => 'def')"},
        // Unsupported value or comparison types.
        {makeRowVector({makeConstant<bool>(true, 10), makeConstant(10, 10)}),
         "",
         true}};
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format("isMin: {}\n {}", isMin, testData.debugString()));
      if (testData.expectedBuildFailure) {
        EXPECT_ANY_THROW(
            PlanBuilder()
                .values({testData.inputRowVector})
                .partialAggregation(
                    {}, {isMin ? "min_by(c0, c1)" : "max_by(c0, c1)"})
                .planNode());
        continue;
      }
      auto op = PlanBuilder()
                    .values({testData.inputRowVector})
                    .partialAggregation(
                        {}, {isMin ? "min_by(c0, c1)" : "max_by(c0, c1)"})
                    .planNode();
      assertQuery(op, testData.verifyDuckDbSql);
    }
  }
}

TEST_F(MinMaxByAggregationTest, minMaxByPartialConstNull) {
  for (bool isMin : {false, true}) {
    struct {
      const RowVectorPtr inputRowVector;
      const std::string verifyDuckDbSql;

      const std::string debugString() const {
        return fmt::format(
            "\ninputRowVector: {}\nverifyDuckDbSql: {}",
            inputRowVector->toString(),
            verifyDuckDbSql);
      }
    } testSettings[] = {
        // Aggregate by with both numeric value and comparison types.
        {makeRowVector(
             {makeConstant(5, 10), makeNullConstant(TypeKind::BIGINT, 10)}),
         "SELECT null"},
        {makeRowVector(
             {makeConstant<double>(5.0, 10),
              makeNullConstant(TypeKind::BIGINT, 10)}),
         "SELECT null"},
        {makeRowVector(
             {makeConstant<float>(5.0, 10),
              makeNullConstant(TypeKind::REAL, 10)}),
         "SELECT null"},
        // Aggregate by with non-numeric comparison types.
        {makeRowVector(
             {makeConstant<float>(5.0, 10),
              makeNullConstant(TypeKind::VARCHAR, 10)}),
         "SELECT NULL"},
        // Aggregate by with both non-numeric value and comparison types.
        {makeRowVector(
             {makeConstant<std::string>("abc", 10),
              makeNullConstant(TypeKind::VARCHAR, 10)}),
         "SELECT NULL"}};
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format("isMin: {}\n {}", isMin, testData.debugString()));
      auto op = PlanBuilder()
                    .values({testData.inputRowVector})
                    .partialAggregation(
                        {}, {isMin ? "min_by(c0, c1)" : "max_by(c0, c1)"})
                    .planNode();
      assertQuery(op, testData.verifyDuckDbSql);
    }
  }
}

TEST_F(MinMaxByAggregationTest, minMaxByPartialNullCase) {
  struct {
    const RowVectorPtr inputRowVector;
    const std::string aggrStatement;
    const std::string verifyDuckDbSql;

    const std::string debugString() const {
      return fmt::format(
          "\ninputRowVector: {}\naggrStatement: {}\nverifyDuckDbSql: {}",
          inputRowVector->toString(),
          aggrStatement,
          verifyDuckDbSql);
    }
  } testSettings[] = {
      // Max by both numeric value and comparison types.
      {makeRowVector({
           makeNullableFlatVector<int64_t>({std::nullopt, 5, 100, 20}),
           makeNullableFlatVector<int64_t>({10, 5, std::nullopt, 8}),
       }),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 10)"},
      {makeRowVector({
           makeNullableFlatVector<double>({std::nullopt, 20, 100, 20}),
           makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 8}),
       }),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 10)"},
      {makeRowVector({
           makeNullableFlatVector<double>(
               {std::nullopt, 20, 100, std::nullopt}),
           makeNullableFlatVector<int64_t>({8, 9, std::nullopt, 10}),
       }),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 10)"},
      // Max by non-numeric value type.
      {makeRowVector({
           makeNullableFlatVector<std::string>(
               {std::nullopt, "abc", "def", "hgd"}),
           makeNullableFlatVector<int64_t>({10, 5, std::nullopt, 8}),
       }),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 10)"},
      {makeRowVector({
           makeNullableFlatVector<std::string>(
               {std::nullopt, "abc", "def", "hgd"}),
           makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 8}),
       }),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 10)"},
      {makeRowVector({
           makeNullableFlatVector<std::string>(
               {std::nullopt, "abc", "def", std::nullopt}),
           makeNullableFlatVector<int64_t>({5, 5, std::nullopt, 10}),
       }),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 10)"},
      // Max by non-numeric comparison types.
      {makeRowVector(
           {makeNullableFlatVector<double>({std::nullopt, 10, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"hgd", std::nullopt, "abc", "def"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'hgd')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 10, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"hgd", "hgd", "abc", std::nullopt})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'hgd')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 10, 10, std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"hgd", std::nullopt, "abc", "wef"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'wef')"},
      // Max by both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"hgd", std::nullopt, "abc", "def"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'hgd')"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<std::string>({"hgd", "hgd", "abc", "def"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'hgd')"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"wef", "abc", std::nullopt, std::nullopt}),
            makeNullableFlatVector<std::string>({"hgd", "hgd", "abc", "wef"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'wef')"},

      // Min by both numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({std::nullopt, 5, 100, 20}),
            makeNullableFlatVector<int64_t>({5, 10, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 5)"},
      {makeRowVector(
           {makeNullableFlatVector<double>({std::nullopt, 20, 100, 20}),
            makeNullableFlatVector<int64_t>({5, 5, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 5)"},
      {makeRowVector({
           makeNullableFlatVector<double>(
               {std::nullopt, 20, 100, std::nullopt}),
           makeNullableFlatVector<int64_t>({8, 9, std::nullopt, 5}),
       }),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 5)"},
      // Min by non-numeric value type.
      {makeRowVector({
           makeNullableFlatVector<std::string>(
               {std::nullopt, "abc", "def", "hgd"}),
           makeNullableFlatVector<int64_t>({5, 10, std::nullopt, 8}),
       }),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 5)"},
      {makeRowVector({
           makeNullableFlatVector<std::string>(
               {std::nullopt, "abc", "def", "hgd"}),
           makeNullableFlatVector<int64_t>({5, 5, std::nullopt, 8}),
       }),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 5)"},
      {makeRowVector({
           makeNullableFlatVector<std::string>(
               {std::nullopt, "abc", "def", std::nullopt}),
           makeNullableFlatVector<int64_t>({8, 9, std::nullopt, 5}),
       }),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 5)"},
      // Min by non-numeric comparison types.
      {makeRowVector(
           {makeNullableFlatVector<double>({std::nullopt, 10, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "hdg", "def"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 10, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", "hgd", std::nullopt})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 10, 10, std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"hgd", std::nullopt, "wef", "abc"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'abc')"},
      // Min by both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "hdg", "def"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<std::string>({"abc", "abc", "hgd", "def"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"wef", "abc", std::nullopt, std::nullopt}),
            makeNullableFlatVector<std::string>({"hgd", "hgd", "wef", "abc"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => NULL, y => 'abc')"}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto op = PlanBuilder()
                  .values({testData.inputRowVector})
                  .partialAggregation({}, {testData.aggrStatement})
                  .planNode();
    assertQuery(op, testData.verifyDuckDbSql);
  }
}

TEST_F(MinMaxByAggregationTest, minMaxByPartialNoNullsCase) {
  struct {
    const RowVectorPtr inputRowVector;
    const std::string aggrStatement;
    const std::string verifyDuckDbSql;

    const std::string debugString() const {
      return fmt::format(
          "\ninputRowVector: {}\naggrStatement: {}\nverifyDuckDbSql: {}",
          inputRowVector->toString(),
          aggrStatement,
          verifyDuckDbSql);
    }
  } testSettings[] = {
    // Max by both numeric value and comparison types.
    {makeRowVector(
         {makeNullableFlatVector<int64_t>({1, 100, std::nullopt, std::nullopt}),
          makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8})}),
     "max_by(c0, c1)",
     "SELECT struct_pack(x => 1, y => 20)"},
#if 0
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, 100, 20}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 10, y => 10)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, 100, 20}),
            makeNullableFlatVector<int64_t>({10, 11, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 20, y => 11)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, 100, 20}),
            makeNullableFlatVector<int64_t>({10, 11, std::nullopt, 12})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 20, y => 12)"},
      // Max by non-numeric value type.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "def", "hgd"}),
            makeNullableFlatVector<int64_t>({11, 10, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 'abc', y => 11)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "hgd"}),
            makeNullableFlatVector<int64_t>({11, 11, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 'abc', y => 11)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "hgd"}),
            makeNullableFlatVector<int64_t>({11, 12, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 'def', y => 12)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "hgd"}),
            makeNullableFlatVector<int64_t>({11, 12, std::nullopt, 13})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 'hgd', y => 13)"},
      // Max by non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, std::nullopt, 3, 8}),
            makeNullableFlatVector<std::string>(
                {"wef", std::nullopt, "abc", "hgd"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 5, y => 'wef')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, 10, 10, 8}),
            makeNullableFlatVector<std::string>(
                {"wef", "wef", std::nullopt, "hgd"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 5, y => 'wef')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, std::nullopt, 3, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "hgd"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 3, y => 'wef')"},
      {makeRowVector({
           makeNullableFlatVector<int64_t>({5, std::nullopt, 3, 8}),
           makeNullableFlatVector<std::string>(
               {"abc", std::nullopt, "hgd", "wef"}),
       }),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 8, y => 'wef')"},
      // Max by both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "def", "hgd"}),
            makeNullableFlatVector<std::string>(
                {"wef", std::nullopt, "abc", "hgd"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 'abc', y => 'wef')"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>({"abc", "wef", "def", "hgd"}),
            makeNullableFlatVector<std::string>({"wef", "wef", "abc", "hgd"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 'abc', y => 'wef')"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "def", "hgd"}),
            makeNullableFlatVector<std::string>({"abc", "hdg", "wef", "dgh"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 'def', y => 'wef')"},
      // Max by itself.
      {makeRowVector({makeNullableFlatVector<std::string>(
           {"abc", std::nullopt, "def", "hgd"})}),
       "max_by(c0, c0)",
       "SELECT struct_pack(x => 'hgd', y => 'hgd')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, std::nullopt, 3, 8})}),
       "max_by(c0, c0)",
       "SELECT struct_pack(x => 8, y => 8)"},

      // Min by both numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {200, 100, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({5, 10, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 200, y => 5)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, 100, 20}),
            makeNullableFlatVector<int64_t>({5, 5, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 10, y => 5)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, 100, 20}),
            makeNullableFlatVector<int64_t>({10, 5, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 20, y => 5)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, 100, 20}),
            makeNullableFlatVector<int64_t>({10, 11, std::nullopt, 5})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 20, y => 5)"},
      // Min by with non-numeric value type.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "def", "hgd"}),
            makeNullableFlatVector<int64_t>({5, 10, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 'abc', y => 5)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "hgd"}),
            makeNullableFlatVector<int64_t>({5, 5, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 'abc', y => 5)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "hgd"}),
            makeNullableFlatVector<int64_t>({11, 5, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 'def', y => 5)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "hgd"}),
            makeNullableFlatVector<int64_t>({11, 12, std::nullopt, 5})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 'hgd', y => 5)"},
      // Min by with non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, std::nullopt, 3, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "hgd"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 5, y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, 10, 10, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, "hgd"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 5, y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, std::nullopt, 3, 8}),
            makeNullableFlatVector<std::string>(
                {"wef", std::nullopt, "abc", "hgd"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 3, y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, std::nullopt, 3, 8}),
            makeNullableFlatVector<std::string>(
                {"wef", std::nullopt, "hgd", "abc"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 8, y => 'abc')"},
      // Min by both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "def", "hgd"}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "abc", "wef"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 'abc', y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>({"abc", "wef", "def", "hgd"}),
            makeNullableFlatVector<std::string>({"abc", "abc", "wef", "hgd"})}),
       "max_by(c0, c1)",
       "SELECT struct_pack(x => 'def', y => 'wef')"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "def", "hgd"}),
            makeNullableFlatVector<std::string>({"dhg", "hdg", "wef", "abc"})}),
       "min_by(c0, c1)",
       "SELECT struct_pack(x => 'hgd', y => 'abc')"},
      // Min by itself.
      {makeRowVector({makeNullableFlatVector<std::string>(
           {"abc", std::nullopt, "def", "hgd"})}),
       "min_by(c0, c0)",
       "SELECT struct_pack(x => 'abc', y => 'abc')"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({5, std::nullopt, 3, 8})}),
       "min_by(c0, c0)",
       "SELECT struct_pack(x => 3, y => 3)"}
#endif
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto op = PlanBuilder()
                  .values({testData.inputRowVector})
                  .partialAggregation({}, {testData.aggrStatement})
                  .planNode();
    assertQuery(op, testData.verifyDuckDbSql);
  }
}

TEST_F(MinMaxByAggregationTest, minMaxByPartialGroupByNullCase) {
  struct {
    const RowVectorPtr inputRowVector;
    const std::string groupByStatement;
    const std::string aggrStatement;
    const std::string verifyDuckDbSql;

    const std::string debugString() const {
      return fmt::format(
          "\ninputRowVector: {}\ngroupByStatement: {}\naggrStatement: {}\nverifyDuckDbSql: {}",
          inputRowVector->toString(),
          groupByStatement,
          aggrStatement,
          verifyDuckDbSql);
    }
  } testSettings[] = {
      // Max by both numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt,
                 100,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 20}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 20)), (2, struct_pack(x => NULL, y => 8)), (3, struct_pack(x => NULL, y => 10))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, std::nullopt, 20, std::nullopt, 100, 20}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 2, 3, 3, 2, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 20)), (2, struct_pack(x => NULL, y => 10)), (3, struct_pack(x => NULL, y => 8))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt,
                 100,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 20}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 20))) AS t"},
      // Max by with non-numeric value.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt,
                 "abc",
                 "wef",
                 std::nullopt,
                 std::nullopt,
                 "ewd"}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 20)), (2, struct_pack(x => NULL, y => 8)), (3, struct_pack(x => NULL, y => 10))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc",
                 std::nullopt,
                 "wef",
                 std::nullopt,
                 "abc",
                 std::nullopt}),
            makeNullableFlatVector<int64_t>({8, 10, std::nullopt, 20, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 3, 2, 1, 3, 2})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 20)), (2, struct_pack(x => NULL, y => 10)), (3, struct_pack(x => NULL, y => 10))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc",
                 std::nullopt,
                 "wef",
                 std::nullopt,
                 "abc",
                 std::nullopt}),
            makeNullableFlatVector<int64_t>({8, 10, std::nullopt, 20, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 20))) AS t"},
      // Max by with non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {20, std::nullopt, std::nullopt, 8, 10, std::nullopt}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc')), (2, struct_pack(x => NULL, y => 'wef')), (3, struct_pack(x => NULL, y => 'wef'))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {20, 30, std::nullopt, std::nullopt, std::nullopt, 100}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 2, 3, 2, 1, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc')), (2, struct_pack(x => NULL, y => 'wef')), (3, struct_pack(x => NULL, y => 'wef'))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {20, std::nullopt, std::nullopt, 8, 10, std::nullopt}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'wef'))) AS t"},
      // Max by with both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"def",
                 std::nullopt,
                 std::nullopt,
                 "acb",
                 "wef",
                 std::nullopt}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc')), (2, struct_pack(x => NULL, y => 'wef')), (3, struct_pack(x => NULL, y => 'wef'))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc",
                 "def",
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 "wef"}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 2, 3, 2, 1, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc')), (2, struct_pack(x => NULL, y => 'wef')), (3, struct_pack(x => NULL, y => 'wef'))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"def",
                 std::nullopt,
                 std::nullopt,
                 "abc",
                 "wef",
                 std::nullopt}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'wef'))) AS t"},

      // Min by both numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt,
                 100,
                 std::nullopt,
                 std::nullopt,
                 20,
                 std::nullopt}),
            makeNullableFlatVector<int64_t>({10, 20, std::nullopt, 8, 20, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 10)), (2, struct_pack(x => NULL, y => 8)), (3, struct_pack(x => NULL, y => 10))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, std::nullopt, 20, std::nullopt, 100, 20}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 8, 10, 20}),
            makeNullableFlatVector<int32_t>({1, 2, 3, 3, 2, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 10)), (2, struct_pack(x => NULL, y => 10)), (3, struct_pack(x => NULL, y => 8))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt,
                 100,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 20}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 8))) AS t"},
      // Min by with non-numeric value.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc",
                 std::nullopt,
                 "wef",
                 std::nullopt,
                 std::nullopt,
                 "ewd"}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 10)), (2, struct_pack(x => NULL, y => 8)), (3, struct_pack(x => NULL, y => 10))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt,
                 std::nullopt,
                 "wef",
                 "abc",
                 "abc",
                 std::nullopt}),
            makeNullableFlatVector<int64_t>({8, 10, std::nullopt, 20, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 3, 2, 1, 3, 2})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 8)), (2, struct_pack(x => NULL, y => 10)), (3, struct_pack(x => NULL, y => 10))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt,
                 "abc",
                 "wef",
                 std::nullopt,
                 "abc",
                 std::nullopt}),
            makeNullableFlatVector<int64_t>({8, 10, std::nullopt, 20, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 8))) AS t"},
      // Min by with non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {20, std::nullopt, std::nullopt, 8, std::nullopt, 10}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc')), (2, struct_pack(x => NULL, y => 'wef')), (3, struct_pack(x => NULL, y => 'abc'))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {20, std::nullopt, std::nullopt, 30, std::nullopt, 100}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 2, 3, 2, 1, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc')), (2, struct_pack(x => NULL, y => 'abc')), (3, struct_pack(x => NULL, y => 'wef'))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {20, std::nullopt, std::nullopt, 8, 10, std::nullopt}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc'))) AS t"},
      // Min by with both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"def",
                 std::nullopt,
                 std::nullopt,
                 "acb",
                 std::nullopt,
                 "wef"}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc')), (2, struct_pack(x => NULL, y => 'wef')), (3, struct_pack(x => NULL, y => 'abc'))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc",
                 std::nullopt,
                 std::nullopt,
                 "def",
                 std::nullopt,
                 "wef"}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 2, 3, 2, 1, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc')), (2, struct_pack(x => NULL, y => 'abc')), (3, struct_pack(x => NULL, y => 'wef'))) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"def",
                 std::nullopt,
                 std::nullopt,
                 "abc",
                 "wef",
                 std::nullopt}),
            makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "wef", "wef", "abc", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, struct_pack(x => NULL, y => 'abc'))) AS t"}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto op = PlanBuilder()
                  .values({testData.inputRowVector})
                  .partialAggregation(
                      {testData.groupByStatement}, {testData.aggrStatement})
                  .planNode();
    assertQuery(op, testData.verifyDuckDbSql);
    SCOPED_TRACE(testData.debugString());
  }
}

TEST_F(MinMaxByAggregationTest, minMaxByFinalNullCase) {
  struct {
    const RowVectorPtr inputRowVector;
    const std::string aggrStatement;

    const std::string debugString() const {
      return fmt::format(
          "\ninputRowVector: {}\naggrStatement: {}",
          inputRowVector->toString(),
          aggrStatement);
    }
  } testSettings[] = {
      // Max by with numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 8})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {100, std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({9, 10, std::nullopt, 8})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {100, std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({9, 9, std::nullopt, 10})}),
       "max_by(c0, c1)"},
      // Max by with non-numeric value type.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({20, 10, std::nullopt, 8})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({10, 10, std::nullopt, 8})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({9, 10, std::nullopt, 8})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({9, 9, std::nullopt, 10})}),
       "max_by(c0, c1)"},
      // Max by with non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<double>({std::nullopt, 20, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"wef", "def", std::nullopt, "def"})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<double>({std::nullopt, 10, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, std::nullopt})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<double>({10, std::nullopt, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, std::nullopt})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<double>({10, 20, 100, std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "wdf"})}),
       "max_by(c0, c1)"},
      // Max by with both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, std::nullopt})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, std::nullopt})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, std::nullopt})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", "wdf", "def", std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "wdf"})}),
       "max_by(c0, c1)"},

      // Min by with numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({5, 10, std::nullopt, 8})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({5, 5, std::nullopt, 8})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {100, std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({9, 10, std::nullopt, 5})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {100, std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({9, 9, std::nullopt, 10})}),
       "max_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({20, 10, std::nullopt, 8})}),
       "max_by(c0, c1)"},
      // Min by with non-numeric value type.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "cds", "eef"}),
            makeNullableFlatVector<int64_t>({5, 10, std::nullopt, 8})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<int64_t>({5, 5, std::nullopt, 8})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "def", std::nullopt}),
            makeNullableFlatVector<int64_t>({9, 5, std::nullopt, 8})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", std::nullopt}),
            makeNullableFlatVector<double>({9, 9, std::nullopt, 5})}),
       "min_by(c0, c1)"},
      // Max by with non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 20, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "hgd"})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 10, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, "wef"})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {10, std::nullopt, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"wef", "abc", std::nullopt, "wdc"})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, 100, std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"wef", "def", std::nullopt, "abc"})}),
       "min_by(c0, c1)"},
      // Max by with both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"abc", "hgk", std::nullopt, "wef"})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, "hgk"})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"wef", "abc", std::nullopt, "hgf"})}),
       "min_by(c0, c1)"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", "wdf", "def", std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"wef", "def", std::nullopt, "abc"})}),
       "min_by(c0, c1)"}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto op = PlanBuilder()
                  .values({testData.inputRowVector})
                  .partialAggregation({}, {testData.aggrStatement})
                  .finalAggregation()
                  .planNode();
    assertQuery(op, "SELECT NULL");
  }
}

TEST_F(MinMaxByAggregationTest, minMaxByFinalNoNullsCase) {
  struct {
    const RowVectorPtr inputRowVector;
    const std::string aggrStatement;
    const std::string verifyDuckDbSql;

    const std::string debugString() const {
      return fmt::format(
          "\ninputRowVector: {}\naggrStatement: {}\nverifyDuckDbSql: {}",
          inputRowVector->toString(),
          aggrStatement,
          verifyDuckDbSql);
    }
  } testSettings[] = {
      // Max by with numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int32_t>(
                {80, 100, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int32_t>({20, 10, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT 80"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {100, std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT 100"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {100, std::nullopt, std::nullopt, 70}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 20})}),
       "max_by(c0, c1)",
       "SELECT 70"},
      // Max by with non-numeric value type.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"def", "abc", std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({20, 10, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT 'def'"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({10, 10, std::nullopt, 8})}),
       "max_by(c0, c1)",
       "SELECT 'abc'"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, "def"}),
            makeNullableFlatVector<double>({9, 9, std::nullopt, 10})}),
       "max_by(c0, c1)",
       "SELECT 'def'"},
      // Max by with non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<double>({20, 30, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"wef", "def", std::nullopt, "def"})}),
       "max_by(c0, c1)",
       "SELECT 20"},
      {makeRowVector(
           {makeNullableFlatVector<double>({10, std::nullopt, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, std::nullopt})}),
       "max_by(c0, c1)",
       "SELECT 10"},
      {makeRowVector(
           {makeNullableFlatVector<double>({10, 20, 100, 30}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "wdf"})}),
       "max_by(c0, c1)",
       "SELECT 30"},
      // Max by with both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"def", "abc", std::nullopt, "abc"}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, "ab"})}),
       "max_by(c0, c1)",
       "SELECT 'def'"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, std::nullopt})}),
       "max_by(c0, c1)",
       "SELECT 'abc'"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>({"abc", "wdf", "def", "hgd"}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "wdf"})}),
       "max_by(c0, c1)",
       "SELECT 'hgd'"},
      // Self max by.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, std::nullopt, 100})}),
       "max_by(c0, c0)",
       "SELECT 100"},
      {makeRowVector({makeNullableFlatVector<std::string>(
           {"abc", "wdf", std::nullopt, "hgd"})}),
       "max_by(c0, c0)",
       "SELECT 'wdf'"},

      // Min by with numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int32_t>(
                {80, 100, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int32_t>({5, 10, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT 80"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {100, std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 20})}),
       "min_by(c0, c1)",
       "SELECT 100"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {100, std::nullopt, std::nullopt, 70}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 5})}),
       "min_by(c0, c1)",
       "SELECT 70"},
      // Min by with non-numeric value type.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"def", "abc", std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({5, 10, std::nullopt, 8})}),
       "min_by(c0, c1)",
       "SELECT 'def'"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, std::nullopt}),
            makeNullableFlatVector<double>({10, 10, std::nullopt, 20})}),
       "min_by(c0, c1)",
       "SELECT 'abc'"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, "def"}),
            makeNullableFlatVector<double>({9, 9, std::nullopt, 5})}),
       "min_by(c0, c1)",
       "SELECT 'def'"},
      // Min by with non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<double>({20, 30, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "hef", std::nullopt, "def"})}),
       "min_by(c0, c1)",
       "SELECT 20"},
      {makeRowVector(
           {makeNullableFlatVector<double>({10, std::nullopt, std::nullopt, 8}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, "hgd"})}),
       "min_by(c0, c1)",
       "SELECT 10"},
      {makeRowVector(
           {makeNullableFlatVector<double>({10, 20, 100, 30}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "ab"})}),
       "min_by(c0, c1)",
       "SELECT 30"},
      // Min by with both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"def", "abc", std::nullopt, "abc"}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, "hgd"})}),
       "min_by(c0, c1)",
       "SELECT 'def'"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, std::nullopt, "def"}),
            makeNullableFlatVector<std::string>(
                {"abc", "abc", std::nullopt, "hgd"})}),
       "min_by(c0, c1)",
       "SELECT 'abc'"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>({"abc", "wdf", "def", "hgd"}),
            makeNullableFlatVector<std::string>(
                {"abc", "def", std::nullopt, "ab"})}),
       "min_by(c0, c1)",
       "SELECT 'hgd'"},
      // Self min by.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({10, 20, std::nullopt, 100})}),
       "min_by(c0, c0)",
       "SELECT 10"},
      {makeRowVector({makeNullableFlatVector<std::string>(
           {"abc", "wdf", std::nullopt, "hgd"})}),
       "min_by(c0, c0)",
       "SELECT 'abc'"}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto op = PlanBuilder()
                  .values({testData.inputRowVector})
                  .partialAggregation({}, {testData.aggrStatement})
                  .finalAggregation()
                  .planNode();
    assertQuery(op, testData.verifyDuckDbSql);
  }
}

TEST_F(MinMaxByAggregationTest, minMaxByFinalGroupByCase) {
  struct {
    const RowVectorPtr inputRowVector;
    const std::string groupByStatement;
    const std::string aggrStatement;
    const std::string verifyDuckDbSql;

    const std::string debugString() const {
      return fmt::format(
          "\ninputRowVector: {}\ngroupByStatement: {}\naggrStatement: {}\nverifyDuckDbSql: {}",
          inputRowVector->toString(),
          groupByStatement,
          aggrStatement,
          verifyDuckDbSql);
    }
  } testSettings[] = {
      // Max by both numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, 30, std::nullopt, 20}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, NULL), (2, 30), (3, NULL)) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, 30, std::nullopt, 20}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 20, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 30)) AS t"},
      // Max by non-numeric value type.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc",
                 std::nullopt,
                 "wef",
                 std::nullopt,
                 "ewd",
                 std::nullopt}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 'abc'), (2, NULL), (3, 'ewd')) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "hgd", "ewd", std::nullopt}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 25, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 'hgd')) AS t"},
      // Max by non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 30, 10}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "wef", "ewd", std::nullopt}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 20), (2, NULL), (3, 30)) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 30, 10}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "wef", "whd", std::nullopt}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 30)) AS t"},
      // Max by both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "tgi", "wef", "abc", std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "wef", "ewd", std::nullopt}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, NULL), (2, 'tgi'), (3, 'abc')) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "tgi", "wef", "abc", std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "wef", "ewd", std::nullopt}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "max_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 'tgi')) AS t"},

      // Min by both numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, 30, std::nullopt, 20}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 100), (2, 30), (3, NULL)) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>(
                {std::nullopt, 100, std::nullopt, 30, std::nullopt, 20}),
            makeNullableFlatVector<int64_t>({10, 10, std::nullopt, 5, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 30)) AS t"},
      // Min by non-numeric value type.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc",
                 std::nullopt,
                 std::nullopt,
                 "wef",
                 "ewd",
                 std::nullopt}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, NULL), (2, 'wef'), (3, 'ewd')) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "hgd", "ewd", std::nullopt}),
            makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 5, 10, 10}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 'hgd')) AS t"},
      // Min by non-numeric comparison type.
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 30, 10}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "wef", "ewd", "wef"}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 20), (2, NULL), (3, 30)) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<int64_t>({20, 10, std::nullopt, 8, 30, 10}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "wef", "ab", std::nullopt}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 30)) AS t"},
      // Min by both non-numeric value and comparison types.
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "tgi", "wef", "abc", std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "wef", "ewd", std::nullopt}),
            makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, NULL), (2, 'tgi'), (3, 'abc')) AS t"},
      {makeRowVector(
           {makeNullableFlatVector<std::string>(
                {std::nullopt, "abc", "tgi", "wef", "abc", std::nullopt}),
            makeNullableFlatVector<std::string>(
                {"abc", std::nullopt, "wef", "ab", "ewd", std::nullopt}),
            makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1})}),
       "c2",
       "min_by(c0, c1)",
       "SELECT * FROM( VALUES (1, 'wef')) AS t"}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto op = PlanBuilder()
                  .values({testData.inputRowVector})
                  .partialAggregation(
                      {testData.groupByStatement}, {testData.aggrStatement})
                  .finalAggregation()
                  .planNode();
    assertQuery(op, testData.verifyDuckDbSql);
  }
}

TEST_F(MinMaxByAggregationTest, randomMinMaxByGlobalBy) {
  const auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  for (const bool isMinBy : {false, true}) {
    for (const auto& columnName : rowType_->names()) {
      testGlobalAggregation(vectors, isMinBy ? kMinBy : kMaxBy, columnName);
    }
  }
}

TEST_F(MinMaxByAggregationTest, randomMinMaxByGroupBy) {
  const auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  for (const bool isMinBy : {false, true}) {
    for (const auto& columnName : rowType_->names()) {
      testGroupByAggregation(vectors, isMinBy ? kMinBy : kMaxBy, columnName);
    }
  }
}

TEST_F(MinMaxByAggregationTest, maxByGroupByWithNumericTypes) {
  const size_t size = 1'000;
  auto vectors = {makeRowVector(
      {makeFlatVector<double>(
           size, [](vector_size_t row) { return row * 0.1; }),
       makeFlatVector<int64_t>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<int32_t>(
           size, [](vector_size_t row) { return row % 10; })})};
  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .partialAggregation({"c2"}, {"max_by(c0, c1)"})
                .finalAggregation()
                .planNode();
  assertQuery(
      op, "SELECT c2, max(CAST(c1 as DOUBLE)) * 0.1 FROM tmp GROUP BY 1");
}

TEST_F(MinMaxByAggregationTest, minByGroupByWithNumericTypes) {
  const size_t size = 1'000;
  auto vectors = {makeRowVector(
      {makeFlatVector<double>(
           size, [](vector_size_t row) { return row * 0.1; }),
       makeFlatVector<int64_t>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<int32_t>(
           size, [](vector_size_t row) { return row % 10; })})};
  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .partialAggregation({"c2"}, {"min_by(c0, c1)"})
                .finalAggregation()
                .planNode();
  assertQuery(
      op, "SELECT c2, min(CAST(c1 as DOUBLE)) * 0.1 FROM tmp GROUP BY 1");
}

} // namespace
} // namespace facebook::velox::aggregate::test
