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
#include "velox/connectors/paimon/PaimonMergeFunction.h"

#include <gtest/gtest.h>

#include <numeric>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::connector::paimon {
namespace {

using FieldAggregation = PaimonTableHandle::MergeOptions::FieldAggregation;

class PaimonMergeFunctionTest : public testing::Test,
                                public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  static PaimonTableHandle::MergeOptions aggregationOptions(
      std::unordered_map<std::string, FieldAggregation> fieldAggregations,
      bool removeRecordOnDelete) {
    PaimonTableHandle::MergeOptions options;
    options.mergeEngine = PaimonMergeEngine::kAggregation;
    options.fieldAggregations = std::move(fieldAggregations);
    options.aggregationRemoveRecordOnDelete = removeRecordOnDelete;
    return options;
  }

  // Merges the rows of 'input', of kinds 'kinds', with the merge function of
  // 'options' over all columns. Returns the kind of the merged row and the
  // row.
  std::pair<PaimonRowKind, RowVectorPtr> merge(
      const RowVectorPtr& input,
      const std::vector<PaimonRowKind>& kinds,
      const PaimonTableHandle::MergeOptions& options) {
    VELOX_CHECK_EQ(kinds.size(), input->size());
    std::vector<column_index_t> channels(input->childrenSize());
    std::iota(channels.begin(), channels.end(), 0);
    const auto mergeFunction = PaimonMergeFunction::create(
        asRowType(input->type()), channels, options);
    mergeFunction->reset();
    for (vector_size_t row = 0; row < input->size(); ++row) {
      mergeFunction->add(*input, row, kinds[row]);
    }
    auto output = BaseVector::create<RowVector>(input->type(), 1, pool());
    mergeFunction->write(0, *output);
    return {mergeFunction->resultKind(), std::move(output)};
  }
};

// Paimon Java's AggregateMergeFunction aggregates each column in merge order
// and retracts DELETE and UPDATE_BEFORE rows. With
// 'aggregation.remove-record-on-delete', a DELETE row instead replaces the
// merged row with its values, and the key returns DELETE when that row comes
// last.
TEST_F(PaimonMergeFunctionTest, aggregation) {
  const auto input = makeRowVector(
      {"id", "total", "last"},
      {makeFlatVector<int64_t>({1, 1, 1}),
       makeFlatVector<int64_t>({10, 7, 20}),
       makeNullableFlatVector<int64_t>({100, 70, std::nullopt})});
  const std::unordered_map<std::string, FieldAggregation> fieldAggregations{
      {"id", {"primary-key", false}},
      {"total", {"sum", false}},
      {"last", {"last_non_null_value", false}},
  };
  constexpr auto kInsert = PaimonRowKind::kInsert;
  constexpr auto kUpdateAfter = PaimonRowKind::kUpdateAfter;
  constexpr auto kUpdateBefore = PaimonRowKind::kUpdateBefore;
  constexpr auto kDelete = PaimonRowKind::kDelete;
  struct TestParam {
    std::string testName;
    std::vector<PaimonRowKind> kinds;
    bool removeRecordOnDelete;
    PaimonRowKind expectedKind;
    int64_t expectedTotal;
    std::optional<int64_t> expectedLast;

    std::string debugString() const {
      return fmt::format(
          "{}, removeRecordOnDelete {}", testName, removeRecordOnDelete);
    }
  };
  const std::vector<TestParam> testSettings{
      {"inserts", {kInsert, kUpdateAfter, kInsert}, false, kInsert, 37, 70},
      {"update before retracts",
       {kInsert, kUpdateBefore, kInsert},
       false,
       kInsert,
       23,
       std::nullopt},
      {"delete retracts", {kInsert, kDelete, kInsert}, false, kInsert, 23, {}},
      {"last delete retracts",
       {kInsert, kInsert, kDelete},
       false,
       kInsert,
       -3,
       70},
      {"delete replaces the row",
       {kInsert, kDelete, kInsert},
       true,
       kInsert,
       27,
       70},
      {"last delete removes the key",
       {kInsert, kInsert, kDelete},
       true,
       kDelete,
       20,
       std::nullopt},
      {"update before still retracts",
       {kInsert, kUpdateBefore, kInsert},
       true,
       kInsert,
       23,
       std::nullopt},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto [kind, row] = merge(
        input,
        testData.kinds,
        aggregationOptions(fieldAggregations, testData.removeRecordOnDelete));
    EXPECT_EQ(kind, testData.expectedKind);
    test::assertEqualVectors(
        makeRowVector(
            {"id", "total", "last"},
            {makeFlatVector<int64_t>({1}),
             makeFlatVector<int64_t>({testData.expectedTotal}),
             makeNullableFlatVector<int64_t>({testData.expectedLast})}),
        row);
  }
}

// A DELETE row that replaces the merged row keeps the state of the functions,
// so 'first_value' keeps the value of the DELETE row, as in Paimon Java.
TEST_F(PaimonMergeFunctionTest, aggregationDeleteKeepsState) {
  const auto input = makeRowVector(
      {"id", "first"},
      {makeFlatVector<int64_t>({1, 1, 1}),
       makeFlatVector<int64_t>({100, 50, 200})});
  const auto [kind, row] = merge(
      input,
      {PaimonRowKind::kInsert, PaimonRowKind::kDelete, PaimonRowKind::kInsert},
      aggregationOptions(
          {{"id", {"primary-key", false}}, {"first", {"first_value", false}}},
          /*removeRecordOnDelete=*/true));
  EXPECT_EQ(kind, PaimonRowKind::kInsert);
  test::assertEqualVectors(
      makeRowVector(
          {"id", "first"},
          {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({50})}),
      row);
}

// A reader merges every key of a split with one merge function, which reset()
// clears between keys. Its channels leave out _SEQUENCE_NUMBER and
// _VALUE_KIND, which write() leaves alone.
TEST_F(PaimonMergeFunctionTest, aggregationAcrossKeys) {
  const auto input = makeRowVector(
      {"id", "_SEQUENCE_NUMBER", "total", "_VALUE_KIND", "first"},
      {makeFlatVector<int64_t>({1, 1, 2, 2}),
       makeFlatVector<int64_t>({0, 1, 2, 3}),
       makeFlatVector<int64_t>({10, 5, 7, 3}),
       makeFlatVector<int8_t>({0, 3, 0, 0}),
       makeFlatVector<int64_t>({100, 50, 70, 30})});
  const auto mergeFunction = PaimonMergeFunction::create(
      asRowType(input->type()),
      {0, 2, 4},
      aggregationOptions(
          {{"id", {"primary-key", false}},
           {"total", {"sum", false}},
           {"first", {"first_value", false}}},
          /*removeRecordOnDelete=*/true));
  auto output = BaseVector::create<RowVector>(input->type(), 2, pool());
  output->childAt(1)->asFlatVector<int64_t>()->set(0, -1);
  output->childAt(3)->asFlatVector<int8_t>()->set(0, -1);

  mergeFunction->reset();
  mergeFunction->add(*input, 0, PaimonRowKind::kInsert);
  mergeFunction->add(*input, 1, PaimonRowKind::kDelete);
  EXPECT_EQ(mergeFunction->resultKind(), PaimonRowKind::kDelete);
  mergeFunction->write(0, *output);

  // Without reset(), 'total' would be 15 and 'first' 50.
  mergeFunction->reset();
  mergeFunction->add(*input, 2, PaimonRowKind::kInsert);
  mergeFunction->add(*input, 3, PaimonRowKind::kInsert);
  EXPECT_EQ(mergeFunction->resultKind(), PaimonRowKind::kInsert);
  mergeFunction->write(1, *output);

  test::assertEqualVectors(makeFlatVector<int64_t>({1, 2}), output->childAt(0));
  test::assertEqualVectors(
      makeFlatVector<int64_t>({5, 10}), output->childAt(2));
  test::assertEqualVectors(
      makeFlatVector<int64_t>({50, 70}), output->childAt(4));
  EXPECT_EQ(output->childAt(1)->asFlatVector<int64_t>()->valueAt(0), -1);
  EXPECT_EQ(output->childAt(3)->asFlatVector<int8_t>()->valueAt(0), -1);
}

TEST_F(PaimonMergeFunctionTest, aggregationOptionConflict) {
  const auto type = ROW({"id", "total"}, BIGINT());
  VELOX_ASSERT_USER_THROW(
      PaimonMergeFunction::create(
          type,
          {0, 1},
          aggregationOptions(
              {{"id", {"primary-key", false}}, {"total", {"sum", true}}},
              /*removeRecordOnDelete=*/true)),
      "Paimon table options 'aggregation.remove-record-on-delete' and "
      "'fields.total.ignore-retract' cannot both be true");
  // Columns the read leaves out are not checked, as in Paimon Java.
  PaimonMergeFunction::create(
      type,
      {0},
      aggregationOptions(
          {{"id", {"primary-key", false}}, {"total", {"sum", true}}},
          /*removeRecordOnDelete=*/true));
}

// Only the aggregation engine merges by column. Deduplicate and first-row pick
// whole rows in PaimonMergeSplitReader.
TEST_F(PaimonMergeFunctionTest, unsupportedMergeEngines) {
  for (const auto mergeEngine :
       {PaimonMergeEngine::kDeduplicate,
        PaimonMergeEngine::kPartialUpdate,
        PaimonMergeEngine::kFirstRow}) {
    SCOPED_TRACE(fmt::format("mergeEngine {}", mergeEngine));
    PaimonTableHandle::MergeOptions options;
    options.mergeEngine = mergeEngine;
    VELOX_ASSERT_THROW(
        PaimonMergeFunction::create(ROW("id", BIGINT()), {0}, options),
        fmt::format(
            "Paimon merge function does not support the merge engine: {}",
            mergeEngine));
  }
}

} // namespace
} // namespace facebook::velox::connector::paimon
