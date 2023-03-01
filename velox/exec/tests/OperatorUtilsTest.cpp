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
#include "velox/exec/OperatorUtils.h"
#include <gtest/gtest.h>
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Operator.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;

class OperatorUtilsTest
    : public ::facebook::velox::exec::test::OperatorTestBase {
 protected:
  void gatherCopyTest(
      const std::shared_ptr<const RowType>& targetType,
      const std::shared_ptr<const RowType>& sourceType,
      int numSources) {
    folly::Random::DefaultGenerator rng(1);
    const int kNumRows = 500;
    const int kNumColumns = sourceType->size();

    // Build source vectors with nulls.
    std::vector<RowVectorPtr> sources;
    for (int i = 0; i < numSources; ++i) {
      sources.push_back(std::static_pointer_cast<RowVector>(
          BatchMaker::createBatch(sourceType, kNumRows, *pool_)));
      for (int j = 0; j < kNumColumns; ++j) {
        auto vector = sources.back()->childAt(j);
        int nullRow = (folly::Random::rand32() % kNumRows) / 4;
        while (nullRow < kNumRows) {
          vector->setNull(nullRow, true);
          nullRow +=
              std::max<int>(1, (folly::Random::rand32() % kNumColumns) / 4);
        }
      }
    }

    std::vector<IdentityProjection> columnMap;
    if (sourceType != targetType) {
      for (column_index_t sourceChannel = 0; sourceChannel < kNumColumns;
           ++sourceChannel) {
        const auto columnName = sourceType->nameOf(sourceChannel);
        const column_index_t targetChannel =
            targetType->getChildIdx(columnName);
        columnMap.emplace_back(sourceChannel, targetChannel);
      }
    }

    std::vector<const RowVector*> sourcesVectors(kNumRows);
    std::vector<vector_size_t> sourceIndices(kNumRows);
    for (int iter = 0; iter < 5; ++iter) {
      const int count =
          folly::Random::oneIn(10) ? 0 : folly::Random::rand32() % kNumRows;
      const int targetIndex = folly::Random::rand32() % (kNumRows - count);
      for (int i = 0; i < count; ++i) {
        sourcesVectors[i] = sources[folly::Random::rand32() % numSources].get();
        sourceIndices[i] = sourceIndices[folly::Random::rand32() % kNumRows];
      }
      auto targetVector =
          BaseVector::create<RowVector>(targetType, kNumRows, pool_.get());
      for (int32_t childIdx = 0; childIdx < targetVector->childrenSize();
           ++childIdx) {
        targetVector->childAt(childIdx)->resize(kNumRows);
      }
      gatherCopy(
          targetVector.get(),
          targetIndex,
          count,
          sourcesVectors,
          sourceIndices,
          columnMap);

      // Verify the copied data in target.
      for (int i = 0; i < kNumColumns; ++i) {
        const column_index_t sourceColumnChannel =
            columnMap.empty() ? i : columnMap[i].inputChannel;
        const column_index_t targetColumnChannel =
            columnMap.empty() ? i : columnMap[i].outputChannel;
        auto vector = targetVector->childAt(targetColumnChannel);
        for (int j = 0; j < count; ++j) {
          auto source = sourcesVectors[j]->childAt(sourceColumnChannel).get();
          if (vector->isNullAt(targetIndex + j)) {
            ASSERT_TRUE(source->isNullAt(sourceIndices[j]));
          } else {
            ASSERT_TRUE(vector->equalValueAt(
                source, targetIndex + j, sourceIndices[j]));
          }
        }
      }
    }
  }

  std::shared_ptr<memory::MemoryPool> pool_{memory::getDefaultMemoryPool()};
};

TEST_F(OperatorUtilsTest, wrapChildConstant) {
  auto constant = makeConstant(11, 1'000);

  BufferPtr mapping = allocateIndices(1'234, pool_.get());
  auto rawMapping = mapping->asMutable<vector_size_t>();
  for (auto i = 0; i < 1'234; i++) {
    rawMapping[i] = i / 2;
  }

  auto wrapped = exec::wrapChild(1'234, mapping, constant);
  ASSERT_EQ(wrapped->size(), 1'234);
  ASSERT_TRUE(wrapped->isConstantEncoding());
  ASSERT_TRUE(wrapped->equalValueAt(constant.get(), 100, 100));
}

TEST_F(OperatorUtilsTest, gatherCopy) {
  std::shared_ptr<const RowType> rowType;
  std::shared_ptr<const RowType> reversedRowType;
  {
    std::vector<std::string> names = {
        "bool_val",
        "tiny_val",
        "small_val",
        "int_val",
        "long_val",
        "ordinal",
        "float_val",
        "double_val",
        "string_val",
        "array_val",
        "struct_val",
        "map_val"};
    std::vector<std::string> reversedNames = names;
    std::reverse(reversedNames.begin(), reversedNames.end());

    std::vector<std::shared_ptr<const Type>> types = {
        BOOLEAN(),
        TINYINT(),
        SMALLINT(),
        INTEGER(),
        BIGINT(),
        BIGINT(),
        REAL(),
        DOUBLE(),
        VARCHAR(),
        ARRAY(VARCHAR()),
        ROW({{"s_int", INTEGER()}, {"s_array", ARRAY(REAL())}}),
        MAP(VARCHAR(),
            MAP(BIGINT(),
                ROW({{"s2_int", INTEGER()}, {"s2_string", VARCHAR()}})))};
    std::vector<std::shared_ptr<const Type>> reversedTypes = types;
    std::reverse(reversedTypes.begin(), reversedTypes.end());

    rowType = ROW(std::move(names), std::move(types));
    reversedRowType = ROW(std::move(reversedNames), std::move(reversedTypes));
  }

  // Gather copy with identical column mapping.
  gatherCopyTest(rowType, rowType, 1);
  gatherCopyTest(rowType, rowType, 5);
  // Gather copy with non-identical column mapping.
  gatherCopyTest(rowType, reversedRowType, 1);
  gatherCopyTest(rowType, reversedRowType, 5);

  // Test with UNKNOWN type.
  int kNumRows = 100;
  auto sourceVector = makeRowVector({
      makeFlatVector<int64_t>(kNumRows, [](auto row) { return row % 7; }),
      BaseVector::createNullConstant(UNKNOWN(), kNumRows, pool()),
  });
  std::vector<const RowVector*> sourceVectors(kNumRows);
  std::vector<vector_size_t> sourceIndices(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    sourceVectors[i] = sourceVector.get();
    sourceIndices[i] = kNumRows - i - 1;
  }
  auto targetVector = BaseVector::create<RowVector>(
      sourceVector->type(), kNumRows, pool_.get());
  for (int32_t childIdx = 0; childIdx < targetVector->childrenSize();
       ++childIdx) {
    targetVector->childAt(childIdx)->resize(kNumRows);
  }

  gatherCopy(targetVector.get(), 0, kNumRows, sourceVectors, sourceIndices);
  // Verify the copied data in target.
  for (int i = 0; i < targetVector->type()->size(); ++i) {
    auto vector = targetVector->childAt(i);
    for (int j = 0; j < kNumRows; ++j) {
      auto source = sourceVectors[j]->childAt(i).get();
      ASSERT_TRUE(vector->equalValueAt(source, j, sourceIndices[j]));
    }
  }
}

TEST_F(OperatorUtilsTest, makeOperatorSpillPath) {
  EXPECT_EQ("spill/3_1_100", makeOperatorSpillPath("spill", 3, 1, 100));
}

TEST_F(OperatorUtilsTest, wrap) {
  auto rowType = ROW({
      {"bool_val", BOOLEAN()},
      {"tiny_val", TINYINT()},
      {"small_val", SMALLINT()},
      {"int_val", INTEGER()},
      {"long_val", BIGINT()},
      {"ordinal", BIGINT()},
      {"float_val", REAL()},
      {"double_val", DOUBLE()},
      {"string_val", VARCHAR()},
      {"array_val", ARRAY(VARCHAR())},
      {"struct_val", ROW({{"s_int", INTEGER()}, {"s_array", ARRAY(REAL())}})},
      {"map_val",
       MAP(VARCHAR(),
           MAP(BIGINT(),
               ROW({{"s2_int", INTEGER()}, {"s2_string", VARCHAR()}})))},
  });

  VectorFuzzer fuzzer({}, pool());
  auto vector = fuzzer.fuzzFlat(rowType);
  auto rowVector = vector->as<RowVector>();

  for (int32_t iter = 0; iter < 20; ++iter) {
    folly::Random::DefaultGenerator rng;
    rng.seed(iter);
    const int32_t wrapVectorSize =
        iter == 0 ? 0 : 1 + folly::Random().rand32(2 * rowVector->size(), rng);
    BufferPtr wrapIndices =
        makeIndices(wrapVectorSize, [&](vector_size_t /* unused */) {
          return folly::Random().rand32(rowVector->size(), rng);
        });
    auto* rawWrapIndices = wrapIndices->as<vector_size_t>();

    auto wrapVector = wrap(
        wrapVectorSize, wrapIndices, rowType, rowVector->children(), pool());
    ASSERT_EQ(wrapVector->size(), wrapVectorSize);
    for (int32_t i = 0; i < wrapVectorSize; ++i) {
      wrapVector->equalValueAt(vector.get(), i, rawWrapIndices[i]);
    }

    wrapVector =
        wrap(wrapVectorSize, nullptr, rowType, rowVector->children(), pool());
    ASSERT_EQ(wrapVector->size(), 0);
  }
}

TEST_F(OperatorUtilsTest, addOperatorRuntimeStats) {
  std::unordered_map<std::string, RuntimeMetric> stats;
  const std::string statsName("stats");
  const RuntimeCounter minStatsValue(100, RuntimeCounter::Unit::kBytes);
  const RuntimeCounter maxStatsValue(200, RuntimeCounter::Unit::kBytes);
  addOperatorRuntimeStats(statsName, minStatsValue, stats);
  ASSERT_EQ(stats[statsName].count, 1);
  ASSERT_EQ(stats[statsName].sum, 100);
  ASSERT_EQ(stats[statsName].max, 100);
  ASSERT_EQ(stats[statsName].min, 100);

  addOperatorRuntimeStats(statsName, maxStatsValue, stats);
  ASSERT_EQ(stats[statsName].count, 2);
  ASSERT_EQ(stats[statsName].sum, 300);
  ASSERT_EQ(stats[statsName].max, 200);
  ASSERT_EQ(stats[statsName].min, 100);

  addOperatorRuntimeStats(statsName, maxStatsValue, stats);
  ASSERT_EQ(stats[statsName].count, 3);
  ASSERT_EQ(stats[statsName].sum, 500);
  ASSERT_EQ(stats[statsName].max, 200);
  ASSERT_EQ(stats[statsName].min, 100);
}
