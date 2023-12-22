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
#include "velox/common/hyperloglog/HllUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {
namespace {
class ApproxDistinctTest : public AggregationTestBase {
 protected:
  static const std::vector<std::string> kFruits;
  static const std::vector<std::string> kVegetables;

  void testGlobalAgg(
      const VectorPtr& values,
      double maxStandardError,
      int64_t expectedResult,
      bool testWithTableScan = true) {
    auto vectors = makeRowVector({values});
    auto expected =
        makeRowVector({makeNullableFlatVector<int64_t>({expectedResult})});

    testAggregations(
        {vectors},
        {},
        {fmt::format("approx_distinct(c0, {})", maxStandardError)},
        {expected},
        {},
        testWithTableScan);
    testAggregationsWithCompanion(
        {vectors},
        [](auto& /*builder*/) {},
        {},
        {fmt::format("approx_distinct(c0, {})", maxStandardError)},
        {{values->type(), DOUBLE()}},
        {},
        {expected});

    testAggregations(
        {vectors},
        {},
        {fmt::format("approx_set(c0, {})", maxStandardError)},
        {"cardinality(a0)"},
        {expected},
        {},
        testWithTableScan);
  }

  void testGlobalAgg(
      const VectorPtr& values,
      int64_t expectedResult,
      bool testWithTableScan = true) {
    auto vectors = makeRowVector({values});
    auto expected =
        makeRowVector({makeNullableFlatVector<int64_t>({expectedResult})});

    testAggregations(
        {vectors},
        {},
        {"approx_distinct(c0)"},
        {expected},
        {},
        testWithTableScan);
    testAggregationsWithCompanion(
        {vectors},
        [](auto& /*builder*/) {},
        {},
        {"approx_distinct(c0)"},
        {{values->type()}},
        {},
        {expected});

    testAggregations(
        {vectors},
        {},
        {"approx_set(c0)"},
        {"cardinality(a0)"},
        {expected},
        {},
        testWithTableScan);
  }

  template <typename T, typename U>
  RowVectorPtr toRowVector(const std::unordered_map<T, U>& data) {
    std::vector<T> keys(data.size());
    transform(data.begin(), data.end(), keys.begin(), [](auto pair) {
      return pair.first;
    });

    std::vector<U> values(data.size());
    transform(data.begin(), data.end(), values.begin(), [](auto pair) {
      return pair.second;
    });

    return makeRowVector({makeFlatVector(keys), makeFlatVector(values)});
  }

  void testGroupByAgg(
      const VectorPtr& keys,
      const VectorPtr& values,
      const std::unordered_map<int32_t, int64_t>& expectedResults) {
    auto vectors = makeRowVector({keys, values});
    auto expected = toRowVector(expectedResults);

    testAggregations({vectors}, {"c0"}, {"approx_distinct(c1)"}, {expected});
    testAggregationsWithCompanion(
        {vectors},
        [](auto& /*builder*/) {},
        {"c0"},
        {"approx_distinct(c1)"},
        {{values->type()}},
        {},
        {expected});

    testAggregations(
        {vectors},
        {"c0"},
        {"approx_set(c1)"},
        {"c0", "cardinality(a0)"},
        {expected});
  }
};

const std::vector<std::string> ApproxDistinctTest::kFruits = {
    "apple",
    "banana",
    "cherry",
    "dragonfruit",
    "grapefruit",
    "melon",
    "orange",
    "pear",
    "pineapple",
    "unknown fruit with a very long name",
    "watermelon"};

const std::vector<std::string> ApproxDistinctTest::kVegetables = {
    "cucumber",
    "tomato",
    "potato",
    "squash",
    "unknown vegetable with a very long name"};

TEST_F(ApproxDistinctTest, groupByIntegers) {
  vector_size_t size = 1'000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 2; });
  auto values = makeFlatVector<int32_t>(
      size, [](auto row) { return row % 2 == 0 ? row % 17 : row % 21 + 100; });

  testGroupByAgg(keys, values, {{0, 17}, {1, 21}});
}

TEST_F(ApproxDistinctTest, groupByStrings) {
  vector_size_t size = 1'000;

  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 2; });
  auto values = makeFlatVector<StringView>(size, [&](auto row) {
    return StringView(
        row % 2 == 0 ? kFruits[row % kFruits.size()]
                     : kVegetables[row % kVegetables.size()]);
  });

  testGroupByAgg(keys, values, {{0, kFruits.size()}, {1, kVegetables.size()}});
}

TEST_F(ApproxDistinctTest, groupByHighCardinalityIntegers) {
  vector_size_t size = 1'000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 2; });
  auto values = makeFlatVector<int32_t>(size, [](auto row) { return row; });

  testGroupByAgg(keys, values, {{0, 488}, {1, 493}});
}

TEST_F(ApproxDistinctTest, groupByVeryLowCardinalityIntegers) {
  vector_size_t size = 1'000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 2; });
  auto values = makeFlatVector<int32_t>(
      size, [](auto row) { return row % 2 == 0 ? 27 : row % 3; });

  testGroupByAgg(keys, values, {{0, 1}, {1, 3}});
}

TEST_F(ApproxDistinctTest, groupByAllNulls) {
  vector_size_t size = 1'000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 2; });
  auto values = makeFlatVector<int32_t>(
      size, [](auto row) { return row % 2 == 0 ? 27 : row % 3; }, nullEvery(2));

  auto vectors = makeRowVector({keys, values});
  auto expected = toRowVector<int32_t, int64_t>({{0, 0}, {1, 3}});

  testAggregations({vectors}, {"c0"}, {"approx_distinct(c1)"}, {expected});
  testAggregationsWithCompanion(
      {vectors},
      [](auto& /*builder*/) {},
      {"c0"},
      {"approx_distinct(c1)"},
      {{values->type()}},
      {},
      {expected});
}

TEST_F(ApproxDistinctTest, globalAggIntegers) {
  vector_size_t size = 1'000;
  auto values =
      makeFlatVector<int32_t>(size, [](auto row) { return row % 17; });

  testGlobalAgg(values, 17);
}

TEST_F(ApproxDistinctTest, globalAggStrings) {
  vector_size_t size = 1'000;

  auto values = makeFlatVector<StringView>(size, [&](auto row) {
    return StringView(kFruits[row % kFruits.size()]);
  });

  testGlobalAgg(values, kFruits.size());
}

TEST_F(ApproxDistinctTest, globalAggTimeStamp) {
  auto mills = makeFlatVector<int64_t>(
      {905746043000,  215588628000,  804489195000,  123780566000,
       1547661049000, 1474116657000, 676849069000,  242395458000,
       1663259494000, 6172523000,    1502828344000, 1165264382000,
       389916048000,  1489969255000, 590028214000,  1530268049000,
       1442140895000, 1101068119000, 18564324000,   1348580750000,
       987304073000,  846890662000,  1086013123000, 673360407000,
       711313738000,  1505910931000, 1277164609000, 776157290000,
       1048000710000, 660705326000,  505691714000,  961828890000,
       537569964000,  1647153315000, 1108509062000, 520390042000,
       1562168178000, 228138641000,  216429757000,  144986243000,
       180040370000,  110553666000,  1454402814000, 1486942531000,
       997177933000,  871037064000,  308541818000,  711722272000,
       986972089000,  215701966000,  152925788000,  366321659000,
       113202996000,  1247813013000, 546682678000,  1037620761000,
       1259331611000, 346821716000,  448349379000,  856678905000,
       1588465889000, 27989033000,   1039987365000, 38011096000,
       548543874000,  1404831961000, 975542822000,  959676592000,
       1608754451000, 267101901000,  213545088000,  411402585000,
       1289797454000, 719489171000,  1368220405000, 1616706430000,
       828203515000,  1570089093000, 1628274222000, 485133624000,
       27427879000,   990773112000,  1459971704000, 1028681149000,
       739003497000,  1118473283000, 1565971490000, 1068799967000,
       1146617959000, 1618569154000, 665063458000,  1097264699000,
       508254755000,  854529091000,  1219877122000, 1242473274000,
       994825085000,  139569085000,  1526186858000, 1171040845000,
       818924697000,  821791397000,  348825393000,  800751100000,
       993949416000,  1606702195000, 204924001000,  316434687000,
       1161599775000, 1556431645000, 1118134980000, 370288917000,
       68683355000,   194139295000,  805980244000,  818881098000,
       360451089000,  1208001474000, 52798147000,   598388005000,
       1697617591000, 754341415000,  359118509000,  1050351834000,
       669223073000,  593933215000,  376515894000,  1234323405000,
       636220801000,  844276664000,  468877292000,  1231620784000,
       1569192080000, 1134676627000, 1544360014000, 1276701796000,
       1589856582000, 814313893000,  1680863679000, 739917942000,
       765373060000,  1053592948000, 591298237000,  152894038000,
       122274444000,  623608432000,  1339595245000, 572295711000,
       1321711446000, 590461652000,  1664387047000, 428889160000,
       1377810728000, 278253551000,  961302697000,  1680351042000,
       432479133000,  305100482000,  121681876000,  162676687000,
       1386185849000, 296454414000,  1085901027000, 476072718000,
       1440824721000, 361407928000,  634185558000,  1053217842000,
       690789996000,  594467035000,  479000962000,  1370697389000,
       534973573000,  573382020000,  92837967000,   1316242771000,
       1230911977000, 493448656000,  996693522000,  624579281000,
       1183702549000, 1485447212000, 1018416963000, 1697783451000,
       565498027000,  303127780000,  663167997000,  1452032688000,
       85813656000,   1240964020000, 1017695886000, 709395319000,
       1685280659000, 90614567000,   80436851000,   238123582000,
       952174127000,  1233525955000, 581688080000,  659899384000,
       1689732617000, 652322190000,  1110302747000, 1118499653000,
       613260577000,  532033743000,  1574560991000, 712631959000,
       1557403197000, 1192713312000, 1301296158000, 244186605000,
       1044350190000, 181831942000,  1290912745000, 822040398000,
       1304588205000, 1662835346000, 1515325990000, 1571678450000,
       1614670680000, 36245749000,   604608314000,  994733406000,
       457432767000,  928114984000,  1301598474000, 1401630896000,
       333023883000,  536581422000,  1492901206000, 1073520903000,
       867502031000,  139580427000,  1020543630000, 548771218000,
       14093829000,   355143596000,  1072109805000, 1666000817000,
       948929112000,  795580571000,  845239387000,  1457143076000,
       1131797345000, 689576653000,  718274990000,  58358977000,
       189340222000,  1020981950000, 56573081000,   1170279140000,
       341385133000,  1016435808000, 392540472000,  1473363502000,
       1503271797000, 742883730000,  1126268774000, 102760974000,
       6724123000,    1128844321000, 364765074000,  640712296000,
       1271970136000, 1376912164000, 904079677000,  155738804000,
       938159854000,  754081924000,  170977612000,  1245648272000,
       594199549000,  93928477000,   383693436000,  49327638000,
       1240835868000, 1472760961000, 810811291000,  39680907000,
       1306166790000, 1132650540000, 1074308364000, 264514521000,
       122149830000,  502527729000,  789859647000,  1257340037000,
       902424944000,  495109640000,  641843161000,  814744535000,
       1654231877000, 1093485783000, 256026031000,  1646443316000,
       864493601000,  888890381000,  313820154000,  140549411000,
       264514521000,  14093829000,   814744535000,  711313738000,
       1459971704000, 1304588205000, 348825393000,  952174127000,
       1290912745000, 139569085000,  376515894000,  1290912745000,
       1526186858000, 1473363502000, 814313893000,  1565971490000,
       1664387047000, 194139295000,  1208001474000, 1376912164000,
       624579281000,  814744535000,  1028681149000, 1368220405000,
       305100482000,  1618569154000, 1128844321000, 593933215000,
       392540472000,  1557403197000, 1097264699000, 623608432000,
       392540472000,  1557403197000, 814744535000,  1017695886000,
       1503271797000, 305100482000,  1588465889000, 1530268049000,
       305100482000,  1557403197000, 93928477000,   476072718000,
       1569192080000, 493448656000,  1440824721000, 1662835346000,
       305100482000,  305100482000,  1485447212000, 1072109805000,
       1108509062000, 256026031000,  624579281000,  789859647000,
       181831942000,  181831942000,  1440824721000, 1503271797000,
       994825085000,  1037620761000, 123780566000,  411402585000,
       376515894000,  1183702549000, 110553666000,  1068799967000,
       1565971490000, 256026031000,  110553666000,  536581422000,
       1662835346000, 1321711446000, 573382020000,  548771218000,
       845239387000,  1028681149000, 1473363502000, 476072718000,
       244186605000,  565498027000,  1473363502000, 6724123000,
       1108509062000, 355143596000,  1277164609000, 1530268049000,
       1473363502000, 495109640000,  536581422000,  14093829000,
       624579281000,  1565971490000, 85813656000,   1440824721000,
       14093829000,   155738804000,  1230911977000, 613260577000,
       493448656000,  905746043000,  1662835346000, 613260577000,
       1085901027000, 1037620761000, 1085901027000, 814744535000,
       1697617591000, 1053217842000, 1588465889000, 1459971704000,
       305100482000,  313820154000,  1680863679000, 1074308364000,
       1321711446000, 659899384000,  613260577000,  162676687000,
       1685280659000, 1289797454000, 123780566000,  264514521000,
       1370697389000, 457432767000,  1680351042000, 110553666000,
       170977612000,  14093829000,   1697617591000, 457432767000,
       122149830000,  1271970136000, 242395458000,  948929112000,
       194139295000,  1565971490000, 1128844321000, 1233525955000,
       1502828344000, 1454402814000, 361407928000,  1044350190000,
       305100482000,  1037620761000, 1588465889000, 432479133000,
       1259331611000, 493448656000});

  auto timestamp = makeFlatVector<facebook::velox::Timestamp>(
      mills->size(), [mills](auto row) {
        return facebook::velox::Timestamp::fromMillis(mills->valueAt(row));
      });

  testGlobalAgg(timestamp, 0.023, 300);
}

TEST_F(ApproxDistinctTest, globalAggHighCardinalityIntegers) {
  vector_size_t size = 1'000;
  auto values = makeFlatVector<int32_t>(size, [](auto row) { return row; });

  testGlobalAgg(values, 977);
}

TEST_F(ApproxDistinctTest, globalAggVeryLowCardinalityIntegers) {
  vector_size_t size = 1'000;
  auto values = makeFlatVector<int32_t>(size, [](auto /*row*/) { return 27; });

  testGlobalAgg(values, 1);
}

TEST_F(ApproxDistinctTest, toIndexBitLength) {
  ASSERT_EQ(
      common::hll::toIndexBitLength(common::hll::kHighestMaxStandardError), 4);
  ASSERT_EQ(
      common::hll::toIndexBitLength(common::hll::kDefaultStandardError), 11);
  ASSERT_EQ(
      common::hll::toIndexBitLength(common::hll::kLowestMaxStandardError), 16);

  ASSERT_EQ(common::hll::toIndexBitLength(0.0325), 10);
  ASSERT_EQ(common::hll::toIndexBitLength(0.0324), 11);
  ASSERT_EQ(common::hll::toIndexBitLength(0.0230), 11);
  ASSERT_EQ(common::hll::toIndexBitLength(0.0229), 12);
  ASSERT_EQ(common::hll::toIndexBitLength(0.0163), 12);
  ASSERT_EQ(common::hll::toIndexBitLength(0.0162), 13);
  ASSERT_EQ(common::hll::toIndexBitLength(0.0115), 13);
  ASSERT_EQ(common::hll::toIndexBitLength(0.0114), 14);
  ASSERT_EQ(common::hll::toIndexBitLength(0.008125), 14);
  ASSERT_EQ(common::hll::toIndexBitLength(0.008124), 15);
  ASSERT_EQ(common::hll::toIndexBitLength(0.00575), 15);
  ASSERT_EQ(common::hll::toIndexBitLength(0.00574), 16);
}

TEST_F(ApproxDistinctTest, globalAggIntegersWithError) {
  vector_size_t size = 1'000;
  auto values = makeFlatVector<int32_t>(size, [](auto row) { return row; });

  testGlobalAgg(values, common::hll::kLowestMaxStandardError, 1000);
  testGlobalAgg(values, 0.01, 1000);
  testGlobalAgg(values, 0.1, 951);
  testGlobalAgg(values, 0.2, 936);
  testGlobalAgg(values, common::hll::kHighestMaxStandardError, 929);

  values = makeFlatVector<int32_t>(50'000, folly::identity);
  testGlobalAgg(values, common::hll::kLowestMaxStandardError, 50043);
  testGlobalAgg(values, common::hll::kHighestMaxStandardError, 39069);
}

TEST_F(ApproxDistinctTest, globalAggAllNulls) {
  vector_size_t size = 1'000;
  auto values = makeFlatVector<int32_t>(
      size, [](auto row) { return row; }, nullEvery(1));

  auto op = PlanBuilder()
                .values({makeRowVector({values})})
                .singleAggregation({}, {"approx_distinct(c0, 0.01)"})
                .planNode();
  EXPECT_EQ(readSingleValue(op), 0ll);

  op = PlanBuilder()
           .values({makeRowVector({values})})
           .partialAggregation({}, {"approx_distinct(c0, 0.01)"})
           .finalAggregation()
           .planNode();
  EXPECT_EQ(readSingleValue(op), 0ll);

  // approx_distinct over null inputs returns zero, but
  // cardinality(approx_set(x)) over null inputs returns null. See
  // https://github.com/prestodb/presto/issues/17465
  op = PlanBuilder()
           .values({makeRowVector({values})})
           .singleAggregation({}, {"approx_set(c0, 0.01)"})
           .project({"cardinality(a0)"})
           .planNode();
  EXPECT_TRUE(readSingleValue(op).isNull());

  op = PlanBuilder()
           .values({makeRowVector({values})})
           .partialAggregation({}, {"approx_set(c0, 0.01)"})
           .finalAggregation()
           .project({"cardinality(a0)"})
           .planNode();
  EXPECT_TRUE(readSingleValue(op).isNull());
}

TEST_F(ApproxDistinctTest, hugeInt) {
  auto hugeIntValues =
      makeFlatVector<int128_t>(50000, [](auto row) { return row; });
  // Last param is set false to disable tablescan test
  // as DWRF writer doesn't have hugeint support.
  // Refer:https://github.com/facebookincubator/velox/issues/7775
  testGlobalAgg(hugeIntValues, 49669, false);
  testGlobalAgg(
      hugeIntValues, common::hll::kLowestMaxStandardError, 50110, false);
  testGlobalAgg(
      hugeIntValues, common::hll::kHighestMaxStandardError, 41741, false);
}

TEST_F(ApproxDistinctTest, streaming) {
  auto rawInput1 = makeFlatVector<int64_t>({1, 2, 3});
  auto rawInput2 = makeFlatVector<int64_t>(1000, folly::identity);
  auto result =
      testStreaming("approx_distinct", true, {rawInput1}, {rawInput2});
  ASSERT_EQ(result->size(), 1);
  ASSERT_EQ(result->asFlatVector<int64_t>()->valueAt(0), 1010);
  result = testStreaming("approx_distinct", false, {rawInput1}, {rawInput2});
  ASSERT_EQ(result->size(), 1);
  ASSERT_EQ(result->asFlatVector<int64_t>()->valueAt(0), 1010);
}

// Ensure that we convert to dense HLL during merge when necessary.
TEST_F(ApproxDistinctTest, memoryLeakInMerge) {
  constexpr int kSize = 500;
  auto nodeIdGen = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodePtr> sources;
  for (int i = 0; i < 100; ++i) {
    auto c0 =
        makeFlatVector<int32_t>(kSize, [i](auto j) { return j + i * kSize; });
    sources.push_back(PlanBuilder(nodeIdGen)
                          .values({makeRowVector({c0})})
                          .partialAggregation({}, {"approx_distinct(c0, 0.01)"})
                          .planNode());
  }
  core::PlanNodeId finalAgg;
  auto op = PlanBuilder(nodeIdGen)
                .localMerge({}, std::move(sources))
                .finalAggregation()
                .capturePlanNodeId(finalAgg)
                .planNode();
  auto expected = makeFlatVector(std::vector<int64_t>({49810}));
  auto task = assertQuery(op, {makeRowVector({expected})});
  // Should be significantly smaller than 500KB (the number before the fix),
  // because we should be able to convert to DenseHll in the process.
  ASSERT_LT(
      toPlanStats(task->taskStats()).at(finalAgg).peakMemoryBytes, 180'000);
}

} // namespace
} // namespace facebook::velox::aggregate::test
