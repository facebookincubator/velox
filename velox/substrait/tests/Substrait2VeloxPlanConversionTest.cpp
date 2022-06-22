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

#include "velox/substrait/tests/JsonToProtoConverter.h"

#include "velox/common/base/tests/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/substrait/SubstraitToVeloxPlan.h"
#include "velox/type/Type.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::exec;
namespace vestrait = facebook::velox::substrait;

class Substrait2VeloxPlanConversionTest
    : public exec::test::HiveConnectorTestBase {
 protected:
  class VeloxConverter {
   public:
    // This class is an iterator for Velox computing.
    class WholeComputeResultIterator {
     public:
      WholeComputeResultIterator(
          const std::shared_ptr<const core::PlanNode>& planNode,
          u_int32_t index,
          const std::vector<std::string>& paths,
          const std::vector<u_int64_t>& starts,
          const std::vector<u_int64_t>& lengths,
          const dwio::common::FileFormat& format)
          : planNode_(planNode),
            index_(index),
            paths_(paths),
            starts_(starts),
            lengths_(lengths),
            format_(format) {
        // Construct the splits.
        std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
            connectorSplits;
        connectorSplits.reserve(paths.size());
        for (int idx = 0; idx < paths.size(); idx++) {
          auto path = paths[idx];
          auto start = starts[idx];
          auto length = lengths[idx];
          auto split = std::make_shared<
              facebook::velox::connector::hive::HiveConnectorSplit>(
              facebook::velox::exec::test::kHiveConnectorId,
              path,
              format,
              start,
              length);
          connectorSplits.emplace_back(split);
        }
        splits_.reserve(connectorSplits.size());
        for (const auto& connectorSplit : connectorSplits) {
          splits_.emplace_back(exec::Split(folly::copy(connectorSplit), -1));
        }

        params_.planNode = planNode;
        cursor_ = std::make_unique<exec::test::TaskCursor>(params_);
        addSplits_ = [&](Task* task) {
          if (noMoreSplits_) {
            return;
          }
          for (auto& split : splits_) {
            task->addSplit("0", std::move(split));
          }
          task->noMoreSplits("0");
          noMoreSplits_ = true;
        };
      }

      bool HasNext() {
        if (!mayHasNext_) {
          return false;
        }
        if (numRows_ > 0) {
          return true;
        } else {
          addSplits_(cursor_->task().get());
          if (cursor_->moveNext()) {
            result_ = cursor_->current();
            numRows_ += result_->size();
            return true;
          } else {
            mayHasNext_ = false;
            return false;
          }
        }
      }

      RowVectorPtr Next() {
        numRows_ = 0;
        return result_;
      }

     private:
      const std::shared_ptr<const core::PlanNode> planNode_;
      std::unique_ptr<exec::test::TaskCursor> cursor_;
      exec::test::CursorParameters params_;
      std::vector<exec::Split> splits_;
      bool noMoreSplits_ = false;
      std::function<void(exec::Task*)> addSplits_;
      u_int32_t index_;
      std::vector<std::string> paths_;
      std::vector<u_int64_t> starts_;
      std::vector<u_int64_t> lengths_;
      dwio::common::FileFormat format_;
      uint64_t numRows_ = 0;
      bool mayHasNext_ = true;
      RowVectorPtr result_;
    };

    // This method will resume the Substrait plan from Json file,
    // and convert it into Velox PlanNode. A result iterator for
    // Velox computing will be returned.
    std::shared_ptr<WholeComputeResultIterator> getResIter(
        const std::string& subPlanPath) {
      // Read json file and resume the Substrait plan.
      std::ifstream subJson(subPlanPath);
      std::stringstream buffer;
      buffer << subJson.rdbuf();
      std::string subData = buffer.str();
      ::substrait::Plan subPlan;
      google::protobuf::util::JsonStringToMessage(subData, &subPlan);

      auto planConverter = std::make_shared<
          facebook::velox::substrait::SubstraitVeloxPlanConverter>();
      // Convert to Velox PlanNode.
      auto planNode = planConverter->toVeloxPlan(subPlan, memoryPool_.get());

      auto splitInfos = planConverter->splitInfos();
      auto leafPlanNodeIds = planNode->leafPlanNodeIds();
      // Here only one leaf node is expected here.
      EXPECT_EQ(1, leafPlanNodeIds.size());
      auto iter = leafPlanNodeIds.begin();
      auto splitInfo = splitInfos[*iter].get();

      // Get the information for TableScan.
      u_int32_t partitionIndex = splitInfo->partitionIndex;
      std::vector<std::string> paths = splitInfo->paths;

      // In test, need to get the absolute path of the generated ORC file.
      auto tempPath = getTmpDirPath();
      std::vector<std::string> absolutePaths;
      absolutePaths.reserve(paths.size());

      for (const auto& path : paths) {
        absolutePaths.emplace_back(fmt::format("file://{}{}", tempPath, path));
      }

      std::vector<u_int64_t> starts = splitInfo->starts;
      std::vector<u_int64_t> lengths = splitInfo->lengths;
      auto format = splitInfo->format;
      // Construct the result iterator.
      auto resIter = std::make_shared<WholeComputeResultIterator>(
          planNode, partitionIndex, absolutePaths, starts, lengths, format);
      return resIter;
    }

    std::string getTmpDirPath() const {
      return tmpDir_->path;
    }

    std::shared_ptr<exec::test::TempDirectoryPath> tmpDir_{
        exec::test::TempDirectoryPath::create()};

   private:
    std::unique_ptr<memory::MemoryPool> memoryPool_{
        memory::getDefaultScopedMemoryPool()};
  };

  std::shared_ptr<exec::test::TempDirectoryPath> tmpDir_{
      exec::test::TempDirectoryPath::create()};

  void genLineitemORC(const std::shared_ptr<VeloxConverter>& veloxConverter) {
    auto type =
        ROW({"l_orderkey",
             "l_partkey",
             "l_suppkey",
             "l_linenumber",
             "l_quantity",
             "l_extendedprice",
             "l_discount",
             "l_tax",
             "l_returnflag",
             "l_linestatus",
             "l_shipdate",
             "l_commitdate",
             "l_receiptdate",
             "l_shipinstruct",
             "l_shipmode",
             "l_comment"},
            {BIGINT(),
             BIGINT(),
             BIGINT(),
             INTEGER(),
             DOUBLE(),
             DOUBLE(),
             DOUBLE(),
             DOUBLE(),
             VARCHAR(),
             VARCHAR(),
             DOUBLE(),
             DOUBLE(),
             DOUBLE(),
             VARCHAR(),
             VARCHAR(),
             VARCHAR()});

    std::vector<VectorPtr> vectors;
    // TPC-H lineitem table has 16 columns.
    int colNum = 16;
    vectors.reserve(colNum);
    // lOrderkeyData
    vectors.emplace_back(makeFlatVector<int64_t>(
        {4636438147,
         2012485446,
         1635327427,
         8374290148,
         2972204230,
         8001568994,
         989963396,
         2142695974,
         6354246853,
         4141748419}));

    // lPartkeyData
    vectors.emplace_back(makeFlatVector<int64_t>(
        {263222018,
         255918298,
         143549509,
         96877642,
         201976875,
         196938305,
         100260625,
         273511608,
         112999357,
         299103530}));

    // lSuppkeyData
    vectors.emplace_back(makeFlatVector<int64_t>(
        {2102019,
         13998315,
         12989528,
         4717643,
         9976902,
         12618306,
         11940632,
         871626,
         1639379,
         3423588}));

    // lLinenumberData
    vectors.emplace_back(
        makeFlatVector<int32_t>({4, 6, 1, 5, 1, 2, 1, 5, 2, 6}));

    // lQuantityData
    vectors.emplace_back(makeFlatVector<double>(
        {6.0, 1.0, 19.0, 4.0, 6.0, 12.0, 23.0, 11.0, 16.0, 19.0}));

    // lExtendedpriceData
    vectors.emplace_back(makeFlatVector<double>(
        {30586.05,
         7821.0,
         1551.33,
         30681.2,
         1941.78,
         66673.0,
         6322.44,
         41754.18,
         8704.26,
         63780.36}));

    // lDiscountData
    vectors.emplace_back(makeFlatVector<double>(
        {0.05, 0.06, 0.01, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07}));

    // lTaxData
    vectors.emplace_back(makeFlatVector<double>(
        {0.02, 0.03, 0.01, 0.0, 0.01, 0.01, 0.03, 0.07, 0.01, 0.04}));

    // lReturnflagData
    vectors.emplace_back(makeFlatVector<StringView>(
        {"N", "A", "A", "R", "A", "N", "A", "A", "N", "R"}));

    std::vector<std::string> lLinestatusData = {
        "O", "F", "F", "F", "F", "O", "F", "F", "O", "F"};
    // lLinestatusData
    vectors.emplace_back(makeFlatVector<StringView>(
        {"O", "F", "F", "F", "F", "O", "F", "F", "O", "F"}));
    // lShipdateNewData
    vectors.emplace_back(makeFlatVector<double>(
        {8953.666666666666,
         8773.666666666666,
         9034.666666666666,
         8558.666666666666,
         9072.666666666666,
         8864.666666666666,
         9004.666666666666,
         8778.666666666666,
         9013.666666666666,
         8832.666666666666}));

    // lCommitdateNewData
    vectors.emplace_back(makeFlatVector<double>(
        {10447.666666666666,
         8953.666666666666,
         8325.666666666666,
         8527.666666666666,
         8438.666666666666,
         10049.666666666666,
         9036.666666666666,
         8666.666666666666,
         9519.666666666666,
         9138.666666666666}));

    // lReceiptdateNewData
    vectors.emplace_back(makeFlatVector<double>(
        {10456.666666666666,
         8979.666666666666,
         8299.666666666666,
         8474.666666666666,
         8525.666666666666,
         9996.666666666666,
         9103.666666666666,
         8726.666666666666,
         9593.666666666666,
         9178.666666666666}));

    // lShipinstructData
    vectors.emplace_back(makeFlatVector<StringView>(
        {"COLLECT COD",
         "NONE",
         "TAKE BACK RETURN",
         "NONE",
         "TAKE BACK RETURN",
         "NONE",
         "DELIVER IN PERSON",
         "DELIVER IN PERSON",
         "TAKE BACK RETURN",
         "NONE"}));

    // lShipmodeData
    vectors.emplace_back(makeFlatVector<StringView>(
        {"FOB",
         "REG AIR",
         "MAIL",
         "FOB",
         "RAIL",
         "SHIP",
         "REG AIR",
         "REG AIR",
         "TRUCK",
         "AIR"}));

    // lCommentData
    vectors.emplace_back(makeFlatVector<StringView>(
        {" the furiously final foxes. quickly final p",
         "thely ironic",
         "ate furiously. even, pending pinto bean",
         "ackages af",
         "odolites. slyl",
         "ng the regular requests sleep above",
         "lets above the slyly ironic theodolites sl",
         "lyly regular excuses affi",
         "lly unusual theodolites grow slyly above",
         " the quickly ironic pains lose car"}));

    // Batches has only one RowVector here.
    uint64_t nullCount = 0;
    std::vector<RowVectorPtr> batches{std::make_shared<RowVector>(
        pool_.get(), type, nullptr, 10, vectors, nullCount)};

    // Writes data into an ORC file.
    auto sink = std::make_unique<facebook::velox::dwio::common::FileSink>(
        veloxConverter->getTmpDirPath() + "/mock_lineitem.orc");
    auto config = std::make_shared<facebook::velox::dwrf::Config>();
    const int64_t writerMemoryCap = std::numeric_limits<int64_t>::max();
    facebook::velox::dwrf::WriterOptions options;
    options.config = config;
    options.schema = type;
    options.memoryBudget = writerMemoryCap;
    options.flushPolicyFactory = nullptr;
    options.layoutPlannerFactory = nullptr;
    auto writer = std::make_unique<facebook::velox::dwrf::Writer>(
        options,
        std::move(sink),
        facebook::velox::memory::getProcessDefaultMemoryManager().getRoot());
    for (size_t i = 0; i < batches.size(); ++i) {
      writer->write(batches[i]);
    }
    writer->close();
  }
};

// This test will firstly generate mock TPC-H lineitem ORC file. Then, Velox's
// computing will be tested based on the generated ORC file.
// Input: Json file of the Substrait plan for the first stage of below modified
// TPC-H Q6 query:
//
//  select sum(l_extendedprice*l_discount) as revenue from lineitem where
//  l_shipdate >= 8766 and l_shipdate < 9131 and l_discount between .06
//  - 0.01 and .06 + 0.01 and l_quantity < 24
//
//  Tested Velox computings include: TableScan (Filter Pushdown) + Project +
//  Aggregate
//  Output: the Velox computed Aggregation result

TEST_F(Substrait2VeloxPlanConversionTest, q6FirstStageTest) {
  auto veloxConverter = std::make_shared<VeloxConverter>();
  genLineitemORC(veloxConverter);
  // Find and deserialize Substrait plan json file.
  std::string subPlanPath =
      getDataFilePath("velox/substrait/tests", "data/q6_first_stage.json");
  auto resIter = veloxConverter->getResIter(subPlanPath);
  while (resIter->HasNext()) {
    auto rv = resIter->Next();
    auto size = rv->size();
    ASSERT_EQ(size, 1);
    std::string res = rv->toString(0);
    ASSERT_EQ(res, "{13613.1921}");
  }
}

// This test will firstly generate mock TPC-H lineitem ORC file. Then, Velox's
// computing will be tested based on the generated ORC file.
// Input: Json file of the Substrait plan for the first stage of the below
// modified TPC-H Q1 query:
//
// select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty,
// sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 -
// l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 +
// l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as
// avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem
// where l_shipdate <= 10471 group by l_returnflag, l_linestatus order by
// l_returnflag, l_linestatus
//
//  Tested Velox computings include: TableScan (Filter Pushdown) + Project +
//  Aggregate
//  Output: the Velox computed Aggregation result

TEST_F(Substrait2VeloxPlanConversionTest, q1FirstStageTest) {
  auto veloxConverter = std::make_shared<VeloxConverter>();
  genLineitemORC(veloxConverter);
  // Find and deserialize Substrait plan json file.
  std::string subPlanPath =
      getDataFilePath("velox/substrait/tests", "data/q1_first_stage.json");
  auto resIter = veloxConverter->getResIter(subPlanPath);
  while (resIter->HasNext()) {
    auto rv = resIter->Next();
    auto size = rv->size();
    ASSERT_EQ(size, 3);
    ASSERT_EQ(
        rv->toString(0),
        "{N, O, 34, 105963.31, 99911.3719, 101201.05309399999, 34, 3, 105963.31, 3, 0.16999999999999998, 3, 3}");
    ASSERT_EQ(
        rv->toString(1),
        "{A, F, 60, 59390.729999999996, 56278.5879, 59485.994223, 60, 5, 59390.729999999996, 5, 0.24, 5, 5}");
    ASSERT_EQ(
        rv->toString(2),
        "{R, F, 23, 94461.56, 87849.2508, 90221.880192, 23, 2, 94461.56, 2, 0.14, 2, 2}");
  }
}