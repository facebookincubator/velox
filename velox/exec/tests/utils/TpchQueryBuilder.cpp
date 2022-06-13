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
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/common/base/tests/Fs.h"
#include "velox/connectors/hive/HiveConnector.h"

namespace facebook::velox::exec::test {

static int64_t toDate(std::string_view stringDate) {
  Date date;
  parseTo(stringDate, date);
  return date.days();
}

void TpchQueryBuilder::initialize(const std::string& dataPath) {
  for (const auto& [tableName, columns] : kTables_) {
    const fs::path tablePath{dataPath + "/" + tableName};
    for (auto const& dirEntry : fs::directory_iterator{tablePath}) {
      if (!dirEntry.is_regular_file()) {
        continue;
      }
      // Ignore hidden files.
      if (dirEntry.path().filename().c_str()[0] == '.') {
        continue;
      }
      if (tableMetadata_[tableName].dataFiles.empty()) {
        dwio::common::ReaderOptions readerOptions;
        readerOptions.setFileFormat(format_);
        std::unique_ptr<dwio::common::Reader> reader =
            dwio::common::getReaderFactory(readerOptions.getFileFormat())
                ->createReader(
                    std::make_unique<dwio::common::FileInputStream>(
                        dirEntry.path()),
                    readerOptions);
        const auto fileType = reader->rowType();
        const auto fileColumnNames = fileType->names();
        // There can be extra columns in the file towards the end.
        VELOX_CHECK_GE(fileColumnNames.size(), columns.size());
        std::unordered_map<std::string, std::string> fileColumnNamesMap(
            columns.size());
        std::transform(
            columns.begin(),
            columns.end(),
            fileColumnNames.begin(),
            std::inserter(fileColumnNamesMap, fileColumnNamesMap.begin()),
            [](std::string a, std::string b) { return std::make_pair(a, b); });
        auto columnNames = columns;
        auto types = fileType->children();
        types.resize(columnNames.size());
        tableMetadata_[tableName].type =
            std::make_shared<RowType>(std::move(columnNames), std::move(types));
        tableMetadata_[tableName].fileColumnNames =
            std::move(fileColumnNamesMap);
      }
      tableMetadata_[tableName].dataFiles.push_back(dirEntry.path());
    }
  }
}

const std::vector<std::string>& TpchQueryBuilder::getTableNames() {
  return kTableNames_;
}

TpchPlan TpchQueryBuilder::getQueryPlan(int queryId) const {
  switch (queryId) {
    case 1:
      return getQ1Plan();
    case 2:
      return getQ2Plan();
    case 6:
      return getQ6Plan();
    case 13:
      return getQ13Plan();
    case 18:
      return getQ18Plan();
    default:
      VELOX_NYI("TPC-H query {} is not supported yet", queryId);
  }
}

namespace {
static const std::string kLineitem = "lineitem";
static const std::string kOrders = "orders";
static const std::string kCustomer = "customer";
static const std::string kPart = "part";
static const std::string kSupplier = "supplier";
static const std::string kPartSupp = "partsupp";
static const std::string kNation = "nation";
static const std::string kRegion = "region";
} // namespace

TpchPlan TpchQueryBuilder::getQ1Plan() const {
  std::vector<std::string> selectedColumns = {
      "l_returnflag",
      "l_linestatus",
      "l_quantity",
      "l_extendedprice",
      "l_discount",
      "l_tax",
      "l_shipdate"};

  auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  // shipdate <= '1998-09-02'
  const auto shipDate = "l_shipdate";
  std::string filter;
  // DWRF does not support Date type. Use Varchar instead.
  if (selectedRowType->findChild(shipDate)->isVarchar()) {
    filter = "l_shipdate <= '1998-09-02'";
  } else {
    filter = "l_shipdate <= '1998-09-02'::DATE";
  }

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;

  const auto partialAggStage =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineitem, selectedRowType, fileColumnNames, {filter})
          .capturePlanNodeId(lineitemPlanNodeId)
          .project(
              {"l_returnflag",
               "l_linestatus",
               "l_quantity",
               "l_extendedprice",
               "l_extendedprice * (1.0 - l_discount) AS l_sum_disc_price",
               "l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax) AS l_sum_charge",
               "l_discount"})
          .partialAggregation(
              {"l_returnflag", "l_linestatus"},
              {"sum(l_quantity)",
               "sum(l_extendedprice)",
               "sum(l_sum_disc_price)",
               "sum(l_sum_charge)",
               "avg(l_quantity)",
               "avg(l_extendedprice)",
               "avg(l_discount)",
               "count(0)"})
          .planNode();

  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition({}, {partialAggStage})
                  .finalAggregation()
                  .orderBy({"l_returnflag", "l_linestatus"}, false)
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

std::shared_ptr<core::PlanNode> TpchQueryBuilder::getQ2MinimumCostSupplierPlan(
    std::shared_ptr<PlanNodeIdGenerator>& planNodeIdGenerator,
    TpchPlan& context) const {
  core::PlanNodeId partsuppScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId regionScanNodeId;

  auto partsupp =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kPartSupp,
              getRowType(
                  kPartSupp, {"ps_partkey", "ps_suppkey", "ps_supplycost"}))
          .capturePlanNodeId(partsuppScanNodeId);

  auto supplier =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {"s_suppkey"},
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       kSupplier,
                       getRowType(kSupplier, {"s_suppkey", "s_nationkey"}))
                   .capturePlanNodeId(supplierScanNodeId)
                   .planNode()})
          .planNode();

  auto nation =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {"n_nationkey"},
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       kNation,
                       getRowType(kNation, {"n_nationkey", "n_regionkey"}))
                   .capturePlanNodeId(nationScanNodeId)
                   .planNode()})
          .planNode();
  auto region = PlanBuilder(planNodeIdGenerator)
                    .localPartition(
                        {"r_regionkey"},
                        {PlanBuilder(planNodeIdGenerator)
                             .tableScan(
                                 kRegion,
                                 getRowType(kRegion, {"r_regionkey", "r_name"}),
                                 getFileColumnNames(kRegion),
                                 {"r_name = 'EUROPE'"})
                             .capturePlanNodeId(regionScanNodeId)
                             .planNode()})
                    .planNode();

  auto partialMinCostSuppliers =
      partsupp
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              {"ps_partkey", "ps_supplycost", "s_nationkey"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"ps_partkey", "ps_supplycost", "n_regionkey"})
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"ps_partkey", "ps_supplycost"})
          .partialAggregation(
              {"ps_partkey"}, {"min(ps_supplycost) as pmin_supplycost"})
          .planNode();

  auto minCostSuppliers = PlanBuilder(planNodeIdGenerator)
                              .localPartition({}, {partialMinCostSuppliers})
                              .finalAggregation(
                                  {"ps_partkey"},
                                  {"min(pmin_supplycost) AS min_supplycost"},
                                  {DOUBLE()})
                              .planNode();

  context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartSupp);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);

  return minCostSuppliers;
}

TpchPlan TpchQueryBuilder::getQ2Plan() const {
  TpchPlan context;
  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  auto minCostSuppliers =
      getQ2MinimumCostSupplierPlan(planNodeIdGenerator, context);

  core::PlanNodeId partScanNodeId;
  core::PlanNodeId partsuppScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId regionScanNodeId;

  auto part =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kPart,
              getRowType(kPart, {"p_partkey", "p_size", "p_mfgr", "p_type"}),
              getFileColumnNames(kPart),
              {},
              {"(p_size = 15) AND (p_type LIKE '%BRASS')"})
          .capturePlanNodeId(partScanNodeId);

  auto partsupp =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {"ps_partkey"},
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       kPartSupp,
                       getRowType(
                           kPartSupp,
                           {"ps_partkey", "ps_suppkey", "ps_supplycost"}))
                   .capturePlanNodeId(partsuppScanNodeId)
                   .planNode()})
          .planNode();

  auto supplier = PlanBuilder(planNodeIdGenerator)
                      .localPartition(
                          {"s_suppkey"},
                          {PlanBuilder(planNodeIdGenerator)
                               .tableScan(
                                   kSupplier,
                                   getRowType(
                                       kSupplier,
                                       {"s_suppkey",
                                        "s_name",
                                        "s_address",
                                        "s_nationkey",
                                        "s_phone",
                                        "s_acctbal",
                                        "s_comment"}))
                               .capturePlanNodeId(supplierScanNodeId)
                               .planNode()})
                      .planNode();

  auto nation =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {"n_nationkey"},
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       kNation,
                       getRowType(
                           kNation, {"n_nationkey", "n_name", "n_regionkey"}))
                   .capturePlanNodeId(nationScanNodeId)
                   .planNode()})
          .planNode();
  auto region = PlanBuilder(planNodeIdGenerator)
                    .localPartition(
                        {"r_regionkey"},
                        {PlanBuilder(planNodeIdGenerator)
                             .tableScan(
                                 kRegion,
                                 getRowType(kRegion, {"r_regionkey", "r_name"}),
                                 getFileColumnNames(kRegion),
                                 {"r_name = 'EUROPE'"})
                             .capturePlanNodeId(regionScanNodeId)
                             .planNode()})
                    .planNode();

  auto plan =
      part.hashJoin(
              {"p_partkey"},
              {"ps_partkey"},
              partsupp,
              "",
              {"p_partkey", "p_mfgr", "ps_suppkey", "ps_supplycost"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              {"p_partkey",
               "p_mfgr",
               "ps_supplycost",
               "s_name",
               "s_address",
               "s_nationkey",
               "s_phone",
               "s_acctbal",
               "s_comment"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"p_partkey",
               "p_mfgr",
               "ps_supplycost",
               "s_name",
               "s_address",
               "s_phone",
               "s_acctbal",
               "s_comment",
               "n_name",
               "n_regionkey"})
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"p_partkey",
               "p_mfgr",
               "ps_supplycost",
               "s_name",
               "s_address",
               "s_phone",
               "s_acctbal",
               "s_comment",
               "n_name"})
          .hashJoin(
              {"p_partkey", "ps_supplycost"},
              {"ps_partkey", "min_supplycost"},
              minCostSuppliers,
              "",
              {"p_partkey",
               "p_mfgr",
               "ps_supplycost",
               "s_name",
               "s_address",
               "s_phone",
               "s_acctbal",
               "s_comment",
               "n_name",
               "min_supplycost"})
          .topN({"s_acctbal desc", "n_name", "s_name", "p_partkey"}, 100, false)
          .planNode();

  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartSupp);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ6Plan() const {
  std::vector<std::string> selectedColumns = {
      "l_shipdate", "l_extendedprice", "l_quantity", "l_discount"};

  auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  const auto shipDate = "l_shipdate";
  std::string shipDateFilter;
  // DWRF does not support Date type. Use Varchar instead.
  if (selectedRowType->findChild(shipDate)->isVarchar()) {
    shipDateFilter = "l_shipdate between '1994-01-01' and '1994-12-31'";
  } else {
    shipDateFilter =
        "l_shipdate between '1994-01-01'::DATE and '1994-12-31'::DATE";
  }

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition(
                      {},
                      {PlanBuilder(planNodeIdGenerator)
                           .tableScan(
                               kLineitem,
                               selectedRowType,
                               fileColumnNames,
                               {shipDateFilter,
                                "l_discount between 0.05 and 0.07",
                                "l_quantity < 24.0"})
                           .capturePlanNodeId(lineitemPlanNodeId)
                           .project({"l_extendedprice * l_discount"})
                           .partialAggregation({}, {"sum(p0)"})
                           .planNode()})
                  .finalAggregation()
                  .planNode();
  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ13Plan() const {
  std::vector<std::string> ordersColumns = {
      "o_custkey", "o_comment", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey"};

  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;

  auto customers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .capturePlanNodeId(customerScanNodeId)
          .planNode();

  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition(
                      {},
                      {PlanBuilder(planNodeIdGenerator)
                           .tableScan(
                               kOrders,
                               ordersSelectedRowType,
                               ordersFileColumns,
                               {},
                               "o_comment not like '%special%requests%'")
                           .capturePlanNodeId(ordersScanNodeId)
                           .hashJoin(
                               {"o_custkey"},
                               {"c_custkey"},
                               customers,
                               "",
                               {"c_custkey", "o_orderkey"},
                               core::JoinType::kRight)
                           .partialAggregation(
                               {"c_custkey"}, {"count(o_orderkey) as pc_count"})
                           .planNode()})
                  .finalAggregation(
                      {"c_custkey"}, {"count(pc_count) as c_count"}, {BIGINT()})
                  .singleAggregation({"c_count"}, {"count(0) as custdist"})
                  .orderBy({"custdist DESC", "c_count DESC"}, false)
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ18Plan() const {
  std::vector<std::string> lineitemColumns = {"l_orderkey", "l_quantity"};
  std::vector<std::string> ordersColumns = {
      "o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"};
  std::vector<std::string> customerColumns = {"c_name", "c_custkey"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;

  auto bigOrders =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {"l_orderkey"},
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       kLineitem, lineitemSelectedRowType, lineitemFileColumns)
                   .capturePlanNodeId(lineitemScanNodeId)
                   .partialAggregation(
                       {"l_orderkey"}, {"sum(l_quantity) AS partial_sum"})
                   .planNode()})
          .finalAggregation(
              {"l_orderkey"}, {"sum(partial_sum) AS quantity"}, {DOUBLE()})
          .filter("quantity > 300.0")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {},
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
                   .capturePlanNodeId(ordersScanNodeId)
                   .hashJoin(
                       {"o_orderkey"},
                       {"l_orderkey"},
                       bigOrders,
                       "",
                       {"o_orderkey",
                        "o_custkey",
                        "o_orderdate",
                        "o_totalprice",
                        "l_orderkey",
                        "quantity"})
                   .hashJoin(
                       {"o_custkey"},
                       {"c_custkey"},
                       PlanBuilder(planNodeIdGenerator)
                           .tableScan(
                               kCustomer,
                               customerSelectedRowType,
                               customerFileColumns)
                           .capturePlanNodeId(customerScanNodeId)
                           .planNode(),
                       "",
                       {"c_name",
                        "c_custkey",
                        "o_orderkey",
                        "o_orderdate",
                        "o_totalprice",
                        "quantity"})
                   .planNode()})
          .orderBy({"o_totalprice DESC", "o_orderdate"}, false)
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

const std::vector<std::string> TpchQueryBuilder::kTableNames_ = {
    "lineitem",
    "orders",
    "customer"};

const std::unordered_map<std::string, std::vector<std::string>>
    TpchQueryBuilder::kTables_ = {
        std::make_pair(
            "lineitem",
            std::vector<std::string>{
                "l_orderkey",
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
                "l_comment"}),
        std::make_pair(
            "orders",
            std::vector<std::string>{
                "o_orderkey",
                "o_custkey",
                "o_orderstatus",
                "o_totalprice",
                "o_orderdate",
                "o_orderpriority",
                "o_clerk",
                "o_shippriority",
                "o_comment"}),
        std::make_pair(
            "customer",
            std::vector<std::string>{
                "c_custkey",
                "c_name",
                "c_address",
                "c_nationkey",
                "c_phone",
                "c_acctbal",
                "c_mktsegment",
                "c_comment"}),
        std::make_pair(
            "part",
            std::vector<std::string>{
                "p_partkey",
                "p_name",
                "p_mfgr",
                "p_brand",
                "p_type",
                "p_size",
                "p_container",
                "p_retailprice",
                "p_comment"}),
        std::make_pair(
            "partsupp",
            std::vector<std::string>{
                "ps_partkey",
                "ps_suppkey",
                "ps_availqty",
                "ps_supplycost",
                "ps_comment"}),
        std::make_pair(
            "supplier",
            std::vector<std::string>{
                "s_suppkey",
                "s_name",
                "s_address",
                "s_nationkey",
                "s_phone",
                "s_acctbal",
                "s_comment"}),
        std::make_pair(
            "nation",
            std::vector<std::string>{
                "n_nationkey",
                "n_name",
                "n_regionkey",
                "n_comment"}),
        std::make_pair(
            "region",
            std::vector<std::string>{"r_regionkey", "r_name", "r_comment"})};
} // namespace facebook::velox::exec::test
