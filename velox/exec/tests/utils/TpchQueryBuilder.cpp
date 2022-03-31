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
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/type/tests/FilterBuilder.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

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
    case 6:
      return getQ6Plan();
    case 18:
      return getQ18Plan();
    default:
      VELOX_NYI("TPC-H query {} is not supported yet", queryId);
  }
}

namespace {
using ColumnHandleMap =
    std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>;

std::shared_ptr<connector::hive::HiveTableHandle> makeTableHandle(
    const std::string& tableName,
    common::test::SubfieldFilters subfieldFilters,
    const std::shared_ptr<const core::ITypedExpr>& remainingFilter = nullptr) {
  return std::make_shared<connector::hive::HiveTableHandle>(
      tableName, true, std::move(subfieldFilters), remainingFilter);
}

std::shared_ptr<connector::hive::HiveColumnHandle> regularColumn(
    const std::string& name,
    const TypePtr& type) {
  return std::make_shared<connector::hive::HiveColumnHandle>(
      name, connector::hive::HiveColumnHandle::ColumnType::kRegular, type);
}

ColumnHandleMap allRegularColumns(
    const std::shared_ptr<const RowType>& rowType,
    const std::unordered_map<std::string, std::string>& fileColumnNames) {
  ColumnHandleMap assignments;
  assignments.reserve(rowType->size());
  for (uint32_t i = 0; i < rowType->size(); ++i) {
    const auto& name = rowType->nameOf(i);
    assignments[name] =
        regularColumn(fileColumnNames.at(name), rowType->childAt(i));
  }
  return assignments;
}
} // namespace

TpchPlan TpchQueryBuilder::getQ1Plan() const {
  static const std::string kLineitem = "lineitem";
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
  common::test::SubfieldFiltersBuilder filtersBuilder;
  // DWRF does not support Date type. Use Varchar instead.
  if (selectedRowType->findChild(shipDate)->isVarchar()) {
    filtersBuilder.add(
        fileColumnNames.at(shipDate),
        common::test::lessThanOrEqual("1998-09-02"));
  } else {
    filtersBuilder.add(
        fileColumnNames.at(shipDate),
        common::test::lessThanOrEqual(toDate("1998-09-02")));
  }
  auto filters = filtersBuilder.build();
  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;

  const auto partialAggStage =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              selectedRowType,
              makeTableHandle(kLineitem, std::move(filters)),
              allRegularColumns(selectedRowType, fileColumnNames))
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
              {0, 1},
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
                  .finalAggregation(
                      {0, 1},
                      {"sum(a0)",
                       "sum(a1)",
                       "sum(a2)",
                       "sum(a3)",
                       "avg(a4)",
                       "avg(a5)",
                       "avg(a6)",
                       "count(a7)"},
                      {DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       BIGINT()})
                  .orderBy({"l_returnflag", "l_linestatus"}, false)
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ6Plan() const {
  static const std::string kLineitem = "lineitem";
  std::vector<std::string> selectedColumns = {
      "l_shipdate", "l_extendedprice", "l_quantity", "l_discount"};

  auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  const auto shipDate = "l_shipdate";
  common::test::SubfieldFiltersBuilder filtersBuilder;
  // DWRF does not support Date type. Use Varchar instead.
  if (selectedRowType->findChild(shipDate)->isVarchar()) {
    filtersBuilder.add(
        fileColumnNames.at(shipDate),
        common::test::between("1994-01-01", "1994-12-31"));
  } else {
    filtersBuilder.add(
        fileColumnNames.at(shipDate),
        common::test::between(toDate("1994-01-01"), toDate("1994-12-31")));
  }
  auto filters = filtersBuilder
                     .add(
                         fileColumnNames.at("l_discount"),
                         common::test::betweenDouble(0.05, 0.07))
                     .add(
                         fileColumnNames.at("l_quantity"),
                         common::test::lessThanDouble(24.0))
                     .build();

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {},
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       selectedRowType,
                       makeTableHandle(kLineitem, std::move(filters)),
                       allRegularColumns(selectedRowType, fileColumnNames))
                   .capturePlanNodeId(lineitemPlanNodeId)
                   .project({"l_extendedprice * l_discount"})
                   .partialAggregation({}, {"sum(p0)"})
                   .planNode()})
          .finalAggregation({}, {"sum(a0)"}, {DOUBLE()})
          .planNode();
  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ18Plan() const {
  static const std::string kLineitem = "lineitem";
  static const std::string kOrders = "orders";
  static const std::string kCustomer = "customer";
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
              {0}, // l_orderkey
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       lineitemSelectedRowType,
                       makeTableHandle(
                           kLineitem, common::test::SubfieldFilters{}),
                       allRegularColumns(
                           lineitemSelectedRowType, lineitemFileColumns))
                   .capturePlanNodeId(lineitemScanNodeId)
                   .partialAggregation({0}, {"sum(l_quantity) AS partial_sum"})
                   .planNode()})
          .finalAggregation({0}, {"sum(partial_sum) AS quantity"}, {DOUBLE()})
          .filter("quantity > 300.0")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {},
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       ordersSelectedRowType,
                       makeTableHandle(
                           kOrders, common::test::SubfieldFilters{}),
                       allRegularColumns(
                           ordersSelectedRowType, ordersFileColumns))
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
                               customerSelectedRowType,
                               makeTableHandle(
                                   kCustomer, common::test::SubfieldFilters{}),
                               allRegularColumns(
                                   customerSelectedRowType,
                                   customerFileColumns))
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
                "c_addres",
                "c_nationkey",
                "c_phone",
                "c_acctbal",
                "c_mktsegment",
                "c_comment"})};

} // namespace facebook::velox::exec::test
