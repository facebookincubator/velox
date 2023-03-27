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
#include "velox/exec/tests/utils/SsbQueryBuilder.h"

#include "velox/common/base/Fs.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/tpch/gen/SsbGen.h"

namespace facebook::velox::exec::test {

namespace {
int64_t toDate(std::string_view stringDate) {
  Date date;
  parseTo(stringDate, date);
  return date.days();
}

/// DWRF does not support Date type and Varchar is used.
/// Return the Date filter expression as per data format.
std::string formatDateFilter(
    const std::string& stringDate,
    const RowTypePtr& rowType,
    const std::string& lowerBound,
    const std::string& upperBound) {
  bool isDwrf = rowType->findChild(stringDate)->isVarchar();
  auto suffix = isDwrf ? "" : "::DATE";

  if (!lowerBound.empty() && !upperBound.empty()) {
    return fmt::format(
        "{} between {}{} and {}{}",
        stringDate,
        lowerBound,
        suffix,
        upperBound,
        suffix);
  } else if (!lowerBound.empty()) {
    return fmt::format("{} > {}{}", stringDate, lowerBound, suffix);
  } else if (!upperBound.empty()) {
    return fmt::format("{} < {}{}", stringDate, upperBound, suffix);
  }

  VELOX_FAIL(
      "Date range check expression must have either a lower or an upper bound");
}

std::vector<std::string> mergeColumnNames(
    const std::vector<std::string>& firstColumnVector,
    const std::vector<std::string>& secondColumnVector) {
  std::vector<std::string> mergedColumnVector = std::move(firstColumnVector);
  mergedColumnVector.insert(
      mergedColumnVector.end(),
      secondColumnVector.begin(),
      secondColumnVector.end());
  return mergedColumnVector;
};
} // namespace

void SsbQueryBuilder::initialize(const std::string& dataPath) {
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
        dwio::common::ReaderOptions readerOptions{pool_.get()};
        readerOptions.setFileFormat(format_);
        auto input = std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<LocalReadFile>(dirEntry.path().string()),
            readerOptions.getMemoryPool());
        std::unique_ptr<dwio::common::Reader> reader =
            dwio::common::getReaderFactory(readerOptions.getFileFormat())
                ->createReader(std::move(input), readerOptions);
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

const std::vector<std::string>& SsbQueryBuilder::getTableNames() {
  return kTableNames_;
}

SsbPlan SsbQueryBuilder::getQueryPlan(int queryId) const {
  switch (queryId) {
    case 1:
      return getQ1Plan();
    case 2:
      return getQ2Plan();
    case 3:
      return getQ3Plan();
    case 4:
      return getQ4Plan();
    case 5:
      return getQ5Plan();
    case 6:
      return getQ6Plan();
    case 7:
      return getQ7Plan();
    case 8:
      return getQ8Plan();
    case 9:
      return getQ9Plan();
    case 10:
      return getQ10Plan();
    case 11:
      return getQ11Plan();
    case 12:
      return getQ12Plan();
    case 13:
      return getQ13Plan();
    default:
      VELOX_NYI("SSB query {} is not supported yet", queryId);
  }
}

SsbPlan SsbQueryBuilder::getQ1Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_extendedprice", "lo_discount", "lo_orderdate", "lo_quantity"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter("d_year = 1993")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .filter("(lo_discount between 1 and 3) AND (lo_quantity < 25) ")
          .project(
              {"lo_extendedprice * lo_discount AS part_revenue",
               "lo_orderdate"})
          .hashJoin(
              {"lo_orderdate"}, {"d_datekey"}, dates, "", {"part_revenue"})
          .partialAggregation({}, {"sum(part_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .project({"revenue"})
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ2Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_extendedprice", "lo_discount", "lo_orderdate", "lo_quantity"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_yearmonthnum"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter("d_yearmonthnum = 199401")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .filter(
              "(lo_discount between 4 and 6) AND (lo_quantity between 26 and 35) ")
          .project(
              {"lo_extendedprice * lo_discount AS part_revenue",
               "lo_orderdate"})
          .hashJoin(
              {"lo_orderdate"}, {"d_datekey"}, dates, "", {"part_revenue"})
          .partialAggregation({}, {"sum(part_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .project({"revenue"})
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ3Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_extendedprice", "lo_discount", "lo_orderdate", "lo_quantity"};
  std::vector<std::string> dateColumns = {
      "d_datekey", "d_year", "d_weeknuminyear"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter("(d_year = 1994) and (d_weeknuminyear = 6)")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .filter(
              "(lo_discount between 5 and 7) AND (lo_quantity between 26 and 35) ")
          .project(
              {"lo_extendedprice * lo_discount AS part_revenue",
               "lo_orderdate"})
          .hashJoin(
              {"lo_orderdate"}, {"d_datekey"}, dates, "", {"part_revenue"})
          .partialAggregation({}, {"sum(part_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .project({"revenue"})
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ4Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue", "lo_orderdate", "lo_partkey", "lo_suppkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> partColumns = {
      "p_brand1", "p_partkey", "p_category"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_region"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId partPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .planNode();
  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_region = 'ASIA'")
          .planNode();
  auto parts =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kPart, partSelectedRowType, partFileColumns, {}, {})
          .capturePlanNodeId(partPlanNodeId)
          .filter("p_category = 'MFGR#12'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue", "lo_partkey", "lo_suppkey", "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue", "lo_partkey", "d_year"})
          .hashJoin(
              {"lo_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"lo_revenue", "d_year", "p_brand1"})
          .project({"d_year", "lo_revenue", "p_brand1"})
          .partialAggregation(
              {"d_year", "p_brand1"}, {"sum(lo_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "p_brand1 ASC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[partPlanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ5Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue", "lo_orderdate", "lo_partkey", "lo_suppkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> partColumns = {"p_brand1", "p_partkey"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_region"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId partPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .planNode();
  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_region = 'ASIA'")
          .planNode();
  auto parts =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kPart, partSelectedRowType, partFileColumns, {}, {})
          .capturePlanNodeId(partPlanNodeId)
          .filter("p_brand1 between 'MFGR#2221' and 'MFGR#2228'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue", "lo_partkey", "lo_suppkey", "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue", "lo_partkey", "d_year"})
          .hashJoin(
              {"lo_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"lo_revenue", "d_year", "p_brand1"})
          .project({"d_year", "lo_revenue", "p_brand1"})
          .partialAggregation(
              {"d_year", "p_brand1"}, {"sum(lo_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "p_brand1 ASC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[partPlanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ6Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue", "lo_orderdate", "lo_partkey", "lo_suppkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> partColumns = {"p_brand1", "p_partkey"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_region"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId partPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .planNode();
  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_region = 'EUROPE'")
          .planNode();
  auto parts =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kPart, partSelectedRowType, partFileColumns, {}, {})
          .capturePlanNodeId(partPlanNodeId)
          .filter("p_brand1 = 'MFGR#2239'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue", "lo_partkey", "lo_suppkey", "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue", "lo_partkey", "d_year"})
          .hashJoin(
              {"lo_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"lo_revenue", "d_year", "p_brand1"})
          .project({"d_year", "lo_revenue", "p_brand1"})
          .partialAggregation(
              {"d_year", "p_brand1"}, {"sum(lo_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "p_brand1 ASC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[partPlanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ7Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue", "lo_orderdate", "lo_custkey", "lo_suppkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> customerColumns = {
      "c_nation", "c_custkey", "c_region"};
  std::vector<std::string> supplierColumns = {
      "s_suppkey", "s_region", "s_nation"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId customerPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter({"d_year >= 1992 and d_year <= 1997"})
          .planNode();

  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_region = 'ASIA'")
          .planNode();
  auto customers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kCustomer, customerSelectedRowType, customerFileColumns, {}, {})
          .capturePlanNodeId(customerPlanNodeId)
          .filter("c_region = 'ASIA'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue", "lo_custkey", "lo_suppkey", "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue", "lo_custkey", "d_year", "s_nation"})
          .hashJoin(
              {"lo_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"lo_revenue", "d_year", "s_nation", "c_nation"})
          .project({"d_year", "lo_revenue", "s_nation", "c_nation"})
          .partialAggregation(
              {"c_nation", "s_nation", "d_year"},
              {"sum(lo_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "revenue DESC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ8Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue", "lo_orderdate", "lo_custkey", "lo_suppkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> customerColumns = {
      "c_nation", "c_custkey", "c_city"};
  std::vector<std::string> supplierColumns = {
      "s_suppkey", "s_city", "s_nation"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId customerPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter({"d_year >= 1992 and d_year <= 1997"})
          .planNode();

  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_nation = 'UNITED STATES'")
          .planNode();
  auto customers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kCustomer, customerSelectedRowType, customerFileColumns, {}, {})
          .capturePlanNodeId(customerPlanNodeId)
          .filter("c_nation = 'UNITED STATES'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue", "lo_custkey", "lo_suppkey", "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue", "lo_custkey", "d_year", "s_city"})
          .hashJoin(
              {"lo_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"lo_revenue", "d_year", "s_city", "c_city"})
          .project({"d_year", "lo_revenue", "s_city", "c_city"})
          .partialAggregation(
              {"c_city", "s_city", "d_year"}, {"sum(lo_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "revenue DESC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ9Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue", "lo_orderdate", "lo_custkey", "lo_suppkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> customerColumns = {"c_custkey", "c_city"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_city"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId customerPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter({"d_year >= 1992 and d_year <= 1997"})
          .planNode();

  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_city='UNITED KI1' or s_city='UNITED KI5'")
          .planNode();
  auto customers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kCustomer, customerSelectedRowType, customerFileColumns, {}, {})
          .capturePlanNodeId(customerPlanNodeId)
          .filter("c_city='UNITED KI1' or c_city='UNITED KI5'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue", "lo_custkey", "lo_suppkey", "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue", "lo_custkey", "d_year", "s_city"})
          .hashJoin(
              {"lo_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"lo_revenue", "d_year", "s_city", "c_city"})
          .project({"d_year", "lo_revenue", "s_city", "c_city"})
          .partialAggregation(
              {"c_city", "s_city", "d_year"}, {"sum(lo_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "revenue DESC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ10Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue", "lo_orderdate", "lo_custkey", "lo_suppkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year", "d_yearmonth"};
  std::vector<std::string> customerColumns = {"c_custkey", "c_city"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_city"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId customerPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter({"d_yearmonth = 'Dec1997'"})
          .planNode();

  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_city='UNITED KI1' or s_city='UNITED KI5'")
          .planNode();
  auto customers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kCustomer, customerSelectedRowType, customerFileColumns, {}, {})
          .capturePlanNodeId(customerPlanNodeId)
          .filter("c_city='UNITED KI1' or c_city='UNITED KI5'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue", "lo_custkey", "lo_suppkey", "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue", "lo_custkey", "d_year", "s_city"})
          .hashJoin(
              {"lo_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"lo_revenue", "d_year", "s_city", "c_city"})
          .project({"d_year", "lo_revenue", "s_city", "c_city"})
          .partialAggregation(
              {"c_city", "s_city", "d_year"}, {"sum(lo_revenue) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "revenue DESC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ11Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue",
      "lo_supplycost",
      "lo_orderdate",
      "lo_custkey",
      "lo_suppkey",
      "lo_partkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> customerColumns = {
      "c_region", "c_custkey", "c_nation"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_region"};
  std::vector<std::string> partColumns = {"p_partkey", "p_mfgr"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId customerPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;
  core::PlanNodeId partPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .planNode();

  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_region = 'AMERICA'")
          .planNode();
  auto customers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kCustomer, customerSelectedRowType, customerFileColumns, {}, {})
          .capturePlanNodeId(customerPlanNodeId)
          .filter("c_region = 'AMERICA'")
          .planNode();
  auto parts =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kPart, partSelectedRowType, partFileColumns, {}, {})
          .capturePlanNodeId(partPlanNodeId)
          .filter("p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue",
               "lo_custkey",
               "lo_suppkey",
               "lo_partkey",
               "lo_supplycost",
               "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue",
               "lo_custkey",
               "lo_partkey",
               "lo_supplycost",
               "d_year"})
          .hashJoin(
              {"lo_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"lo_revenue",
               "lo_partkey",
               "d_year",
               "lo_supplycost",
               "c_nation"})
          .hashJoin(
              {"lo_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"lo_revenue", "d_year", "lo_supplycost", "c_nation"})
          .project(
              {"d_year",
               "c_nation",
               "(lo_revenue-lo_supplycost) AS part_profit"})
          .partialAggregation(
              {"d_year", "c_nation"}, {"sum(part_profit) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "c_nation ASC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[partPlanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ12Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue",
      "lo_supplycost",
      "lo_orderdate",
      "lo_custkey",
      "lo_suppkey",
      "lo_partkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> customerColumns = {"c_region", "c_custkey"};
  std::vector<std::string> supplierColumns = {
      "s_suppkey", "s_region", "s_nation"};
  std::vector<std::string> partColumns = {"p_partkey", "p_mfgr", "p_category"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId customerPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;
  core::PlanNodeId partPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter("d_year = 1997 or d_year = 1998")
          .planNode();

  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_region = 'AMERICA'")
          .planNode();
  auto customers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kCustomer, customerSelectedRowType, customerFileColumns, {}, {})
          .capturePlanNodeId(customerPlanNodeId)
          .filter("c_region = 'AMERICA'")
          .planNode();
  auto parts =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kPart, partSelectedRowType, partFileColumns, {}, {})
          .capturePlanNodeId(partPlanNodeId)
          .filter("p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue",
               "lo_custkey",
               "lo_suppkey",
               "lo_partkey",
               "lo_supplycost",
               "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue",
               "lo_custkey",
               "lo_partkey",
               "lo_supplycost",
               "d_year",
               "s_nation"})
          .hashJoin(
              {"lo_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"lo_revenue",
               "lo_partkey",
               "d_year",
               "lo_supplycost",
               "s_nation"})
          .hashJoin(
              {"lo_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"lo_revenue",
               "d_year",
               "lo_supplycost",
               "s_nation",
               "p_category"})
          .project(
              {"d_year",
               "s_nation",
               "p_category",
               "(lo_revenue-lo_supplycost) AS part_profit"})
          .partialAggregation(
              {"d_year", "s_nation", "p_category"},
              {"sum(part_profit) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "s_nation ASC", "p_category ASC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[partPlanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

SsbPlan SsbQueryBuilder::getQ13Plan() const {
  std::vector<std::string> lineorderColumns = {
      "lo_revenue",
      "lo_supplycost",
      "lo_orderdate",
      "lo_custkey",
      "lo_suppkey",
      "lo_partkey"};
  std::vector<std::string> dateColumns = {"d_datekey", "d_year"};
  std::vector<std::string> customerColumns = {"c_custkey"};
  std::vector<std::string> supplierColumns = {
      "s_suppkey", "s_nation", "s_city"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_brand1", "p_category"};

  const auto lineorderSelectedRowType =
      getRowType(kLineorder, lineorderColumns);
  const auto& lineorderFileColumns = getFileColumnNames(kLineorder);

  const auto dateSelectedRowType = getRowType(kDate, dateColumns);
  const auto& dateFileColumns = getFileColumnNames(kDate);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  // shipdate <= '1998-09-02'

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineorderPlanNodeId;
  core::PlanNodeId datePlanNodeId;
  core::PlanNodeId customerPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;
  core::PlanNodeId partPlanNodeId;

  auto dates =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kDate, dateSelectedRowType, dateFileColumns, {}, {})
          .capturePlanNodeId(datePlanNodeId)
          .filter("d_year = 1997 or d_year = 1998")
          .planNode();

  auto suppliers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kSupplier, supplierSelectedRowType, supplierFileColumns, {}, {})
          .capturePlanNodeId(supplierPlanNodeId)
          .filter("s_nation = 'UNITED STATES'")
          .planNode();
  auto customers =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              kCustomer, customerSelectedRowType, customerFileColumns, {}, {})
          .capturePlanNodeId(customerPlanNodeId)
          .planNode();
  auto parts =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kPart, partSelectedRowType, partFileColumns, {}, {})
          .capturePlanNodeId(partPlanNodeId)
          .filter("p_category = 'MFGR#14'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(kLineorder, lineorderSelectedRowType, lineorderFileColumns)
          .capturePlanNodeId(lineorderPlanNodeId)
          .hashJoin(
              {"lo_orderdate"},
              {"d_datekey"},
              dates,
              "",
              {"lo_revenue",
               "lo_custkey",
               "lo_suppkey",
               "lo_partkey",
               "lo_supplycost",
               "d_year"})
          .hashJoin(
              {"lo_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"lo_revenue",
               "lo_custkey",
               "lo_partkey",
               "lo_supplycost",
               "d_year",
               "s_city"})
          .hashJoin(
              {"lo_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"lo_revenue", "lo_partkey", "d_year", "lo_supplycost", "s_city"})
          .hashJoin(
              {"lo_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"lo_revenue", "d_year", "lo_supplycost", "s_city", "p_brand1"})
          .project(
              {"d_year",
               "s_city",
               "p_brand1",
               "(lo_revenue-lo_supplycost) AS part_profit"})
          .partialAggregation(
              {"d_year", "s_city", "p_brand1"}, {"sum(part_profit) as revenue"})
          .localPartition({})
          .finalAggregation()
          .orderBy({"d_year ASC", "s_city ASC", "p_brand1 ASC"}, false)
          .planNode();

  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[datePlanNodeId] = getTableFilePaths(kDate);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[partPlanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[lineorderPlanNodeId] = getTableFilePaths(kLineorder);
  context.dataFileFormat = format_;
  return context;
}

/*SsbPlan SsbQueryBuilder::getQ23Plan() const {
  std::vector<std::string> selectedColumns =
{"l_orderkey","l_partkey","l_suppkey","l_linenumber","l_returnflag","l_linestatus","l_receiptdate","l_shipinstruct","l_shipmode",
      "l_shipdate", "l_extendedprice", "l_quantity",
"l_discount","l_tax","l_commitdate","l_comment"};

  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);


  core::PlanNodeId lineitemPlanNodeId;
  auto plan = PlanBuilder()
                  .tableScan(
                      kLineitem,
                      selectedRowType,
                      fileColumnNames)
                  .capturePlanNodeId(lineitemPlanNodeId)
                  //.project({"l_extendedprice * l_discount*l_quantity"})
                  //.partialAggregation({}, {"sum(p0)"})
                  //.localPartition({})
                  //.finalAggregation()
                  .planNode();
  SsbPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}*/

const std::vector<std::string> SsbQueryBuilder::kTableNames_ =
    {kLineorder, kSupplier, kCustomer, kPart, kDate};

const std::unordered_map<std::string, std::vector<std::string>>
    SsbQueryBuilder::kTables_ = {
        std::make_pair(
            "lineorder",
            ssb::getTableSchema(ssb::SSB_Table::TBL_LINEORDER)->names()),
        std::make_pair(
            "customer",
            ssb::getTableSchema(ssb::SSB_Table::TBL_CUSTOMER)->names()),
        std::make_pair(
            "supplier",
            ssb::getTableSchema(ssb::SSB_Table::TBL_SUPPLIER)->names()),
        std::make_pair(
            "part",
            ssb::getTableSchema(ssb::SSB_Table::TBL_PART)->names()),
        std::make_pair(
            "date",
            ssb::getTableSchema(ssb::SSB_Table::TBL_DATE)->names())};

} // namespace facebook::velox::exec::test
