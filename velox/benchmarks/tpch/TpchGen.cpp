
#include <filesystem>
#include <iostream>

#include <folly/init/Init.h>
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/tpch/gen/TpchGen.h"

using namespace facebook::velox;
using facebook::velox::parquet::Writer;

namespace {

constexpr size_t kScaleFactor = 1;
constexpr size_t kRowsPerSplit = 10'000;

RowVectorPtr getTpchData(
    tpch::Table table,
    size_t maxRows,
    size_t offset,
    double scaleFactor,
    memory::MemoryPool* pool) {
  // TODO: This is basically copied from TpchConnector.cpp, because it is in an
  // anonymous namespace there. This should be made available so we can reuse
  // it.
  switch (table) {
    case tpch::Table::TBL_PART:
      return tpch::genTpchPart(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_SUPPLIER:
      return tpch::genTpchSupplier(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_PARTSUPP:
      return tpch::genTpchPartSupp(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_CUSTOMER:
      return tpch::genTpchCustomer(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_ORDERS:
      return tpch::genTpchOrders(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_LINEITEM:
      return tpch::genTpchLineItem(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_NATION:
      return tpch::genTpchNation(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_REGION:
      return tpch::genTpchRegion(maxRows, offset, scaleFactor, pool);
  }
}

void generateTable(
    tpch::Table table,
    const std::filesystem::path& dataDirectory) {
  auto pool = memory::getDefaultMemoryPool();
  const std::string tableName = std::string{tpch::toTableName(table)};

  const std::filesystem::path& tableDirectory = dataDirectory / tableName;
  std::filesystem::create_directories(tableDirectory);
  const std::filesystem::path& filePath = tableDirectory / "001.parquet";

  Writer writer{dwio::common::DataSink::create(filePath), *pool, kRowsPerSplit};

  const size_t numRows = tpch::getRowCount(table, kScaleFactor);

  // Make sure we include all rows and don't forget some at the end.
  const size_t endOffset = numRows + kRowsPerSplit;
  for (size_t offset = 0; offset < endOffset; offset += kRowsPerSplit) {
    auto data =
        getTpchData(table, kRowsPerSplit, offset, kScaleFactor, pool.get());

    if (offset == 0) {
      std::cout << std::endl
                << "> first 10 rows from TPC-H " << tableName
                << " table: " << data->toString() << std::endl;
      std::cout << data->toString(0, 10) << std::endl;
    }

    writer.write(data);
    writer.flush();
    std::cout << "written offset " << offset << " (/" << numRows << ")"
              << std::endl;
  }

  writer.close();
}

void generateOrdersAndLineitems(const std::filesystem::path& dataDirectory) {
  auto pool = memory::getDefaultMemoryPool();
  const std::string lineitemTableName =
      std::string{tpch::toTableName(tpch::Table::TBL_LINEITEM)};
  const std::string ordersTableName =
      std::string{tpch::toTableName(tpch::Table::TBL_ORDERS)};

  const std::filesystem::path& lineitemDirectory =
      dataDirectory / lineitemTableName;
  const std::filesystem::path& ordersDirectory =
      dataDirectory / ordersTableName;
  std::filesystem::create_directories(lineitemDirectory);
  std::filesystem::create_directories(ordersDirectory);
  const std::filesystem::path& lineitemFilePath =
      lineitemDirectory / "001.parquet";
  const std::filesystem::path& ordersFilePath = ordersDirectory / "001.parquet";

  Writer lineitemWriter{
      dwio::common::DataSink::create(lineitemFilePath), *pool, kRowsPerSplit};
  Writer ordersWriter{
      dwio::common::DataSink::create(ordersFilePath), *pool, kRowsPerSplit};

  const size_t numOrderRows =
      tpch::getRowCount(facebook::velox::tpch::Table::TBL_ORDERS, kScaleFactor);

  size_t expectedLineitemRows = tpch::getRowCount(
      facebook::velox::tpch::Table::TBL_LINEITEM, kScaleFactor);
  size_t numGeneratedLineitemRows = 0;

  // Make sure we include all rows and don't forget some at the end.
  const size_t endOffset = numOrderRows + kRowsPerSplit;
  for (size_t offset = 0; offset < endOffset; offset += kRowsPerSplit) {
    auto orderData = getTpchData(
        facebook::velox::tpch::Table::TBL_ORDERS,
        kRowsPerSplit,
        offset,
        kScaleFactor,
        pool.get());

    auto lineitemData = getTpchData(
        facebook::velox::tpch::Table::TBL_LINEITEM,
        kRowsPerSplit,
        offset,
        kScaleFactor,
        pool.get());

    ordersWriter.write(orderData);
    ordersWriter.flush();

    numGeneratedLineitemRows += lineitemData->size();

    lineitemWriter.write(lineitemData);
    lineitemWriter.flush();

    std::cout << "written offset " << offset << " (/" << numOrderRows << ")"
              << std::endl;
  }

  VELOX_CHECK_EQ(numGeneratedLineitemRows, expectedLineitemRows);

  ordersWriter.close();
  lineitemWriter.close();
}

} // namespace

int main() {
  const std::filesystem::path dataDirectory = "/tmp/tpch-data ";
  std::filesystem::create_directories(dataDirectory);

  // We need to create these together, as lineitems are created per order.
  // generateTable(tpch::Table::TBL_LINEITEM, dataDirectory);
  // generateTable(tpch::Table::TBL_ORDERS, dataDirectory);
  generateOrdersAndLineitems(dataDirectory);

  generateTable(tpch::Table::TBL_PART, dataDirectory);
  generateTable(tpch::Table::TBL_SUPPLIER, dataDirectory);
  generateTable(tpch::Table::TBL_PARTSUPP, dataDirectory);
  generateTable(tpch::Table::TBL_CUSTOMER, dataDirectory);
  generateTable(tpch::Table::TBL_NATION, dataDirectory);
  generateTable(tpch::Table::TBL_REGION, dataDirectory);
}
