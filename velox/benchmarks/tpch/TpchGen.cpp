#include <filesystem>
#include <iostream>

#include <folly/init/Init.h>
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/tpch/gen/TpchGen.h"

using namespace facebook::velox;
using facebook::velox::parquet::Writer;

namespace {

constexpr double kScaleFactor = 10;
constexpr size_t kWriteBatchSize = 10'000;
constexpr size_t kRowsInRowgroup = 10'000;

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

  auto writerProperties = ::parquet::WriterProperties::Builder()
                              .max_row_group_length(kRowsInRowgroup)
                              ->build();
  Writer writer{
      dwio::common::DataSink::create(filePath), *pool, kRowsInRowgroup};

  const uint64_t numRows = tpch::getRowCount(table, kScaleFactor);

  size_t offset = 0;
  uint64_t rowCount = 0;
  while (rowCount < numRows) {
    auto data =
        getTpchData(table, kWriteBatchSize, offset, kScaleFactor, pool.get());

    if (offset == 0) {
      std::cout << std::endl
                << "> first 10 rows from TPC-H " << tableName
                << " table: " << data->toString() << std::endl;
      std::cout << data->toString(0, 10) << std::endl;
    }
    // Resize is for lineitems table since the rowCount can exceed the numRows.
    data->resize(
        std::min((numRows - rowCount), static_cast<uint64_t>(data->size())));
    writer.write(data);
    offset += kWriteBatchSize;
    rowCount += data->size();

    std::cout << "written row (" << rowCount << "/"  << numRows << ")" << " (offset: " << offset << ")"
              << std::endl;
  }
  writer.close();
}

} // namespace

int main() {
  const std::filesystem::path dataDirectory = "/tmp/tpch-sf10-new";
  std::filesystem::create_directories(dataDirectory);

  generateTable(tpch::Table::TBL_LINEITEM, dataDirectory);
  generateTable(tpch::Table::TBL_ORDERS, dataDirectory);
  generateTable(tpch::Table::TBL_PART, dataDirectory);
  generateTable(tpch::Table::TBL_SUPPLIER, dataDirectory);
  generateTable(tpch::Table::TBL_PARTSUPP, dataDirectory);
  generateTable(tpch::Table::TBL_CUSTOMER, dataDirectory);
  generateTable(tpch::Table::TBL_NATION, dataDirectory);
  generateTable(tpch::Table::TBL_REGION, dataDirectory);
}
