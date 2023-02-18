
#include <filesystem>
#include <iostream>

#include <folly/init/Init.h>
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/tpch/gen/TpchGen.h"

using namespace facebook::velox;
using facebook::velox::parquet::Writer;

namespace {

constexpr size_t SCALE_FACTOR = 1;
constexpr size_t ROWS_PER_SPLIT = 10'000;

RowVectorPtr getTpchData(
    tpch::Table table,
    size_t maxRows,
    size_t offset,
    double scaleFactor,
    memory::MemoryPool* pool) {
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

void generateTable(tpch::Table table, const std::filesystem::path& data_dir) {
  auto pool = memory::getDefaultMemoryPool();
  const std::string table_name = std::string{tpch::toTableName(table)};

  const std::filesystem::path& table_dir = data_dir / table_name;
  std::filesystem::create_directories(table_dir);
  const std::filesystem::path& file_path = table_dir / "001.parquet";

  Writer writer{
      dwio::common::DataSink::create(file_path), *pool, ROWS_PER_SPLIT};

  const size_t num_rows = tpch::getRowCount(table, SCALE_FACTOR);

  // Make sure we include all rows and don't forget some at the end.
  const size_t end_offset = num_rows + ROWS_PER_SPLIT;
  for (size_t offset = 0; offset < end_offset; offset += ROWS_PER_SPLIT) {
    auto data =
        getTpchData(table, ROWS_PER_SPLIT, offset, SCALE_FACTOR, pool.get());

//    if (offset == 0) {
//      std::cout << std::endl
//                << "> first 10 rows from TPC-H " << table_name
//                << " table: " << data->toString() << std::endl;
//      std::cout << data->toString(0, 10) << std::endl;
//    }

    writer.write(data);
    writer.flush();
    std::cout << "written offset " << offset << " (/" << num_rows << ")" << std::endl;
  }

  writer.close();
}

} // namespace

int main() {
  const std::filesystem::path data_dir = "/tmp/tpch-data";
  std::filesystem::create_directories(data_dir);

  generateTable(tpch::Table::TBL_LINEITEM, data_dir);
  generateTable(tpch::Table::TBL_ORDERS, data_dir);
  generateTable(tpch::Table::TBL_PART, data_dir);
  generateTable(tpch::Table::TBL_SUPPLIER, data_dir);
  generateTable(tpch::Table::TBL_PARTSUPP, data_dir);
  generateTable(tpch::Table::TBL_CUSTOMER, data_dir);
  generateTable(tpch::Table::TBL_NATION, data_dir);
  generateTable(tpch::Table::TBL_REGION, data_dir);
}
