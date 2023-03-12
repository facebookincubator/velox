#include <filesystem>
#include <iostream>

#include <folly/init/Init.h>
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/tpch/gen/TpchGen.h"

using namespace facebook::velox;
using facebook::velox::parquet::Writer;

namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}
} // namespace

DEFINE_int32(scale_factor, 1, "TPC-H scale factor to use.");
DEFINE_int32(
    rows_per_row_group,
    10'000,
    "Number of rows per row group in Parquet file.");
DEFINE_int32(
    rows_per_flush,
    100'000,
    "Minimum number of rows to buffer in memory before writing to disk.");
DEFINE_string(data_dir, "", "Output directory to write data to.");

DEFINE_validator(data_dir, &notEmpty);

namespace {

struct TpchGenerationParameters {
  std::filesystem::path dataDirectory;
  int32_t scaleFactor;
  int32_t rowsPerGroup;
  int32_t rowsPerFlush;
};

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

void generateTable(tpch::Table table, const TpchGenerationParameters& params) {
  auto pool = memory::getDefaultMemoryPool();
  const std::string tableName = std::string{tpch::toTableName(table)};

  const std::filesystem::path& tableDirectory =
      params.dataDirectory / tableName;
  std::filesystem::create_directories(tableDirectory);
  const std::filesystem::path& filePath = tableDirectory / "001.parquet";

  auto writerProperties = ::parquet::WriterProperties::Builder()
                              .max_row_group_length(params.rowsPerGroup)
                              ->build();
  Writer writer{
      dwio::common::DataSink::create(filePath),
      *pool,
      params.rowsPerGroup,
      writerProperties};

  const uint64_t numRows = tpch::getRowCount(table, params.scaleFactor);

  size_t offset = 0;
  uint64_t rowCount = 0;

  uint64_t rowsInChunkToFlush = 0;

  while (rowCount < numRows) {
    auto data = getTpchData(
        table, params.rowsPerGroup, offset, params.scaleFactor, pool.get());

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

    offset += params.rowsPerGroup;
    rowCount += data->size();

    rowsInChunkToFlush += data->size();
    if (rowsInChunkToFlush >= params.rowsPerFlush) {
      writer.flush();
      rowsInChunkToFlush = 0;
      std::cout << "\rWritten rows " << rowCount << "/" << numRows
                << " (offset: " << offset << ")" << std::flush;
    }
  }
  writer.close();
}

} // namespace

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);

  const std::filesystem::path dataDirectory = FLAGS_data_dir;
  std::filesystem::create_directories(dataDirectory);

  TpchGenerationParameters params{
      dataDirectory,
      FLAGS_scale_factor,
      FLAGS_rows_per_row_group,
      FLAGS_rows_per_flush};

  generateTable(tpch::Table::TBL_NATION, params);
  generateTable(tpch::Table::TBL_LINEITEM, params);
  generateTable(tpch::Table::TBL_ORDERS, params);
  generateTable(tpch::Table::TBL_PART, params);
  generateTable(tpch::Table::TBL_SUPPLIER, params);
  generateTable(tpch::Table::TBL_PARTSUPP, params);
  generateTable(tpch::Table::TBL_CUSTOMER, params);
  generateTable(tpch::Table::TBL_REGION, params);
}
