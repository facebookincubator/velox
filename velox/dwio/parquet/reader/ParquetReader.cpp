/*
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

#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/duckdb/conversion/DuckConversion.h"
#include "velox/duckdb/conversion/DuckWrapper.h"

namespace facebook::velox::parquet {

ParquetRowReader::ParquetRowReader(
    std::shared_ptr<::duckdb::ParquetReader> _reader,
    const dwio::common::RowReaderOptions& options,
    memory::MemoryPool& _pool)
    : reader(std::move(_reader)), pool(_pool) {
  auto& selector = *options.getSelector();
  rowType = selector.buildSelectedReordered();

  std::vector<::duckdb::column_t> columnIds(rowType->size());
  duckdbRowType.resize(rowType->size());
  for (size_t i = 0; i < reader->names.size(); i++) {
    auto& filter = *selector.findColumn(i);
    if (filter.shouldRead()) {
      columnIds[filter.getProjectOrder()] = i;
      duckdbRowType[filter.getProjectOrder()] = reader->return_types[i];
    }
  }

  // TODO: select proper groups
  std::vector<idx_t> groups;
  for (idx_t i = 0; i < reader->NumRowGroups(); i++) {
    groups.push_back(i);
  }

  // TODO: set filters
  reader->InitializeScan(
      state, std::move(columnIds), std::move(groups), nullptr);
}

uint64_t ParquetRowReader::seekToRow(uint64_t rowNumber) {
  VELOX_CHECK(false, "ParquetRowReader::seekToRow is NYI");
  return 0;
}

uint64_t ParquetRowReader::next(uint64_t size, velox::VectorPtr& result) {
  ::duckdb::DataChunk output;
  output.Initialize(duckdbRowType);

  reader->Scan(state, output);

  if (output.size() > 0) {
    std::vector<VectorPtr> columns;
    columns.reserve(output.data.size());
    for (int i = 0; i < output.data.size(); i++) {
      columns.emplace_back(duckdb::toVeloxVector(
          output.size(), output.data[i], rowType->childAt(i), &pool));
    }

    result = std::make_shared<RowVector>(
        &pool,
        rowType,
        BufferPtr(nullptr),
        columns[0]->size(),
        columns,
        folly::none);
  }

  return output.size();
}

size_t ParquetRowReader::estimatedRowSize() const {
  VELOX_CHECK(false, "ParquetRowReader::estimatedRowSize is NYI");
  return 0;
}

ParquetReader::ParquetReader(
    std::unique_ptr<dwio::common::InputStream> stream,
    const dwio::common::ReaderOptions& options)
    : reader(std::make_shared<::duckdb::ParquetReader>(
          allocator,
          fileSystem.OpenStream(std::move(stream)))),
      pool(options.getMemoryPool()) {
  auto names = reader->names;
  std::vector<TypePtr> types;
  types.reserve(reader->return_types.size());
  for (auto& t : reader->return_types) {
    types.emplace_back(duckdb::toVeloxType(t));
  }
  type = ROW(std::move(names), std::move(types));
}

std::optional<uint64_t> ParquetReader::getNumberOfRows() const {
  return const_cast<::duckdb::ParquetReader*>(reader.get())->NumRows();
}

const std::shared_ptr<const velox::RowType>& ParquetReader::getType() const {
  return type;
}

const std::shared_ptr<const dwio::common::TypeWithId>&
ParquetReader::getTypeWithId() const {
  if (!typeWithId) {
    typeWithId = dwio::common::TypeWithId::create(type);
  }
  return typeWithId;
}

std::unique_ptr<dwio::common::RowReader> ParquetReader::createRowReader(
    const dwio::common::RowReaderOptions& options) const {
  return std::make_unique<ParquetRowReader>(reader, options, pool);
}

duckdb::InputStreamFileSystem ParquetReader::fileSystem;

VELOX_REGISTER_READER_FACTORY(std::make_shared<ParquetReaderFactory>())

} // namespace facebook::velox::parquet
