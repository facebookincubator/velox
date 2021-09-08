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
#include "velox/dwio/parquet/reader/Statistics.h"

namespace facebook::velox::parquet {

namespace {

void toDuckDbFilter(
    uint64_t colIdx,
    common::Filter* filter,
    ::duckdb::TableFilterSet& filters) {
  switch (filter->kind()) {
    case common::FilterKind::kBigintRange: {
      auto rangeFilter = dynamic_cast<common::BigintRange*>(filter);
      if (rangeFilter->lower() != std::numeric_limits<int64_t>::min()) {
        filters.PushFilter(
            colIdx,
            std::make_unique<::duckdb::ConstantFilter>(
                ::duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                ::duckdb::Value(rangeFilter->lower())));
      }
      if (rangeFilter->upper() != std::numeric_limits<int64_t>::max()) {
        filters.PushFilter(
            colIdx,
            std::make_unique<::duckdb::ConstantFilter>(
                ::duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO,
                ::duckdb::Value(rangeFilter->upper())));
      }
    } break;

    case common::FilterKind::kAlwaysFalse:
    case common::FilterKind::kAlwaysTrue:
    case common::FilterKind::kIsNull:
    case common::FilterKind::kIsNotNull:
    case common::FilterKind::kBoolValue:
    case common::FilterKind::kBigintValuesUsingHashTable:
    case common::FilterKind::kBigintValuesUsingBitmask:
    case common::FilterKind::kDoubleRange:
    case common::FilterKind::kFloatRange:
    case common::FilterKind::kBytesRange:
    case common::FilterKind::kBytesValues:
    case common::FilterKind::kBigintMultiRange:
    case common::FilterKind::kMultiRange:
    default:
      VELOX_UNSUPPORTED(
          "Unsupported filter in parquet reader: {}", filter->toString());
  }
}

} // anonymous namespace

ParquetRowReader::ParquetRowReader(
    std::shared_ptr<::duckdb::ParquetReader> reader,
    const dwio::common::RowReaderOptions& options,
    memory::MemoryPool& pool)
    : reader_(std::move(reader)), pool_(pool) {
  auto& selector = *options.getSelector();
  rowType_ = selector.buildSelectedReordered();

  std::vector<::duckdb::column_t> columnIds(rowType_->size());
  duckdbRowType_.resize(rowType_->size());
  for (size_t i = 0; i < reader_->names.size(); i++) {
    auto& filter = *selector.findColumn(i);
    if (filter.shouldRead()) {
      columnIds[filter.getProjectOrder()] = i;
      duckdbRowType_[filter.getProjectOrder()] = reader_->return_types[i];
    }
  }

  std::vector<idx_t> groups;
  for (idx_t i = 0; i < reader_->NumRowGroups(); i++) {
    auto groupOffset = reader_->GetFileMetadata()->row_groups[i].file_offset;
    if (groupOffset >= options.getOffset() &&
        groupOffset < (options.getLength() + options.getOffset())) {
      groups.push_back(i);
    }
  }

  auto& scanSpec = *options.getScanSpec();
  for (auto& colSpec : scanSpec.children()) {
    VELOX_CHECK(
        !colSpec->extractValues(), "Subfield access is NYI in parquet reader");
    if (colSpec->filter()) {
      // TODO: remove linear search
      uint64_t colIdx = std::find(
                            reader_->names.begin(),
                            reader_->names.end(),
                            colSpec->fieldName()) -
          reader_->names.begin();
      VELOX_CHECK(
          colIdx < reader_->names.size(),
          "Unexpected columns name: {}",
          colSpec->fieldName());
      toDuckDbFilter(colIdx, colSpec->filter(), filters_);
    }
  }

  reader_->InitializeScan(
      state_, std::move(columnIds), std::move(groups), &filters_);
}

uint64_t ParquetRowReader::next(uint64_t size, velox::VectorPtr& result) {
  ::duckdb::DataChunk output;
  output.Initialize(duckdbRowType_);

  reader_->Scan(state_, output);

  if (output.size() > 0) {
    std::vector<VectorPtr> columns;
    columns.reserve(output.data.size());
    for (int i = 0; i < output.data.size(); i++) {
      columns.emplace_back(duckdb::toVeloxVector(
          output.size(), output.data[i], rowType_->childAt(i), &pool_));
    }

    result = std::make_shared<RowVector>(
        &pool_,
        rowType_,
        BufferPtr(nullptr),
        columns[0]->size(),
        columns,
        std::nullopt);
  }

  return output.size();
}

void ParquetRowReader::resetFilterCaches() {
  VELOX_FAIL("ParquetRowReader::resetFilterCaches is NYI");
}

size_t ParquetRowReader::estimatedRowSize() const {
  VELOX_FAIL("ParquetRowReader::estimatedRowSize is NYI");
  return 0;
}

ParquetReader::ParquetReader(
    std::unique_ptr<dwio::common::InputStream> stream,
    const dwio::common::ReaderOptions& options)
    : reader_(std::make_shared<::duckdb::ParquetReader>(
          allocator_,
          fileSystem_.OpenStream(std::move(stream)))),
      pool_(options.getMemoryPool()) {
  auto names = reader_->names;
  std::vector<TypePtr> types;
  types.reserve(reader_->return_types.size());
  for (auto& t : reader_->return_types) {
    types.emplace_back(duckdb::toVeloxType(t));
  }
  type_ = ROW(std::move(names), std::move(types));
}

std::optional<uint64_t> ParquetReader::getNumberOfRows() const {
  return const_cast<::duckdb::ParquetReader*>(reader_.get())->NumRows();
}

std::unique_ptr<velox::dwrf::ColumnStatistics>
ParquetReader::getColumnStatistics(uint32_t index) const {
  // TODO: implement proper stats
  return std::make_unique<ColumnStatistics>();
}

const std::shared_ptr<const velox::RowType>& ParquetReader::getType() const {
  return type_;
}

const std::shared_ptr<const dwio::common::TypeWithId>&
ParquetReader::getTypeWithId() const {
  if (!typeWithId_) {
    typeWithId_ = dwio::common::TypeWithId::create(type_);
  }
  return typeWithId_;
}

std::unique_ptr<dwio::common::RowReader> ParquetReader::createRowReader(
    const dwio::common::RowReaderOptions& options) const {
  return std::make_unique<ParquetRowReader>(reader_, options, pool_);
}

duckdb::InputStreamFileSystem ParquetReader::fileSystem_;

VELOX_REGISTER_READER_FACTORY(std::make_shared<ParquetReaderFactory>())

} // namespace facebook::velox::parquet
