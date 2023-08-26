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

#include "velox/experimental/query/LocalSchema.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/experimental/query/QueryGraph.h"

#include "velox/common/base/Fs.h"
#include "velox/dwio/dwrf/writer/StatisticsBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

namespace facebook::verax {
using namespace facebook::velox;

LocalSchema::LocalSchema(
    const std::string& path,
    velox::dwio::common::FileFormat fmt,
    velox::connector::hive::HiveConnector* hiveConnector,
    std::shared_ptr<velox::connector::ConnectorQueryCtx> ctx)
    : hiveConnector_(hiveConnector),
      connectorId_(hiveConnector_->connectorId()),
      connectorQueryCtx_(ctx),
      pool_(connectorQueryCtx_->memoryPool()) {
  format_ = fmt;
  initialize(path);
  locus_ = std::make_unique<Locus>(connectorId_.c_str());
}

void LocalSchema::initialize(const std::string& path) {
  for (auto const& dirEntry : fs::directory_iterator{path}) {
    if (!dirEntry.is_directory() ||
        dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    readTable(dirEntry.path().filename(), dirEntry.path());
  }
}

template <typename Builder, typename BuildT, typename T>
void addStats(
    velox::dwrf::StatisticsBuilder* builder,
    const BaseVector& vector) {
  auto typedVector = vector.asUnchecked<SimpleVector<T>>();
  for (auto i = 0; i < typedVector->size(); ++i) {
    if (typedVector->isNullAt(i)) {
    } else {
      reinterpret_cast<Builder*>(builder)->addValues(typedVector->valueAt(i));
    }
  }
}

std::pair<int64_t, int64_t> LocalTable::sample(
    float pct,
    const std::vector<common::Subfield>& fields,
    velox::connector::hive::SubfieldFilters filters,
    const velox::core::TypedExprPtr& remainingFilter,
    HashStringAllocator* allocator,
    std::vector<std::unique_ptr<velox::dwrf::StatisticsBuilder>>* stats) {
  dwrf::StatisticsBuilderOptions options(100, 0, true, allocator);
  std::vector<std::unique_ptr<dwrf::StatisticsBuilder>> builders;
  auto tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
      schema->connector()->connectorId(),
      name,
      true,
      std::move(filters),
      remainingFilter);

  std::unordered_map<
      std::string,
      std::shared_ptr<velox::connector::ColumnHandle>>
      columnHandles;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto& field : fields) {
    VELOX_CHECK(
        allocator && stats,
        "Must specify allocator and stats return if specifying fields");
    auto& path = field.path();
    auto column =
        dynamic_cast<const common::Subfield::NestedField*>(path[0].get())
            ->name();
    auto idx = type->getChildIdx(column);
    names.push_back(type->nameOf(idx));
    types.push_back(type->childAt(idx));
    columnHandles[names.back()] =
        std::make_shared<connector::hive::HiveColumnHandle>(
            names.back(),
            connector::hive::HiveColumnHandle::ColumnType::kRegular,
            types.back(),
            types.back());
    switch (types.back()->kind()) {
      case TypeKind::BIGINT:
      case TypeKind::INTEGER:
      case TypeKind::SMALLINT:
        builders.push_back(
            std::make_unique<dwrf::IntegerStatisticsBuilder>(options));
        break;
      case TypeKind::REAL:
      case TypeKind::DOUBLE:
        builders.push_back(
            std::make_unique<dwrf::DoubleStatisticsBuilder>(options));
        break;
      case TypeKind::VARCHAR:
        builders.push_back(
            std::make_unique<dwrf::StringStatisticsBuilder>(options));
        break;

      default:
        builders.push_back(nullptr);
    }
  }
  auto outputType = ROW(std::move(names), std::move(types));
  int64_t passingRows = 0;
  int64_t scannedRows = 0;
  for (auto& file : files) {
    auto dataSource = schema->connector()->createDataSource(
        outputType,
        tableHandle,
        columnHandles,
        schema->connectorQueryCtx().get());

    auto split =
        exec::test::HiveConnectorSplitBuilder(file).fileFormat(format).build();
    dataSource->addSplit(split);
    constexpr int32_t kBatchSize = 1000;
    for (;;) {
      ContinueFuture ignore{ContinueFuture::makeEmpty()};

      auto data = dataSource->next(kBatchSize, ignore).value();
      if (data == nullptr) {
        break;
      }
      passingRows += data->size();
      for (auto column = 0; column < builders.size(); ++column) {
        if (!builders[column]) {
          continue;
        }
        auto builder = builders[column].get();
        switch (type->childAt(column)->kind()) {
          case TypeKind::SMALLINT:
            addStats<dwrf::IntegerStatisticsBuilder, int64_t, short>(
                builder, *data->childAt(column));
            break;
          case TypeKind::INTEGER:
            addStats<dwrf::IntegerStatisticsBuilder, int64_t, int32_t>(
                builder, *data->childAt(column));
            break;
          case TypeKind::BIGINT:
            addStats<dwrf::IntegerStatisticsBuilder, int64_t, int64_t>(
                builder, *data->childAt(column));
            break;
          case TypeKind::REAL:
            addStats<dwrf::DoubleStatisticsBuilder, double, float>(
                builder, *data->childAt(column));
            break;
          case TypeKind::DOUBLE:
            addStats<dwrf::DoubleStatisticsBuilder, double, double>(
                builder, *data->childAt(column));
            break;
          case TypeKind::VARCHAR:
            addStats<
                dwrf::StringStatisticsBuilder,
                folly::StringPiece,
                StringView>(builder, *data->childAt(column));
            break;

          default:
            VELOX_UNREACHABLE();
        }
      }
      break;
    }
    scannedRows += dataSource->getCompletedRows();
    if (scannedRows > numRows * (pct / 100)) {
      break;
    }
  }
  if (stats) {
    *stats = std::move(builders);
  }
  return std::pair(scannedRows, passingRows);
}

void LocalSchema::readTable(
    const std::string& tableName,
    const fs::path& tablePath) {
  RowTypePtr tableType;
  LocalTable* table = nullptr;

  for (auto const& dirEntry : fs::directory_iterator{tablePath}) {
    if (!dirEntry.is_regular_file()) {
      continue;
    }
    // Ignore hidden files.
    if (dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    auto it = tables_.find(tableName);
    if (it != tables_.end()) {
      table = it->second.get();
    } else {
      tables_[tableName] =
          std::make_unique<LocalTable>(tableName, format_, this);
      table = tables_[tableName].get();
    }
    dwio::common::ReaderOptions readerOptions{pool_};
    readerOptions.setFileFormat(format_);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(dirEntry.path().string()),
        readerOptions.getMemoryPool());
    std::unique_ptr<dwio::common::Reader> reader =
        dwio::common::getReaderFactory(readerOptions.getFileFormat())
            ->createReader(std::move(input), readerOptions);
    const auto fileType = reader->rowType();
    if (!tableType) {
      tableType = fileType;
    }
    auto rows = reader->numberOfRows();
    if (rows.has_value()) {
      table->numRows += rows.value();
    }
    for (auto i = 0; i < fileType->size(); ++i) {
      auto name = fileType->nameOf(i);
      LocalColumn* column;
      auto columnIt = table->columns.find(name);
      if (columnIt != table->columns.end()) {
        column = columnIt->second.get();
      } else {
        table->columns[name] =
            std::make_unique<LocalColumn>(name, fileType->childAt(i));
        column = table->columns[name].get();
      }
      column->addStats(reader->columnStatistics(i));
    }
    table->files.push_back(dirEntry.path());
  }
  if (table) {
    table->type = tableType;
    std::vector<common::Subfield> fields;
    for (auto i = 0; i < tableType->size(); ++i) {
      fields.push_back(common::Subfield(tableType->nameOf(i)));
    }
    auto allocator = std::make_unique<HashStringAllocator>(pool());
    std::vector<std::unique_ptr<dwrf::StatisticsBuilder>> stats;
    auto [sampled, passed] =
        table->sample(2, fields, {}, nullptr, allocator.get(), &stats);
    table->numSampledRows = sampled;
    for (auto i = 0; i < stats.size(); ++i) {
      if (stats[i]) {
        int64_t cardinality = stats[i]->cardinality();
        if (table->numSampledRows < table->numRows) {
          if (cardinality > sampled / 50) {
            float numDups =
                table->numSampledRows / static_cast<float>(cardinality);
            cardinality =
                std::min<float>(table->numRows, table->numRows / numDups);
            if (auto ints = dynamic_cast<dwrf::IntegerStatisticsBuilder*>(
                    stats[i].get())) {
              auto min = ints->getMinimum();
              auto max = ints->getMaximum();
              if (min.has_value() && max.has_value()) {
                auto range = max.value() - min.value();
                cardinality = std::min<float>(cardinality, range);
              }
            }
          }
        }
        table->columns[tableType->nameOf(i)]->numDistinct = cardinality;
      }
    }
  }
}

void LocalColumn::addStats(
    std::unique_ptr<dwio::common::ColumnStatistics> _stats) {
  if (!stats && stats) {
    stats = std::move(_stats);
  }
}

void LocalSchema::fetchSchemaTable(
    std::string_view name,
    const Schema* schema) {
  auto str = std::string(name);
  auto it = tables_.find(str);
  if (it == tables_.end()) {
    return;
  }
  auto table = it->second.get();
  Declare(SchemaTable, schemaTable, toName(str), table->rowType());
  ColumnVector columns;
  for (auto& pair : table->columns) {
    auto& tableColumn = pair.second;
    float cardinality = tableColumn->numDistinct;
    Value value(tableColumn->type.get(), cardinality);
    auto columnName = toName(pair.first);
    Declare(Column, column, columnName, nullptr, value);
    schemaTable->columns[columnName] = column;
    columns.push_back(column);
  }
  DistributionType defaultDist;
  defaultDist.locus = locus_.get();
  schemaTable->addIndex(
      toName("pk"), table->numRows, 0, 0, {}, defaultDist, {}, columns);
  schema->addTable(schemaTable);
}

} // namespace facebook::verax
