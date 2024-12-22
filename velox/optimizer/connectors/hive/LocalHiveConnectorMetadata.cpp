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

#include "velox/optimizer/connectors/hive/LocalHiveConnectorMetadata.h"
#include "velox/common/base/Fs.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/dwrf/common/Statistics.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive {

std::vector<std::shared_ptr<const PartitionHandle>>
LocalHiveSplitManager::listPartitions(
    const ConnectorTableHandlePtr& tableHandle) {
  // All tables are unpartitioned.
  std::unordered_map<std::string, std::optional<std::string>> empty;
  return {std::make_shared<HivePartitionHandle>(empty, std::nullopt)};
}

std::shared_ptr<SplitSource> LocalHiveSplitManager::getSplitSource(
    const ConnectorTableHandlePtr& tableHandle,
    std::vector<std::shared_ptr<const PartitionHandle>> partitions) {
  // Since there are only unpartitioned tables now, always makes a SplitSource
  // that goes over all the files in the handle's layout.
  auto tableName = tableHandle->tableName();
  auto* metadata = getConnector(tableHandle->connectorId())->metadata();
  auto* table = metadata->findTable(tableName);
  VELOX_CHECK_NOT_NULL(
      table, "Could not find {} in its ConnectorMetadata", tableName);
  auto* layout = dynamic_cast<const LocalHiveTableLayout*>(table->layouts()[0]);
  VELOX_CHECK_NOT_NULL(layout);
  auto files = layout->files();
  return std::make_shared<LocalHiveSplitSource>(
      files, 2, layout->fileFormat(), layout->connector()->connectorId());
}

std::vector<SplitSource::SplitAndGroup> LocalHiveSplitSource::getSplits(
    uint64_t targetBytes) {
  std::vector<SplitAndGroup> result;
  uint64_t bytes = 0;
  for (;;) {
    if (currentFile_ >= static_cast<int32_t>(files_.size())) {
      result.push_back(SplitSource::SplitAndGroup{nullptr, 0});
      return result;
    }

    if (currentSplit_ >= fileSplits_.size()) {
      fileSplits_.clear();
      ++currentFile_;
      if (currentFile_ >= files_.size()) {
        result.push_back(SplitSource::SplitAndGroup{nullptr, 0});
        return result;
      }

      currentSplit_ = 0;
      auto filePath = files_[currentFile_];
      const auto fileSize = fs::file_size(filePath);
      // Take the upper bound.
      const int splitSize = std::ceil((fileSize) / splitsPerFile_);
      for (int i = 0; i < splitsPerFile_; ++i) {
        fileSplits_.push_back(
            connector::hive::HiveConnectorSplitBuilder(filePath)
                .connectorId(connectorId_)
                .fileFormat(format_)
                .start(i * splitSize)
                .length(splitSize)
                .build());
      }
    }
    result.push_back(SplitAndGroup{std::move(fileSplits_[currentSplit_++]), 0});
    bytes +=
        reinterpret_cast<const HiveConnectorSplit*>(result.back().split.get())
            ->length;
    if (bytes > targetBytes) {
      return result;
    }
  }
}

LocalHiveConnectorMetadata::LocalHiveConnectorMetadata(
    HiveConnector* hiveConnector)
    : HiveConnectorMetadata(hiveConnector),
      hiveConfig_(
          std::make_shared<HiveConfig>(hiveConnector_->connectorConfig())),
      splitManager_(this) {}

void LocalHiveConnectorMetadata::initialize() {
  auto formatName = hiveConfig_->localDefaultFileFormat();
  auto path = hiveConfig_->localDataPath();
  format_ = formatName == "dwrf" ? dwio::common::FileFormat::DWRF
      : formatName == "parquet"  ? dwio::common::FileFormat::PARQUET
                                 : dwio::common::FileFormat::UNKNOWN;
  makeQueryCtx();
  makeConnectorQueryCtx();
  readTables(path);
}

void LocalHiveConnectorMetadata::makeQueryCtx() {
  std::unordered_map<std::string, std::string> config;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs;
  connectorConfigs[hiveConnector_->connectorId()] =
      std::const_pointer_cast<config::ConfigBase>(hiveConfig_->config());

  queryCtx_ = core::QueryCtx::create(
      hiveConnector_->executor(),
      core::QueryConfig(config),
      std::move(connectorConfigs),
      cache::AsyncDataCache::getInstance(),
      rootPool_->shared_from_this(),
      nullptr,
      "local_hive_metadata");
}

void LocalHiveConnectorMetadata::makeConnectorQueryCtx() {
  common::SpillConfig spillConfig;
  common::PrefixSortConfig prefixSortConfig;
  schemaPool_ = queryCtx_->pool()->addLeafChild("schemaReader");
  connectorQueryCtx_ = std::make_shared<connector::ConnectorQueryCtx>(
      schemaPool_.get(),
      queryCtx_->pool(),
      queryCtx_->connectorSessionProperties(hiveConnector_->connectorId()),
      &spillConfig,
      prefixSortConfig,
      std::make_unique<exec::SimpleExpressionEvaluator>(
          queryCtx_.get(), schemaPool_.get()),
      queryCtx_->cache(),
      "scan_for_schema",
      "schema",
      "N/a",
      0,
      queryCtx_->queryConfig().sessionTimezone());
}

void LocalHiveConnectorMetadata::readTables(const std::string& path) {
  for (auto const& dirEntry : fs::directory_iterator{path}) {
    if (!dirEntry.is_directory() ||
        dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    loadTable(dirEntry.path().filename(), dirEntry.path());
  }
}

// Feeds the values in 'vector' into 'builder'.
template <typename Builder, typename T>
void addStats(
    velox::dwrf::StatisticsBuilder* builder,
    const BaseVector& vector) {
  auto* typedVector = vector.asUnchecked<SimpleVector<T>>();
  for (auto i = 0; i < typedVector->size(); ++i) {
    if (!typedVector->isNullAt(i)) {
      reinterpret_cast<Builder*>(builder)->addValues(typedVector->valueAt(i));
    }
  }
}
std::unique_ptr<ColumnStatistics> toRunnerStats(
    std::unique_ptr<dwio::common::ColumnStatistics> dwioStats) {
  auto result = std::make_unique<ColumnStatistics>();
  result->numDistinct = dwioStats->numDistinct();

  return result;
}

std::pair<int64_t, int64_t> LocalHiveTableLayout::sample(
    const connector::ConnectorTableHandlePtr& handle,
    float pct,
    std::vector<core::TypedExprPtr> extraFilters,
    const std::vector<common::Subfield>& fields,
    HashStringAllocator* allocator,
    std::vector<ColumnStatistics>* statistics) const {
  std::vector<std::unique_ptr<velox::dwrf::StatisticsBuilder>> builders;
  VELOX_CHECK(extraFilters.empty());
  auto result = sample(handle, pct, fields, allocator, &builders);
  if (!statistics) {
    return result;
  }
  statistics->resize(builders.size());
  for (auto i = 0; i < builders.size(); ++i) {
    ColumnStatistics runnerStats;
    if (builders[i]) {
      dwrf::proto::ColumnStatistics proto;
      builders[i]->toProto(proto);
      dwrf::StatsContext context("", dwrf::WriterVersion::ORIGINAL);
      auto wrapper = dwrf::ColumnStatisticsWrapper(&proto);
      auto stats = buildColumnStatisticsFromProto(wrapper, context);
      runnerStats = *toRunnerStats(std::move(stats));
    }

    (*statistics)[i] = std::move(runnerStats);
  }
  return result;
}

std::pair<int64_t, int64_t> LocalHiveTableLayout::sample(
    const connector::ConnectorTableHandlePtr& tableHandle,
    float pct,
    const std::vector<common::Subfield>& fields,
    HashStringAllocator* /*allocator*/,
    std::vector<std::unique_ptr<velox::dwrf::StatisticsBuilder>>* statsBuilders)
    const {
  dwrf::StatisticsBuilderOptions options(
      /*stringLengthLimit=*/100, /*initialSize=*/0);
  std::vector<std::unique_ptr<dwrf::StatisticsBuilder>> builders;

  std::unordered_map<
      std::string,
      std::shared_ptr<velox::connector::ColumnHandle>>
      columnHandles;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto& field : fields) {
    auto& path = field.path();
    auto column =
        dynamic_cast<const common::Subfield::NestedField*>(path[0].get())
            ->name();
    const auto idx = rowType()->getChildIdx(column);
    names.push_back(rowType()->nameOf(idx));
    types.push_back(rowType()->childAt(idx));
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

  const auto outputType = ROW(std::move(names), std::move(types));
  int64_t passingRows = 0;
  int64_t scannedRows = 0;
  for (auto& file : files_) {
    // TODO: make createDataSource take a ConnectorTableHandlePtr instead of a
    // shared_ptr to mutable handle.
    auto handleCopy =
        std::const_pointer_cast<connector::ConnectorTableHandle>(tableHandle);
    auto connectorQueryCtx =
        reinterpret_cast<LocalHiveConnectorMetadata*>(connector()->metadata())
            ->connectorQueryCtx();
    auto dataSource = connector()->createDataSource(
        outputType, handleCopy, columnHandles, connectorQueryCtx.get());

    auto split = connector::hive::HiveConnectorSplitBuilder(file)
                     .fileFormat(fileFormat_)
                     .connectorId(connector()->connectorId())
                     .build();
    dataSource->addSplit(split);
    constexpr int32_t kBatchSize = 1000;
    for (;;) {
      ContinueFuture ignore{ContinueFuture::makeEmpty()};

      auto data = dataSource->next(kBatchSize, ignore).value();
      if (data == nullptr) {
        scannedRows += dataSource->getCompletedRows();
        break;
      }
      passingRows += data->size();
      for (auto column = 0; column < builders.size(); ++column) {
        if (!builders[column]) {
          continue;
        }
        auto* builder = builders[column].get();
        auto loadChild = [](RowVectorPtr data, int32_t column) {
          data->childAt(column) =
              BaseVector::loadedVectorShared(data->childAt(column));
        };
        switch (rowType()->childAt(column)->kind()) {
          case TypeKind::SMALLINT:
            loadChild(data, column);
            addStats<dwrf::IntegerStatisticsBuilder, short>(
                builder, *data->childAt(column));
            break;
          case TypeKind::INTEGER:
            loadChild(data, column);
            addStats<dwrf::IntegerStatisticsBuilder, int32_t>(
                builder, *data->childAt(column));
            break;
          case TypeKind::BIGINT:
            loadChild(data, column);
            addStats<dwrf::IntegerStatisticsBuilder, int64_t>(
                builder, *data->childAt(column));
            break;
          case TypeKind::REAL:
            loadChild(data, column);
            addStats<dwrf::DoubleStatisticsBuilder, float>(
                builder, *data->childAt(column));
            break;
          case TypeKind::DOUBLE:
            loadChild(data, column);
            addStats<dwrf::DoubleStatisticsBuilder, double>(
                builder, *data->childAt(column));
            break;
          case TypeKind::VARCHAR:
            loadChild(data, column);
            addStats<dwrf::StringStatisticsBuilder, StringView>(
                builder, *data->childAt(column));
            break;

          default:
            break;
        }
      }
      if (scannedRows + dataSource->getCompletedRows() >
          table()->numRows() * (pct / 100)) {
        break;
      }
    }
  }
  if (statsBuilders) {
    *statsBuilders = std::move(builders);
  }
  return std::pair(scannedRows, passingRows);
}

void LocalTable::makeDefaultLayout(
    std::vector<std::string> files,
    LocalHiveConnectorMetadata& metadata) {
  std::vector<const Column*> columns;
  for (auto i = 0; i < type_->size(); ++i) {
    auto name = type_->nameOf(i);
    columns.push_back(columns_[name].get());
  }
  auto* connector = metadata.hiveConnector();
  auto format = metadata.fileFormat();
  std::vector<const Column*> empty;
  auto layout = std::make_unique<LocalHiveTableLayout>(
      name_,
      this,
      connector,
      std::move(columns),
      empty,
      empty,
      std::vector<SortOrder>{},
      empty,
      empty,
      format,
      std::nullopt);
  layout->setFiles(std::move(files));
  exportedLayouts_.push_back(layout.get());
  layouts_.push_back(std::move(layout));
}

void LocalHiveConnectorMetadata::loadTable(
    const std::string& tableName,
    const fs::path& tablePath) {
  // open each file in the directory and check their type and add up the row
  // counts.
  RowTypePtr tableType;
  LocalTable* table = nullptr;
  std::vector<std::string> files;

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
      table = reinterpret_cast<LocalTable*>(it->second.get());
    } else {
      tables_[tableName] = std::make_unique<LocalTable>(tableName, format_);
      table = tables_[tableName].get();
    }
    dwio::common::ReaderOptions readerOptions{schemaPool_.get()};
    readerOptions.setFileFormat(format_);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(dirEntry.path().string()),
        readerOptions.memoryPool());
    std::unique_ptr<dwio::common::Reader> reader =
        dwio::common::getReaderFactory(readerOptions.fileFormat())
            ->createReader(std::move(input), readerOptions);
    const auto fileType = reader->rowType();
    if (!tableType) {
      tableType = fileType;
    } else if (fileType->size() > tableType->size()) {
      // The larger type is the later since there is only addition of columns.
      // TODO: Check the column types are compatible where they overlap.
      tableType = fileType;
    }
    const auto rows = reader->numberOfRows();

    if (rows.has_value()) {
      table->numRows_ += rows.value();
    }
    for (auto i = 0; i < fileType->size(); ++i) {
      auto name = fileType->nameOf(i);
      Column* column;
      auto columnIt = table->columns().find(name);
      if (columnIt != table->columns().end()) {
        column = columnIt->second.get();
      } else {
        table->columns()[name] =
            std::make_unique<Column>(name, fileType->childAt(i));
        column = table->columns()[name].get();
      }
      // Initialize the stats from the first file.
      if (column->stats() == nullptr) {
        auto readerStats = reader->columnStatistics(i);
        if (readerStats) {
          auto numValues = readerStats->getNumberOfValues();
          column->setStats(toRunnerStats(std::move(readerStats)));
          if (rows.has_value() && rows.value() > 0 && numValues.has_value()) {
            column->mutableStats()->nullPct =
                100 * (rows.value() - numValues.value()) / rows.value();
          }
        }
      }
    }
    files.push_back(dirEntry.path());
  }
  VELOX_CHECK_NOT_NULL(table, "Table directory {} is empty", tablePath);

  table->setType(tableType);
  table->makeDefaultLayout(std::move(files), *this);
  table->sampleNumDistincts(2, schemaPool_.get());
}

void LocalTable::sampleNumDistincts(float samplePct, memory::MemoryPool* pool) {
  std::vector<common::Subfield> fields;
  for (auto i = 0; i < type_->size(); ++i) {
    fields.push_back(common::Subfield(type_->nameOf(i)));
  }

  // Sample the table. Adjust distinct values according to the samples.
  auto allocator = std::make_unique<HashStringAllocator>(pool);
  auto* layout = layouts_[0].get();
  std::vector<connector::ColumnHandlePtr> columns;
  for (auto i = 0; i < type_->size(); ++i) {
    columns.push_back(layout->connector()->metadata()->createColumnHandle(
        *layout, type_->nameOf(i)));
  }
  auto* metadata = dynamic_cast<const LocalHiveConnectorMetadata*>(
      layout->connector()->metadata());
  auto& evaluator = *metadata->connectorQueryCtx()->expressionEvaluator();
  std::vector<core::TypedExprPtr> ignore;
  auto handle = layout->connector()->metadata()->createTableHandle(
      *layout, columns, evaluator, {}, ignore);
  std::vector<std::unique_ptr<dwrf::StatisticsBuilder>> statsBuilders;
  auto* localLayout = dynamic_cast<LocalHiveTableLayout*>(layout);
  VELOX_CHECK_NOT_NULL(localLayout, "Expecting a local hive layout");
  auto [sampled, passed] = localLayout->sample(
      handle, samplePct, fields, allocator.get(), &statsBuilders);
  numSampledRows_ = sampled;
  for (auto i = 0; i < statsBuilders.size(); ++i) {
    if (statsBuilders[i]) {
      // TODO: Use HLL estimate of distinct values here after this is added to
      // the stats builder. Now assume that all rows have a different value.
      // Later refine this by observed min-max range.
      int64_t approxNumDistinct = numRows_;
      // For tiny tables the sample is 100% and the approxNumDistinct is
      // accurate. For partial samples, the distinct estimate is left to be the
      // distinct estimate of the sample if there are few distincts. This is an
      // enumeration where values in unsampled rows are likely the same. If
      // there are many distincts, we multiply by 1/sample rate assuming that
      // unsampled rows will mostly have new values.

      if (numSampledRows_ < numRows_) {
        if (approxNumDistinct > sampled / 50) {
          float numDups =
              numSampledRows_ / static_cast<float>(approxNumDistinct);
          approxNumDistinct = std::min<float>(numRows_, numRows_ / numDups);

          // If the type is an integer type, num distincts cannot be larger than
          // max - min.
          if (auto* ints = dynamic_cast<dwrf::IntegerStatisticsBuilder*>(
                  statsBuilders[i].get())) {
            auto min = ints->getMinimum();
            auto max = ints->getMaximum();
            if (min.has_value() && max.has_value()) {
              auto range = max.value() - min.value();
              approxNumDistinct = std::min<float>(approxNumDistinct, range);
            }
          }
        }

        const_cast<Column*>(findColumn(type_->nameOf(i)))
            ->mutableStats()
            ->numDistinct = approxNumDistinct;
      }
    }
  }
}

const std::unordered_map<std::string, const Column*>& LocalTable::columnMap()
    const {
  std::lock_guard<std::mutex> l(mutex_);
  if (columns_.empty()) {
    return exportedColumns_;
  }
  for (auto& pair : columns_) {
    exportedColumns_[pair.first] = pair.second.get();
  }
  return exportedColumns_;
}

const Table* LocalHiveConnectorMetadata::findTable(const std::string& name) {
  auto it = tables_.find(name);
  if (it == tables_.end()) {
    return nullptr;
  }
  return it->second.get();
}

namespace {
class LocalHiveConnectorMetadataFactory : public HiveConnectorMetadataFactory {
 public:
  std::shared_ptr<ConnectorMetadata> create(HiveConnector* connector) override {
    auto hiveConfig =
        std::make_shared<HiveConfig>(connector->connectorConfig());
    auto path = hiveConfig->localDataPath();
    if (path.empty()) {
      return nullptr;
    }
    return std::make_shared<LocalHiveConnectorMetadata>(connector);
  }
  void initialize(ConnectorMetadata* metadata) override {
    dynamic_cast<LocalHiveConnectorMetadata*>(metadata)->initialize();
  }
};

bool dummy = registerHiveConnectorMetadataFactory(
    std::make_unique<LocalHiveConnectorMetadataFactory>());
} // namespace

} // namespace facebook::velox::connector::hive
