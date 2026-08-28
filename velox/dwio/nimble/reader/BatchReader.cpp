/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/reader/BatchReader.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "fmt/core.h"
#include "folly/container/F14Map.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/common/OnDemandUnitLoader.h"
#include "velox/dwio/common/UnitLoader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/MetadataGenerated.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/reader/ChunkedStreamDecoder.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/type/Type.h"

namespace facebook::nimble {

namespace {

std::shared_ptr<const Type> loadSchema(const TabletReader& tabletReader) {
  auto section = tabletReader.loadOptionalSection(std::string(kSchemaSection));
  NIMBLE_CHECK(section.has_value(), "Schema not found.");
  return SchemaDeserializer::deserialize(section->content().data());
}

std::map<std::string, std::string> loadMetadata(
    const TabletReader& tabletReader) {
  std::map<std::string, std::string> result;
  auto section =
      tabletReader.loadOptionalSection(std::string(kMetadataSection));

  if (!section.has_value()) {
    return result;
  }

  auto metadata =
      flatbuffers::GetRoot<serialization::Metadata>(section->content().data());
  auto entryCount = metadata->entries()->size();
  for (auto i = 0; i < entryCount; ++i) {
    auto* entry = metadata->entries()->Get(i);
    result.insert({entry->key()->str(), entry->value()->str()});
  }

  return result;
}

Encoding::Options encodingOptions(const TabletReader& tabletReader) {
  Encoding::Options options;
  options.useVarintRowCount =
      tabletReader.properties().compactRowCountEncoding();
  return options;
}

class NimbleUnit : public velox::dwio::common::LoadUnit {
 public:
  NimbleUnit(
      uint32_t stripeId,
      const TabletReader& tabletReader,
      std::shared_ptr<const Type> schema,
      std::shared_ptr<StreamLabels> valueStreamLabels,
      const std::vector<uint32_t>& valueStreamIdentifiers)
      : tabletReader_{tabletReader},
        schema_{std::move(schema)},
        valueStreamLabels_{std::move(valueStreamLabels)},
        stripeId_{stripeId},
        valueStreamIdentifiers_{valueStreamIdentifiers},
        loadStreamIdentifiers_{valueStreamIdentifiers} {
    addDictionaryStreams();
  }

  // Perform the IO (read)
  void load() override;

  // Unload the unit to free memory
  void unload() override;

  // Number of rows in the unit
  uint64_t getNumRows() override {
    return tabletReader_.stripeRowCount(stripeId_);
  }

  // Number of bytes that the IO will read
  uint64_t getIoSize() override;

  std::vector<std::unique_ptr<StreamLoader>> extractStreamLoaders() {
    return std::move(streamLoaders_);
  }

  const folly::F14FastMap<uint32_t, uint32_t>& dictionaryStreamIds() const {
    return dictionaryStreamIds_;
  }

  const StripeLoadMetrics& getMetrics() const {
    return metrics_;
  }

 private:
  // Appends projected dictionary streams.
  void addDictionaryStreams();

  const TabletReader& tabletReader_;
  std::shared_ptr<const Type> schema_;
  std::shared_ptr<StreamLabels> valueStreamLabels_;
  uint32_t stripeId_;
  const std::vector<uint32_t>& valueStreamIdentifiers_;
  // Projected value streams followed by auxiliary stripe dictionary streams.
  std::vector<uint32_t> loadStreamIdentifiers_;
  // Projected value stream IDs mapped to auxiliary dictionary stream IDs.
  folly::F14FastMap<uint32_t, uint32_t> dictionaryStreamIds_;
  // Dictionary stream labels keyed by auxiliary dictionary stream ID.
  folly::F14FastMap<uint32_t, std::string> dictionaryStreamLabels_;

  // Lazy
  std::optional<uint64_t> ioSize_;

  std::optional<StripeIdentifier> stripeIdentifier_;
  // Will be loaded on load() and moved away in extractStreamLoaders()
  std::vector<std::unique_ptr<StreamLoader>> streamLoaders_;
  StripeLoadMetrics metrics_;
};

void NimbleUnit::addDictionaryStreams() {
  NIMBLE_CHECK_EQ(
      loadStreamIdentifiers_.size(),
      valueStreamIdentifiers_.size(),
      "Load streams must initially match projected streams.");
  if (!tabletReader_.hasStripeDictionaries()) {
    return;
  }
  NIMBLE_CHECK(dictionaryStreamIds_.empty());
  dictionaryStreamIds_ =
      tabletReader_.stripeDictionaryStreamIds(valueStreamIdentifiers_);
  if (dictionaryStreamIds_.empty()) {
    return;
  }
  size_t dictionaryLoaderIndex = loadStreamIdentifiers_.size();
  const auto dictionaryStreamCount = dictionaryStreamIds_.size();
  loadStreamIdentifiers_.resize(dictionaryLoaderIndex + dictionaryStreamCount);
  for (const auto& entry : dictionaryStreamIds_) {
    // TODO: Deduplicate equal dictionary stream IDs if value streams are
    // allowed to share a stripe dictionary.
    loadStreamIdentifiers_[dictionaryLoaderIndex++] = entry.second;
    dictionaryStreamLabels_.emplace(
        entry.second,
        fmt::format(
            "shared dictionary for {}",
            valueStreamLabels_->streamLabel(entry.first)));
  }
}

void NimbleUnit::load() {
  velox::CpuWallTiming timing{};
  {
    velox::CpuWallTimer timer{timing};
    if (!stripeIdentifier_.has_value()) {
      stripeIdentifier_ = tabletReader_.stripeIdentifier(stripeId_);
    }
    streamLoaders_ = tabletReader_.load(
        stripeIdentifier_.value(),
        loadStreamIdentifiers_,
        [this](offset_size offset) -> std::string_view {
          const auto dictionaryIt =
              dictionaryStreamLabels_.find(static_cast<uint32_t>(offset));
          if (dictionaryIt != dictionaryStreamLabels_.end()) {
            return dictionaryIt->second;
          }
          return valueStreamLabels_->streamLabel(offset);
        });
  }
  metrics_.cpuUsec = timing.cpuNanos / 1000;
  metrics_.wallTimeUsec = timing.wallNanos / 1000;
}

void NimbleUnit::unload() {
  streamLoaders_.clear();
  stripeIdentifier_.reset();
}

uint64_t NimbleUnit::getIoSize() {
  if (ioSize_.has_value()) {
    return ioSize_.value();
  }
  if (!stripeIdentifier_.has_value()) {
    stripeIdentifier_ = tabletReader_.stripeIdentifier(stripeId_);
  }
  ioSize_ = tabletReader_.totalStreamSize(
      stripeIdentifier_.value(), loadStreamIdentifiers_);
  return ioSize_.value();
}

// Builds `TabletReader::Options` for `BatchReader`'s convenience constructors.
TabletReader::Options tabletReaderOptions(
    velox::memory::MemoryPool* pool,
    std::shared_ptr<const ExternalDictionaryResolver>
        externalDictionaryResolver) {
  TabletReader::Options options;
  options.preloadOptionalSections = {std::string(kSchemaSection)};
  options.ioOptions.emplace(pool)
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
      .setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
  options.externalDictionaryResolver = std::move(externalDictionaryResolver);
  return options;
}

} // namespace

BatchReader::BatchReader(
    velox::ReadFile* file,
    velox::memory::MemoryPool& pool,
    std::shared_ptr<const velox::dwio::common::ColumnSelector> selector,
    const BatchReadParams& params)
    : BatchReader(
          TabletReader::create(
              std::shared_ptr<velox::ReadFile>(file, [](auto*) {}),
              &pool,
              tabletReaderOptions(&pool, params.externalDictionaryResolver)),
          pool,
          std::move(selector),
          params) {}

BatchReader::BatchReader(
    std::shared_ptr<velox::ReadFile> file,
    velox::memory::MemoryPool& pool,
    std::shared_ptr<const velox::dwio::common::ColumnSelector> selector,
    const BatchReadParams& params)
    : BatchReader(
          TabletReader::create(
              std::move(file),
              &pool,
              tabletReaderOptions(&pool, params.externalDictionaryResolver)),
          pool,
          std::move(selector),
          params) {}

BatchReader::BatchReader(
    std::shared_ptr<const TabletReader> tabletReader,
    velox::memory::MemoryPool& pool,
    std::shared_ptr<const velox::dwio::common::ColumnSelector> selector,
    BatchReadParams params)
    : pool_{pool},
      tabletReader_{std::move(tabletReader)},
      parameters_{std::move(params)},
      schema_{loadSchema(*tabletReader_)},
      type_{
          selector ? selector->getSchema()
                   : std::dynamic_pointer_cast<const velox::RowType>(
                         convertToVeloxType(*schema_))},
      barrier_{
          parameters_.decodingExecutor
              ? std::make_unique<velox::dwio::common::ExecutorBarrier>(
                    parameters_.decodingExecutor)
              : nullptr},
      logger_{
          parameters_.metricsLogger ? parameters_.metricsLogger
                                    : std::make_shared<MetricsLogger>()} {
  static_assert(std::is_same_v<velox::vector_size_t, int32_t>);

  if (!selector) {
    selector = std::make_shared<velox::dwio::common::ColumnSelector>(type_);
  }
  auto schemaWithId = selector->getSchemaWithId();
  rootFieldReaderFactory_ = FieldReaderFactory::create(
      parameters_,
      schema_,
      schemaWithId,
      offsets_,
      [selector](auto nodeId) { return selector->shouldReadNode(nodeId); },
      &pool_);

  // We scope down the allowed stripes based on the passed in offset ranges.
  // These ranges represent file splits.
  // File splits contain a file path and a range of bytes (not rows) withing
  // that file. It is possible that multiple file splits map to the same file
  // (but each covers a diffrent range within this file).
  // File splits are guaranteed to not overlap and also to cover the entire file
  // range.
  // We want to guarantee that each row in the file is processed by
  // exactly one split. When a reader is created, we only know about a single
  // split (the current range passed in), and we have no additional context
  // about other splits being processed by other readers. Therefore, we apply
  // the following heuristics to guarantee uniqueness across splits:
  // 1. We transpose the passed in range to match stripe boundaries.
  // 2. We consider a stripe to be part of this range, only if the stripe
  // beginning offset falls inside the range.
  // NOTE: With these heuristics, it is possible that a file split will map to
  // zero rows in a file (for example, if the file split is falls completely
  // inside a single stripe, without covering byte 0). This is perfectly ok, as
  // usually the caller will then fetch another file split to process and other
  // file splits will cover the rest of the file.

  const auto stripeCount = tabletReader_->stripeCount();
  cummulativeStripeRowCount_.reserve(stripeCount);
  firstStripe_ = stripeCount;
  lastStripe_ = 0;
  firstRow_ = 0;
  lastRow_ = 0;
  uint64_t rows = 0;
  for (auto i = 0; i < stripeCount; ++i) {
    const auto stripeOffset = tabletReader_->stripeOffset(i);
    const auto stripeRowCount = tabletReader_->stripeRowCount(i);
    if ((stripeOffset >= parameters_.fileRangeStartOffset) &&
        (stripeOffset < parameters_.fileRangeEndOffset)) {
      if (i < firstStripe_) {
        firstStripe_ = i;
        firstRow_ = rows;
      }
      if (i >= lastStripe_) {
        lastStripe_ = i + 1;
        lastRow_ = rows + stripeRowCount;
      }
    }
    rows += stripeRowCount;
    cummulativeStripeRowCount_.push_back(rows);
  }

  nextStripe_ = firstStripe_;

  if (parameters_.stripeCountCallback) {
    if (firstStripe_ <= lastStripe_) {
      parameters_.stripeCountCallback(lastStripe_ - firstStripe_);
    } else {
      parameters_.stripeCountCallback(0);
    }
  }

  VLOG(1) << "TabletReader handling stripes: " << firstStripe_ << " -> "
          << lastStripe_ << " (rows " << firstRow_ << " -> " << lastRow_
          << "). Total stripes: " << stripeCount
          << ". Total rows: " << tabletReader_->tabletRowCount();

  unitLoader_ = getUnitLoader();
}

void BatchReader::loadStripeIfAny() {
  if (nextStripe_ < lastStripe_) {
    loadNextStripe();
  }
}

BatchReadParams::StreamEncodingFactory BatchReader::createStreamEncodingFactory(
    uint32_t valueStreamId,
    std::unique_ptr<StreamLoader> dictionaryStream) const {
  if (dictionaryStream == nullptr &&
      !tabletReader_->hasFileOrExternalDictionaries()) {
    return parameters_.encodingFactory;
  }

  std::shared_ptr<const SharedDictionaryAlphabet> dictionaryAlphabet;
  if (dictionaryStream == nullptr) {
    dictionaryAlphabet =
        tabletReader_->resolveDictionaryAlphabet(valueStreamId);
    if (dictionaryAlphabet == nullptr) {
      return parameters_.encodingFactory;
    }
  }

  // Match the selective reader path: resolve the per-stream alphabet directly
  // when the first chunk is decoded. Streams with dictionary bindings are
  // written consistently as shared-dictionary streams.
  return [dictionaryStreamOwner =
              std::shared_ptr<const StreamLoader>{std::move(dictionaryStream)},
          _dictionaryAlphabet = std::move(dictionaryAlphabet),
          options = encodingOptions(*tabletReader_)](
             velox::memory::MemoryPool& pool,
             std::string_view data,
             std::function<void*(uint32_t)> stringBufferFactory) mutable {
    if (_dictionaryAlphabet == nullptr) {
      NIMBLE_CHECK_NOT_NULL(
          dictionaryStreamOwner,
          "Shared dictionary requires a stripe dictionary stream.");
      _dictionaryAlphabet = SharedDictionaryAlphabet::create(
          dictionaryStreamOwner->getStream(), dictionaryStreamOwner, &pool);
    }
    options.sharedDictionaryAlphabet = _dictionaryAlphabet;
    return EncodingFactory().create(
        pool, data, std::move(stringBufferFactory), options);
  };
}

void BatchReader::loadNextStripe() {
  if (loadedStripe_.has_value() && loadedStripe_.value() == nextStripe_) {
    // We are not reloading the current stripe, but we expect all
    // decoders/readers to be reset after calling loadNextStripe(), therefore,
    // we need to explicitly reset all decoders and readers.
    rootReader_->reset();

    rowsRemainingInStripe_ = tabletReader_->stripeRowCount(nextStripe_);
    unitLoader_->onSeek(
        getUnitIndex(loadedStripe_.value()), /* rowInStripe */ 0);
    ++nextStripe_;
    return;
  }

  try {
    StripeLoadMetrics metrics;
    velox::CpuWallTiming timing{};
    {
      // Free up any memory used by the previous stripe
      rootReader_ = nullptr;
      decoders_.clear();

      auto& unit = unitLoader_->getLoadedUnit(getUnitIndex(nextStripe_));

      velox::CpuWallTimer timer{timing};
      auto* nimbleUnit = dynamic_cast<NimbleUnit*>(&unit);
      NIMBLE_CHECK_NOT_NULL(nimbleUnit, "Should be a NimbleUnit");
      rowsRemainingInStripe_ = nimbleUnit->getNumRows();
      metrics = nimbleUnit->getMetrics();
      metrics.totalStreamSize = nimbleUnit->getIoSize();

      auto streams = nimbleUnit->extractStreamLoaders();
      const auto& dictionaryStreamIds = nimbleUnit->dictionaryStreamIds();
      NIMBLE_CHECK_GE(streams.size(), offsets_.size());
      folly::F14FastMap<uint32_t, uint32_t> dictionaryLoaderIndices;
      if (!dictionaryStreamIds.empty()) {
        dictionaryLoaderIndices.reserve(dictionaryStreamIds.size());
        uint32_t dictionaryLoaderIndex = static_cast<uint32_t>(offsets_.size());
        for (const auto& entry : dictionaryStreamIds) {
          dictionaryLoaderIndices.emplace(entry.first, dictionaryLoaderIndex++);
        }
      }
      decoders_.reserve(offsets_.size());
      for (uint32_t i = 0; i < offsets_.size(); ++i) {
        auto& stream = streams.at(i);
        if (stream == nullptr) {
          // As this stream is not present in current stripe (might be present
          // in previous one) we set to nullptr, One of the case is where you
          // are projecting more fields in FlatMap than the stripe actually
          // has.
          decoders_[offsets_[i]] = nullptr;
        } else {
          ++metrics.streamCount;
          std::unique_ptr<StreamLoader> dictionaryStream;
          const auto dictionaryIt = dictionaryLoaderIndices.find(offsets_[i]);
          if (dictionaryIt != dictionaryLoaderIndices.end()) {
            // Direct-encoded stripes can omit the dictionary stream even when
            // the value stream has a stripe dictionary binding.
            dictionaryStream = std::move(streams.at(dictionaryIt->second));
          }
          auto streamEncodingFactory = createStreamEncodingFactory(
              offsets_[i], std::move(dictionaryStream));
          auto decoder = std::make_unique<ChunkedStreamDecoder>(
              pool_,
              std::make_unique<InMemoryChunkedStream>(pool_, std::move(stream)),
              std::move(streamEncodingFactory),
              parameters_.optimizeStringBufferHandling,
              *logger_);
          decoder->ensureLoaded();
          decoders_[offsets_[i]] = std::move(decoder);
        }
      }
      loadedStripe_ = nextStripe_++;
      rootReader_ = rootFieldReaderFactory_->createReader(decoders_);
    }
    metrics.stripeIndex = loadedStripe_.value();
    metrics.rowsInStripe = rowsRemainingInStripe_;
    metrics.cpuUsec += timing.cpuNanos / 1000;
    metrics.wallTimeUsec += timing.wallNanos / 1000;
    logger_->logStripeLoad(metrics);
  } catch (const std::exception& e) {
    logger_->logException(LogOperation::StripeLoad, e.what());
    throw;
  } catch (...) {
    logger_->logException(
        LogOperation::StripeLoad,
        folly::to<std::string>(folly::exceptionStr(std::current_exception())));
    throw;
  }
}

uint64_t BatchReader::estimatedRowSize() {
  if (!loadedStripe_.has_value() || rowsRemainingInStripe_ == 0) {
    // We don't load to do the estimation if there isn't any stripe loaded or we
    // are currently at stripe boundary. Instead we return a highly conservative
    // large row size value.
    return kConservativeEstimatedRowSize;
  }
  if (cachedRowSizeEstimationStripeIdx_ != loadedStripe_) {
    auto estimatedRowSize = rootReader_->estimatedRowSize();
    if (!estimatedRowSize.has_value()) {
      cachedRowSizeEstimation_ = kConservativeEstimatedRowSize;
    } else {
      cachedRowSizeEstimation_ = estimatedRowSize.value().second;
    }
    cachedRowSizeEstimationStripeIdx_ = loadedStripe_;
  }
  return cachedRowSizeEstimation_;
}

bool BatchReader::next(uint64_t rowCount, velox::VectorPtr& result) {
  if (rowsRemainingInStripe_ == 0) {
    if (nextStripe_ < lastStripe_) {
      loadNextStripe();
    } else {
      return false;
    }
  }

  uint64_t rowsToRead = std::min(rowsRemainingInStripe_, rowCount);
  std::optional<std::chrono::steady_clock::time_point> startTime;
  if (parameters_.decodingTimeCallback) {
    startTime = std::chrono::steady_clock::now();
  }
  unitLoader_->onRead(
      getUnitIndex(loadedStripe_.value()), getCurrentRowInStripe(), rowsToRead);
  rootReader_->next(rowsToRead, result);
  if (barrier_) {
    // Wait for all reader tasks to complete.
    barrier_->waitAll();
  }
  if (startTime.has_value()) {
    parameters_.decodingTimeCallback(
        std::chrono::steady_clock::now() - startTime.value());
  }

  // Update reader state
  rowsRemainingInStripe_ -= rowsToRead;
  return true;
}

const TabletReader& BatchReader::tabletReader() const {
  return *tabletReader_;
}

const std::shared_ptr<const velox::RowType>& BatchReader::type() const {
  return type_;
}

const std::shared_ptr<const Type>& BatchReader::schema() const {
  return schema_;
}

const std::map<std::string, std::string>& BatchReader::metadata() const {
  if (!metadata_.has_value()) {
    metadata_ = loadMetadata(*tabletReader_);
  }

  return metadata_.value();
}

uint64_t BatchReader::seekToRow(uint64_t rowNumber) {
  if (isEmptyFile()) {
    return 0;
  }

  if (rowNumber < firstRow_) {
    LOG(INFO) << "Trying to seek to row " << rowNumber
              << " which is outside of the allowed range [" << firstRow_ << ", "
              << lastRow_ << ").";

    nextStripe_ = firstStripe_;
    rowsRemainingInStripe_ = 0;
    return firstRow_;
  }

  if (rowNumber >= lastRow_) {
    LOG(INFO) << "Trying to seek to row " << rowNumber
              << " which is outside of the allowed range [" << firstRow_ << ", "
              << lastRow_ << ").";

    nextStripe_ = lastStripe_;
    rowsRemainingInStripe_ = 0;
    return lastRow_;
  }

  auto rowsSkipped = skipStripes(0, rowNumber);
  loadNextStripe();
  skipInCurrentStripe(rowNumber - rowsSkipped);
  return rowNumber;
}

uint64_t BatchReader::skipRows(uint64_t numberOfRowsToSkip) {
  if (isEmptyFile() || numberOfRowsToSkip == 0) {
    LOG(INFO) << "Nothing to skip!";
    return 0;
  }

  // When we skipped or exhausted the whole file we can return 0
  if (rowsRemainingInStripe_ == 0 && nextStripe_ == lastStripe_) {
    LOG(INFO) << "Current index is beyond EOF. Nothing to skip.";
    return 0;
  }

  // Skips remaining rows in stripe
  if (rowsRemainingInStripe_ >= numberOfRowsToSkip) {
    skipInCurrentStripe(numberOfRowsToSkip);
    return numberOfRowsToSkip;
  }

  uint64_t rowsSkipped = rowsRemainingInStripe_;
  auto rowsToSkip = numberOfRowsToSkip;
  // Skip the leftover rows from currently loaded stripe
  rowsToSkip -= rowsRemainingInStripe_;
  rowsSkipped += skipStripes(nextStripe_, rowsToSkip);
  if (nextStripe_ >= lastStripe_) {
    LOG(INFO) << "Skipped to last allowed row in the file.";
    return rowsSkipped;
  }

  loadNextStripe();
  skipInCurrentStripe(numberOfRowsToSkip - rowsSkipped);
  return numberOfRowsToSkip;
}

uint64_t BatchReader::skipStripes(
    uint32_t startStripeIndex,
    uint64_t rowsToSkip) {
  NIMBLE_DCHECK(
      startStripeIndex <= lastStripe_,
      fmt::format("Invalid stripe {}.", startStripeIndex));

  uint64_t totalRowsToSkip = rowsToSkip;
  while (startStripeIndex < lastStripe_ &&
         rowsToSkip >= tabletReader_->stripeRowCount(startStripeIndex)) {
    rowsToSkip -= tabletReader_->stripeRowCount(startStripeIndex);
    ++startStripeIndex;
  }

  nextStripe_ = startStripeIndex;
  rowsRemainingInStripe_ = nextStripe_ >= lastStripe_
      ? 0
      : tabletReader_->stripeRowCount(nextStripe_);

  return totalRowsToSkip - rowsToSkip;
}

void BatchReader::skipInCurrentStripe(uint64_t rowsToSkip) {
  NIMBLE_DCHECK(
      rowsToSkip <= rowsRemainingInStripe_,
      "Not Enough rows to skip in stripe!");
  rowsRemainingInStripe_ -= rowsToSkip;
  unitLoader_->onSeek(
      getUnitIndex(loadedStripe_.value()), getCurrentRowInStripe());
  rootReader_->skip(rowsToSkip);
}

BatchReader::~BatchReader() = default;

std::unique_ptr<velox::dwio::common::UnitLoader> BatchReader::getUnitLoader() {
  if (lastStripe_ <= firstStripe_) {
    return nullptr;
  }

  std::vector<std::unique_ptr<velox::dwio::common::LoadUnit>> units;
  units.reserve(lastStripe_ - firstStripe_);
  const auto valueStreamLabels = std::make_shared<StreamLabels>(schema_);
  for (uint32_t stripe = firstStripe_; stripe < lastStripe_; ++stripe) {
    units.push_back(
        std::make_unique<NimbleUnit>(
            stripe, *tabletReader_, schema_, valueStreamLabels, offsets_));
  }

  if (parameters_.unitLoaderFactory) {
    return parameters_.unitLoaderFactory->create(std::move(units), 0);
  }
  velox::dwio::common::OnDemandUnitLoaderFactory factory(
      parameters_.blockedOnIoCallback);
  return factory.create(std::move(units), 0);
}

uint32_t BatchReader::getUnitIndex(uint32_t stripeIndex) const {
  return stripeIndex - firstStripe_;
}

uint32_t BatchReader::getCurrentRowInStripe() const {
  return tabletReader_->stripeRowCount(loadedStripe_.value()) -
      static_cast<uint32_t>(rowsRemainingInStripe_);
}

uint64_t BatchReader::getRowNumber() {
  if (!loadedStripe_.has_value()) {
    return firstRow_;
  }

  if (rowsRemainingInStripe_ > 0) {
    return cummulativeStripeRowCount_[loadedStripe_.value()] -
        rowsRemainingInStripe_;
  }

  // When rowsRemainingInStripe_ is 0, it either means that we finished reading
  // a stripe or that we attempted to seek outside of an acceptable row range.
  // This means we cannot rely on the value of rowsRemainingInStripe_ to be
  // correct. At this point, we know that nextStripe_ is greater than 0 because:
  // 1. If we finished reading a stripe, then nextStripe_ is guaranteed to point
  // to loadedStripe_ + 1.
  // 2. Seeking to before firstRow_ (this will make rowsRemainingInStripe_ to be
  // inaccurate) can only happen if firstRow_ is greater than 0 which also means
  // that nextStripe_ is greater than 0.
  if (nextStripe_ >= lastStripe_) {
    return lastRow_;
  }

  DWIO_ENSURE_GT(nextStripe_, 0);
  return cummulativeStripeRowCount_[nextStripe_ - 1];
}
} // namespace facebook::nimble
