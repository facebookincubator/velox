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

#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/common/exception/Exception.h"

namespace facebook::velox::dwrf {

using dwio::common::ColumnSelector;
using dwio::common::FileFormat;
using dwio::common::InputStream;
using dwio::common::ReaderOptions;
using dwio::common::RowReaderOptions;

DwrfRowReader::DwrfRowReader(
    const std::shared_ptr<ReaderBase>& reader,
    const RowReaderOptions& opts)
    : StripeReaderBase(reader),
      options_(opts),
      columnSelector_{std::make_shared<ColumnSelector>(
          ColumnSelector::apply(opts.getSelector(), reader->getSchema()))} {
  auto& footer = getReader().getFooter();
  uint32_t numberOfStripes = footer.stripesSize();
  currentStripe = numberOfStripes;
  lastStripe = 0;
  currentRowInStripe = 0;
  newStripeLoaded = false;
  rowsInCurrentStripe = 0;
  uint64_t rowTotal = 0;

  firstRowOfStripe.reserve(numberOfStripes);
  for (uint32_t i = 0; i < numberOfStripes; ++i) {
    firstRowOfStripe.push_back(rowTotal);
    auto stripeInfo = footer.stripes(i);
    rowTotal += stripeInfo.numberOfRows();
    if ((stripeInfo.offset() >= opts.getOffset()) &&
        (stripeInfo.offset() < opts.getLimit())) {
      if (i < currentStripe) {
        currentStripe = i;
      }
      if (i >= lastStripe) {
        lastStripe = i + 1;
      }
    }
  }
  firstStripe = currentStripe;

  if (currentStripe == 0) {
    previousRow = std::numeric_limits<uint64_t>::max();
  } else if (currentStripe == numberOfStripes) {
    previousRow = footer.numberOfRows();
  } else {
    previousRow = firstRowOfStripe[firstStripe] - 1;
  }

  // Validate the requested type is compatible with what's in the file
  std::function<std::string()> createExceptionContext = [&]() {
    std::string exceptionMessageContext = fmt::format(
        "The schema loaded in the reader does not match the schema in the file footer."
        "Input Name: {},\n"
        "File Footer Schema (without partition columns): {},\n"
        "Input Table Schema (with partition columns): {}\n",
        getReader().getBufferedInput().getName(),
        getReader().getSchema()->toString(),
        getType()->toString());
    return exceptionMessageContext;
  };

  dwio::common::typeutils::checkTypeCompatibility(
      *getReader().getSchema(), *columnSelector_, createExceptionContext);
}

uint64_t DwrfRowReader::seekToRow(uint64_t rowNumber) {
  // Empty file
  if (isEmptyFile()) {
    return 0;
  }

  LOG(INFO) << "rowNumber: " << rowNumber << " currentStripe: " << currentStripe
            << " firstStripe: " << firstStripe << " lastStripe: " << lastStripe;

  // If we are reading only a portion of the file
  // (bounded by firstStripe and lastStripe),
  // seeking before or after the portion of interest should return no data.
  // Implement this by setting previousRow to the number of rows in the file.

  // seeking past lastStripe
  auto& footer = getReader().getFooter();
  uint32_t num_stripes = footer.stripesSize();
  if ((lastStripe == num_stripes && rowNumber >= footer.numberOfRows()) ||
      (lastStripe < num_stripes && rowNumber >= firstRowOfStripe[lastStripe])) {
    LOG(INFO) << "Trying to seek past lastStripe, total rows: "
              << footer.numberOfRows() << " num_stripes: " << num_stripes;

    currentStripe = num_stripes;

    previousRow = footer.numberOfRows();
    return previousRow;
  }

  uint32_t seekToStripe = 0;
  while (seekToStripe + 1 < lastStripe &&
         firstRowOfStripe[seekToStripe + 1] <= rowNumber) {
    seekToStripe++;
  }

  // seeking before the first stripe
  if (seekToStripe < firstStripe) {
    currentStripe = num_stripes;
    previousRow = footer.numberOfRows();
    return previousRow;
  }

  currentStripe = seekToStripe;
  currentRowInStripe = rowNumber - firstRowOfStripe[currentStripe];
  previousRow = rowNumber;
  newStripeLoaded = false;
  startNextStripe();

  if (selectiveColumnReader_) {
    selectiveColumnReader_->skip(currentRowInStripe);
  } else {
    columnReader_->skip(currentRowInStripe);
  }

  return previousRow;
}

uint64_t DwrfRowReader::skipRows(uint64_t numberOfRowsToSkip) {
  if (isEmptyFile()) {
    LOG(INFO) << "Empty file, nothing to skip";
    return 0;
  }

  // When no rows to skip - just return 0
  if (numberOfRowsToSkip == 0) {
    LOG(INFO) << "No skipping is needed";
    return 0;
  }

  // when we skipped or exhausted the whole file we can return 0
  auto& footer = getReader().getFooter();
  if (previousRow == footer.numberOfRows()) {
    LOG(INFO) << "previousRow is beyond EOF, nothing to skip";
    return 0;
  }

  if (previousRow == std::numeric_limits<uint64_t>::max()) {
    LOG(INFO) << "Start of the file, skipping: " << numberOfRowsToSkip;
    seekToRow(numberOfRowsToSkip);
    if (previousRow == footer.numberOfRows()) {
      LOG(INFO) << "Reached end of the file, returning: " << previousRow;
      return previousRow;
    } else {
      LOG(INFO) << "previousRow after skipping: " << previousRow;
      return previousRow;
    }
  }

  LOG(INFO) << "Previous row: " << previousRow
            << " Skipping: " << numberOfRowsToSkip;

  uint64_t initialRow = previousRow;
  seekToRow(previousRow + numberOfRowsToSkip);

  LOG(INFO) << "After skipping: " << previousRow
            << " InitialRow: " << initialRow;

  if (previousRow == footer.numberOfRows()) {
    LOG(INFO) << "When seeking past lastStripe";
    return previousRow - initialRow - 1;
  }
  return previousRow - initialRow;
}

void DwrfRowReader::checkSkipStrides(
    const StatsContext& context,
    uint64_t strideSize) {
  if (!selectiveColumnReader_) {
    return;
  }
  if (currentRowInStripe % strideSize != 0) {
    return;
  }

  DWIO_ENSURE(
      selectiveColumnReader_ != nullptr, "selectiveColumnReader_ is null.");

  if (currentRowInStripe == 0 || recomputeStridesToSkip_) {
    DwrfData::FilterRowGroupsResult res;
    selectiveColumnReader_->filterRowGroups(strideSize, context, res);
    if (auto& metadataFilter = options_.getMetadataFilter()) {
      metadataFilter->eval(res.metadataFilterResults, res.filterResult);
    }
    stridesToSkip_ = res.filterResult.data();
    stridesToSkipSize_ = res.totalCount;
    stripeStridesToSkip_[currentStripe] = std::move(res.filterResult);
    recomputeStridesToSkip_ = false;
  }

  bool foundStridesToSkip = false;
  auto currentStride = currentRowInStripe / strideSize;
  while (currentStride < stridesToSkipSize_ &&
         bits::isBitSet(stridesToSkip_, currentStride)) {
    foundStridesToSkip = true;
    currentRowInStripe =
        std::min(currentRowInStripe + strideSize, rowsInCurrentStripe);
    currentStride++;
    skippedStrides_++;
  }
  if (foundStridesToSkip && currentRowInStripe < rowsInCurrentStripe) {
    selectiveColumnReader_->seekToRowGroup(currentStride);
  }
}

uint64_t DwrfRowReader::next(uint64_t size, VectorPtr& result) {
  DWIO_ENSURE_GT(size, 0);
  auto& footer = getReader().getFooter();
  StatsContext context(
      getReader().getWriterName(), getReader().getWriterVersion());

  for (;;) {
    if (currentStripe >= lastStripe) {
      if (lastStripe > 0) {
        previousRow = firstRowOfStripe[lastStripe - 1] +
            footer.stripes(lastStripe - 1).numberOfRows();
      } else {
        previousRow = 0;
      }
      return 0;
    }

    if (currentRowInStripe == 0) {
      startNextStripe();
    }

    auto strideSize = footer.rowIndexStride();
    if (LIKELY(strideSize > 0) && selectiveColumnReader_) {
      checkSkipStrides(context, strideSize);
    }

    uint64_t rowsToRead = std::min(
        static_cast<uint64_t>(size), rowsInCurrentStripe - currentRowInStripe);

    if (rowsToRead > 0) {
      // don't allow read to cross stride
      if (LIKELY(strideSize > 0)) {
        rowsToRead =
            std::min(rowsToRead, strideSize - currentRowInStripe % strideSize);
      }

      // Record strideIndex for use by the columnReader_ which may delay actual
      // reading of the data.
      setStrideIndex(strideSize > 0 ? currentRowInStripe / strideSize : 0);

      if (selectiveColumnReader_) {
        selectiveColumnReader_->next(rowsToRead, result);
      } else {
        columnReader_->next(rowsToRead, result);
      }
    }

    // update row number
    previousRow = firstRowOfStripe[currentStripe] + currentRowInStripe;
    currentRowInStripe += rowsToRead;
    if (currentRowInStripe >= rowsInCurrentStripe) {
      currentStripe += 1;
      currentRowInStripe = 0;
      newStripeLoaded = false;
    }

    if (rowsToRead > 0) {
      return rowsToRead;
    }
  }
}

void DwrfRowReader::resetFilterCaches() {
  if (selectiveColumnReader_) {
    selectiveColumnReader_->resetFilterCaches();
    recomputeStridesToSkip_ = true;
  }

  // For columnReader_, this is no-op.
}

void DwrfRowReader::startNextStripe() {
  if (newStripeLoaded || currentStripe >= lastStripe) {
    return;
  }

  bool preload = options_.getPreloadStripe();
  auto currentStripeInfo = loadStripe(currentStripe, preload);
  rowsInCurrentStripe = currentStripeInfo.numberOfRows();

  StripeStreamsImpl stripeStreams(
      *this,
      getColumnSelector(),
      options_,
      currentStripeInfo.offset(),
      *this,
      currentStripe);

  // Create column reader
  columnReader_.reset();
  selectiveColumnReader_.reset();
  auto scanSpec = options_.getScanSpec().get();
  auto requestedType = getColumnSelector().getSchemaWithId();
  auto dataType = getReader().getSchemaWithId();
  auto flatMapContext = FlatMapContext::nonFlatMapContext();

  if (scanSpec) {
    selectiveColumnReader_ = SelectiveDwrfReader::build(
        requestedType, dataType, stripeStreams, scanSpec, flatMapContext);
    selectiveColumnReader_->setIsTopLevel();
  } else {
    columnReader_ = ColumnReader::build(
        requestedType, dataType, stripeStreams, flatMapContext);
  }
  DWIO_ENSURE(
      (columnReader_ != nullptr) != (selectiveColumnReader_ != nullptr),
      "ColumnReader was not created");

  // load data plan according to its updated selector
  // during column reader construction
  // if planReads is off which means stripe data loaded as whole
  if (!preload) {
    VLOG(1) << "[DWRF] Load read plan for stripe " << currentStripe;
    stripeStreams.loadReadPlan();
  }

  stripeDictionaryCache_ = stripeStreams.getStripeDictionaryCache();
  newStripeLoaded = true;
}

size_t DwrfRowReader::estimatedReaderMemory() const {
  return 2 * DwrfReader::getMemoryUse(getReader(), -1, *columnSelector_);
}

std::optional<size_t> DwrfRowReader::estimatedRowSizeHelper(
    const FooterWrapper& footer,
    const dwio::common::Statistics& stats,
    uint32_t nodeId) const {
  DWIO_ENSURE_LT(nodeId, footer.typesSize(), "Types missing in footer");

  const auto& s = stats.getColumnStatistics(nodeId);
  const auto& t = footer.types(nodeId);
  if (!s.getNumberOfValues()) {
    return std::nullopt;
  }
  auto valueCount = s.getNumberOfValues().value();
  if (valueCount < 1) {
    return 0;
  }
  switch (t.kind()) {
    case TypeKind::BOOLEAN: {
      return valueCount * sizeof(uint8_t);
    }
    case TypeKind::TINYINT: {
      return valueCount * sizeof(uint8_t);
    }
    case TypeKind::SMALLINT: {
      return valueCount * sizeof(uint16_t);
    }
    case TypeKind::INTEGER: {
      return valueCount * sizeof(uint32_t);
    }
    case TypeKind::BIGINT: {
      return valueCount * sizeof(uint64_t);
    }
    case TypeKind::REAL: {
      return valueCount * sizeof(float);
    }
    case TypeKind::DOUBLE: {
      return valueCount * sizeof(double);
    }
    case TypeKind::DATE: {
      return valueCount * sizeof(uint32_t);
    }
    case TypeKind::VARCHAR: {
      auto stringStats =
          dynamic_cast<const dwio::common::StringColumnStatistics*>(&s);
      if (!stringStats) {
        return std::nullopt;
      }
      auto length = stringStats->getTotalLength();
      if (!length) {
        return std::nullopt;
      }
      return length.value();
    }
    case TypeKind::VARBINARY: {
      auto binaryStats =
          dynamic_cast<const dwio::common::BinaryColumnStatistics*>(&s);
      if (!binaryStats) {
        return std::nullopt;
      }
      auto length = binaryStats->getTotalLength();
      if (!length) {
        return std::nullopt;
      }
      return length.value();
    }
    case TypeKind::TIMESTAMP: {
      return valueCount * sizeof(uint64_t) * 2;
    }
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW: {
      // start the estimate with the offsets and hasNulls vectors sizes
      size_t totalEstimate = valueCount * (sizeof(uint8_t) + sizeof(uint64_t));
      for (int32_t i = 0; i < t.subtypesSize() &&
           columnSelector_->shouldReadNode(t.subtypes(i));
           ++i) {
        auto subtypeEstimate =
            estimatedRowSizeHelper(footer, stats, t.subtypes(i));
        if (subtypeEstimate.has_value()) {
          totalEstimate += subtypeEstimate.value();
        } else {
          return std::nullopt;
        }
      }
      return totalEstimate;
    }
    default:
      return std::nullopt;
  }
}

std::optional<size_t> DwrfRowReader::estimatedRowSize() const {
  auto& reader = getReader();
  auto& footer = reader.getFooter();

  if (!footer.hasNumberOfRows()) {
    return std::nullopt;
  }

  if (footer.numberOfRows() < 1) {
    return 0;
  }

  // Estimate with projections
  constexpr uint32_t ROOT_NODE_ID = 0;
  auto stats = reader.getStatistics();
  auto projectedSize = estimatedRowSizeHelper(footer, *stats, ROOT_NODE_ID);
  if (projectedSize.has_value()) {
    return projectedSize.value() / footer.numberOfRows();
  }

  return std::nullopt;
}

DwrfReader::DwrfReader(
    const ReaderOptions& options,
    std::unique_ptr<dwio::common::BufferedInput> input)
    : readerBase_(std::make_unique<ReaderBase>(
          options.getMemoryPool(),
          std::move(input),
          options.getDecrypterFactory(),
          options.getDirectorySizeGuess(),
          options.getFilePreloadThreshold(),
          options.getFileFormat() == FileFormat::ORC ? FileFormat::ORC
                                                     : FileFormat::DWRF)),
      options_(options) {}

std::unique_ptr<StripeInformation> DwrfReader::getStripe(
    uint32_t stripeIndex) const {
  DWIO_ENSURE_LT(
      stripeIndex, getNumberOfStripes(), "stripe index out of range");
  auto stripeInfo = readerBase_->getFooter().stripes(stripeIndex);

  return std::make_unique<StripeInformationImpl>(
      stripeInfo.offset(),
      stripeInfo.indexLength(),
      stripeInfo.dataLength(),
      stripeInfo.footerLength(),
      stripeInfo.numberOfRows());
}

std::vector<std::string> DwrfReader::getMetadataKeys() const {
  std::vector<std::string> result;
  auto& footer = readerBase_->getFooter();
  result.reserve(footer.metadataSize());
  for (int32_t i = 0; i < footer.metadataSize(); ++i) {
    result.push_back(footer.metadata(i).name());
  }
  return result;
}

std::string DwrfReader::getMetadataValue(const std::string& key) const {
  auto& footer = readerBase_->getFooter();
  for (int32_t i = 0; i < footer.metadataSize(); ++i) {
    if (footer.metadata(i).name() == key) {
      return footer.metadata(i).value();
    }
  }
  DWIO_RAISE("key not found");
}

bool DwrfReader::hasMetadataValue(const std::string& key) const {
  auto& footer = readerBase_->getFooter();
  for (int32_t i = 0; i < footer.metadataSize(); ++i) {
    if (footer.metadata(i).name() == key) {
      return true;
    }
  }
  return false;
}

uint64_t maxStreamsForType(const TypeWrapper& type) {
  if (type.format() == DwrfFormat::kOrc) {
    switch (type.kind()) {
      case TypeKind::ROW:
        return 1;
      case TypeKind::SMALLINT:
      case TypeKind::INTEGER:
      case TypeKind::BIGINT:
      case TypeKind::REAL:
      case TypeKind::DOUBLE:
      case TypeKind::BOOLEAN:
      case TypeKind::DATE:
      case TypeKind::ARRAY:
      case TypeKind::MAP:
        return 2;
      case TypeKind::VARBINARY:
      case TypeKind::SHORT_DECIMAL:
      case TypeKind::LONG_DECIMAL:
      case TypeKind::TIMESTAMP:
        return 3;
      case TypeKind::TINYINT:
      case TypeKind::VARCHAR:
        return 4;
      default:
        return 0;
    }
  }

  // DWRF
  switch (type.kind()) {
    case TypeKind::ROW:
      return 1;
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
      return 2;
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      return 3;
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return 4;
    case TypeKind::VARCHAR:
      return 7;
    default:
      return 0;
  }
}

uint64_t DwrfReader::getMemoryUse(int32_t stripeIx) {
  ColumnSelector cs(readerBase_->getSchema());
  return getMemoryUse(*readerBase_, stripeIx, cs);
}

uint64_t DwrfReader::getMemoryUseByFieldId(
    const std::vector<uint64_t>& include,
    int32_t stripeIx) {
  ColumnSelector cs(readerBase_->getSchema(), include);
  return getMemoryUse(*readerBase_, stripeIx, cs);
}

uint64_t DwrfReader::getMemoryUseByName(
    const std::vector<std::string>& names,
    int32_t stripeIx) {
  ColumnSelector cs(readerBase_->getSchema(), names);
  return getMemoryUse(*readerBase_, stripeIx, cs);
}

uint64_t DwrfReader::getMemoryUseByTypeId(
    const std::vector<uint64_t>& include,
    int32_t stripeIx) {
  ColumnSelector cs(readerBase_->getSchema(), include, true);
  return getMemoryUse(*readerBase_, stripeIx, cs);
}

uint64_t DwrfReader::getMemoryUse(
    ReaderBase& readerBase,
    int32_t stripeIx,
    const ColumnSelector& cs) {
  uint64_t maxDataLength = 0;
  auto& footer = readerBase.getFooter();
  if (stripeIx >= 0 && stripeIx < footer.stripesSize()) {
    uint64_t stripe = footer.stripes(stripeIx).dataLength();
    if (maxDataLength < stripe) {
      maxDataLength = stripe;
    }
  } else {
    for (int32_t i = 0; i < footer.stripesSize(); i++) {
      uint64_t stripe = footer.stripes(i).dataLength();
      if (maxDataLength < stripe) {
        maxDataLength = stripe;
      }
    }
  }

  bool hasStringColumn = false;
  uint64_t nSelectedStreams = 0;
  for (int32_t i = 0; !hasStringColumn && i < footer.typesSize(); i++) {
    if (cs.shouldReadNode(i)) {
      const auto type = footer.types(i);
      nSelectedStreams += maxStreamsForType(type);
      switch (type.kind()) {
        case TypeKind::VARCHAR:
        case TypeKind::VARBINARY: {
          hasStringColumn = true;
          break;
        }
        default: {
          break;
        }
      }
    }
  }

  /* If a string column is read, use stripe datalength as a memory estimate
   * because we don't know the dictionary size. Multiply by 2 because
   * a string column requires two buffers:
   * in the input stream and in the seekable input stream.
   * If no string column is read, estimate from the number of streams.
   */
  uint64_t memory = hasStringColumn ? 2 * maxDataLength
                                    : std::min(
                                          uint64_t(maxDataLength),
                                          nSelectedStreams *
                                              readerBase.getBufferedInput()
                                                  .getReadFile()
                                                  ->getNaturalReadSize());

  // Do we need even more memory to read the footer or the metadata?
  auto footerLength = readerBase.getPostScript().footerLength();
  if (memory < footerLength + readerBase.getDirectorySizeGuess()) {
    memory = footerLength + readerBase.getDirectorySizeGuess();
  }

  // Account for firstRowOfStripe.
  memory += static_cast<uint64_t>(footer.stripesSize()) * sizeof(uint64_t);

  // Decompressors need buffers for each stream
  uint64_t decompressorMemory = 0;
  auto compression = readerBase.getCompressionKind();
  if (compression != dwio::common::CompressionKind_NONE) {
    for (int32_t i = 0; i < footer.typesSize(); i++) {
      if (cs.shouldReadNode(i)) {
        const auto type = footer.types(i);
        decompressorMemory +=
            maxStreamsForType(type) * readerBase.getCompressionBlockSize();
      }
    }
    if (compression == dwio::common::CompressionKind_SNAPPY) {
      decompressorMemory *= 2; // Snappy decompressor uses a second buffer
    }
  }

  return memory + decompressorMemory;
}

std::unique_ptr<dwio::common::RowReader> DwrfReader::createRowReader(
    const RowReaderOptions& opts) const {
  auto rowReader = std::make_unique<DwrfRowReader>(readerBase_, opts);
  // Load the first stripe on construction so that readers created in
  // background have a reader tree and can preload the first
  // stripe. Also the reader tree needs to exist in order to receive
  // adaptation from a previous reader.
  rowReader->startNextStripe();
  return rowReader;
}

std::unique_ptr<DwrfRowReader> DwrfReader::createDwrfRowReader(
    const RowReaderOptions& opts) const {
  auto rowReader = std::make_unique<DwrfRowReader>(readerBase_, opts);
  // Load the first stripe on construction so that readers created in
  // background have a reader tree and can preload the first
  // stripe. Also the reader tree needs to exist in order to receive
  // adaptation from a previous reader.
  rowReader->startNextStripe();
  return rowReader;
}

std::unique_ptr<DwrfReader> DwrfReader::create(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const ReaderOptions& options) {
  return std::make_unique<DwrfReader>(options, std::move(input));
}

void registerDwrfReaderFactory() {
  dwio::common::registerReaderFactory(std::make_shared<DwrfReaderFactory>());
}

void unregisterDwrfReaderFactory() {
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::DWRF);
}

} // namespace facebook::velox::dwrf
