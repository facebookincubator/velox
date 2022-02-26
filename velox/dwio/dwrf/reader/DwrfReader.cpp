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
#include "velox/dwio/common/exception/Exception.h"
#include "velox/dwio/dwrf/common/CachedBufferedInput.h"
#include "velox/dwio/dwrf/reader/SelectiveColumnReader.h"

DEFINE_bool(prefetch_stripes, false, "Enable prefetch of stripes.");

namespace facebook::velox::dwrf {

using dwio::common::InputStream;
using dwio::common::ReaderOptions;
using dwio::common::RowReaderOptions;

std::unique_ptr<dwio::common::RowReader> DwrfReader::createRowReader(
    const RowReaderOptions& opts) const {
  auto rowReader = std::make_unique<DwrfRowReader>(readerBase_, opts);
  if (FLAGS_prefetch_stripes) {
    rowReader->startNextStripe();
  }
  return rowReader;
}

std::unique_ptr<DwrfRowReader> DwrfReader::createDwrfRowReader(
    const RowReaderOptions& opts) const {
  return std::make_unique<DwrfRowReader>(readerBase_, opts);
}

void DwrfRowReader::checkSkipStrides(
    const StatsContext& context,
    uint64_t strideSize) {
  if (currentRowInStripe % strideSize != 0) {
    return;
  }

  if (currentRowInStripe == 0 || recomputeStridesToSkip_) {
    stridesToSkip_ = columnReader_->filterRowGroups(strideSize, context);
    recomputeStridesToSkip_ = false;
  }

  if (stridesToSkip_.empty()) {
    return;
  }

  bool foundStridesToSkip = false;
  auto currentStride = currentRowInStripe / strideSize;
  for (auto strideToSkip : stridesToSkip_) {
    if (currentStride < strideToSkip) {
      break;
    }

    if (currentStride == strideToSkip) {
      foundStridesToSkip = true;
      currentRowInStripe =
          std::min(currentRowInStripe + strideSize, rowsInCurrentStripe);
      currentStride++;
      skippedStrides_++;
    }
  }
  if (foundStridesToSkip && currentRowInStripe < rowsInCurrentStripe) {
    columnReader_->seekToRowGroup(currentStride);
  }
}

uint64_t DwrfRowReader::next(uint64_t size, VectorPtr& result) {
  if (!FLAGS_prefetch_stripes) {
    for (;;) {
      if (currentStripe >= lastStripe) {
        return 0;
      }
      auto numRows = nextInStripe(size, result);
      if (currentRowInStripe >= rowsInCurrentStripe) {
        currentStripe += 1;
        currentRowInStripe = 0;
        newStripeLoaded = false;
      }
      if (numRows) {
        return numRows;
      }
    }
  }

  for (;;) {
    if (currentStripe >= lastStripe) {
      return 0;
    }
    DwrfRowReader* FOLLY_NONNULL rowReader;
    if (currentStripe == firstStripe) {
      rowReader = this;
    } else {
      if (startWithNewDelegate_) {
        startWithNewDelegate_ = false;
        delegate_ = readerForStripe(currentStripe);
      }
      rowReader = delegate_.get();
    }
    VELOX_CHECK(rowReader);
    bool isFirstBatch = rowReader->currentRowInStripe == 0;
    auto numRows = rowReader->nextInStripe(size, result);
    if (isFirstBatch && currentStripe + 1 < lastStripe) {
      // Start prefetch of next stripe after first batch of current.
      preloadStripe(currentStripe + 1);
    }
    if (rowReader->currentRowInStripe >= rowReader->rowsInCurrentStripe) {
      ++currentStripe;
      startWithNewDelegate_ = true;
    }
    if (numRows) {
      return numRows;
    }
  }
}

uint64_t DwrfRowReader::nextInStripe(uint64_t size, VectorPtr& result) {
  DWIO_ENSURE_GT(size, 0);
  auto& footer = getReader().getFooter();
  StatsContext context(
      getReader().getWriterName(), getReader().getWriterVersion());

  if (currentStripe >= lastStripe) {
    if (lastStripe > 0) {
      previousRow = firstRowOfStripe[lastStripe - 1] +
          footer.stripes(lastStripe - 1).numberofrows();
    } else {
      previousRow = 0;
    }
    return 0;
  }

  if (currentRowInStripe == 0) {
    startNextStripe();
  }

  auto strideSize = footer.rowindexstride();
  if (LIKELY(strideSize > 0)) {
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

    columnReader_->next(rowsToRead, result);
  }

  // update row number
  previousRow = firstRowOfStripe[currentStripe] + currentRowInStripe;
  currentRowInStripe += rowsToRead;
  return rowsToRead;
}

void DwrfRowReader::preloadStripe(int32_t stripeIndex) {
  auto it = prefetchedStripeReaders_.find(stripeIndex);
  if (it != prefetchedStripeReaders_.end()) {
    return;
  }
  auto executor = getReader().bufferedInputFactory().executor();
  if (!executor) {
    return;
  }
  auto& footer = getReader().getFooter();
  DWIO_ENSURE_LT(stripeIndex, footer.stripes_size(), "invalid stripe index");
  auto& stripe = footer.stripes(stripeIndex);

  auto newOpts = options_;
  newOpts.range(stripe.offset(), 1);
  auto readerBase = readerBaseShared();
  auto source = std::make_shared<AsyncSource<DwrfRowReader>>(
      [readerBase, stripeIndex, newOpts]() {
        auto stripeReader =
            std::make_unique<DwrfRowReader>(readerBase, newOpts);
        stripeReader->startNextStripe();
        return stripeReader;
      });
  executor->add([source]() { source->prepare(); });
  prefetchedStripeReaders_[stripeIndex] = std::move(source);
}

std::unique_ptr<DwrfRowReader> DwrfRowReader::readerForStripe(
    int32_t stripeIndex) {
  auto it = prefetchedStripeReaders_.find(stripeIndex);
  if (it == prefetchedStripeReaders_.end()) {
    return nullptr;
  }
  return it->second->move();
}

void DwrfRowReader::resetFilterCaches() {
  dynamic_cast<SelectiveColumnReader*>(columnReader())->resetFilterCaches();
  recomputeStridesToSkip_ = true;
}

bool DwrfRowReader::allPrefetchIssued() const {
  return currentStripe + 1 >= lastStripe ||
      prefetchedStripeReaders_.find(lastStripe - 1) !=
      prefetchedStripeReaders_.end();
}

bool DwrfRowReader::moveAdaptation(RowReader& other) {
  auto otherReader = dynamic_cast<DwrfRowReader*>(&other);
  if (!columnReader_ || !otherReader->columnReader_) {
    return false;
  }
  columnReader_->moveScanSpec(*otherReader->columnReader_);
  return true;
}

std::unique_ptr<DwrfReader> DwrfReader::create(
    std::unique_ptr<InputStream> stream,
    const ReaderOptions& options) {
  return std::make_unique<DwrfReader>(options, std::move(stream));
}

void registerDwrfReaderFactory() {
  dwio::common::registerReaderFactory(std::make_shared<DwrfReaderFactory>());
}

void unregisterDwrfReaderFactory() {
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::ORC);
}

} // namespace facebook::velox::dwrf
