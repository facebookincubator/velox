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

#include "velox/dwio/pagefile/PagefileReader.h"

#include <chrono>

#include "velox/dwio/common/OnDemandUnitLoader.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/common/exception/Exception.h"
#include "velox/dwio/dwrf/reader/ColumnReader.h"
#include "velox/dwio/dwrf/reader/StreamLabels.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::pagefile {

using dwio::common::ColumnSelector;
using dwio::common::FileFormat;
using dwio::common::LoadUnit;
using dwio::common::ReaderOptions;
using dwio::common::RowReaderOptions;
using dwio::common::UnitLoader;
using dwio::common::UnitLoaderFactory;

PagefileRowReader::PagefileRowReader(
    const RowReaderOptions& opts)
    options_(opts) {
}

uint64_t PagefileRowReader::next(
    uint64_t size,
    velox::VectorPtr& result,
    const dwio::common::Mutation* mutation) {
  auto nextRow = nextRowNumber();
  if (nextRow == kAtEnd) {
    if (!isEmptyFile()) {
      previousRow_ = firstRowOfStripe_[stripeCeiling_ - 1] +
          getReader().getFooter().stripes(stripeCeiling_ - 1).numberOfRows();
    } else {
      previousRow_ = 0;
    }
    return 0;
  }
  auto rowsToRead = nextReadSize(size);
  nextRowNumber_.reset();
  previousRow_ = nextRow;
  // Record strideIndex for use by the columnReader_ which may delay actual
  // reading of the data.
  auto strideSize = getReader().getFooter().rowIndexStride();
  strideIndex_ = strideSize > 0 ? currentRowInStripe_ / strideSize : 0;
  const auto loadUnitIdx = currentStripe_ - firstStripe_;
  unitLoader_->onRead(loadUnitIdx, currentRowInStripe_, rowsToRead);
  readNext(rowsToRead, mutation, result);
  currentRowInStripe_ += rowsToRead;
  return rowsToRead;
}

void PagefileRowReader::resetFilterCaches() {
  if (getSelectiveColumnReader()) {
    getSelectiveColumnReader()->resetFilterCaches();
    recomputeStridesToSkip_ = true;
  }

  // For columnReader_, this is no-op.
}

uint64_t PagefileRowReader::next(
    uint64_t size,
    velox::VectorPtr& result,
    const Mutation* mutation = nullptr) {

}

int64_t PagefileRowReader::nextRowNumber() {

}

int64_t PagefileRowReader::nextReadSize(uint64_t size) {

}

void PagefileRowReader::updateRuntimeStats(RuntimeStatistics& stats) const {

}

void PagefileRowReader::resetFilterCaches() {}

std::optional<size_t> PagefileRowReader::estimatedRowSize() const {}

bool PagefileRowReader::allPrefetchIssued() const {
  return false;
}

std::optional<std::vector<PrefetchUnit>> PagefileRowReader::prefetchUnits() {
  return std::nullopt;
}

std::optional<size_t> PagefileRowReader::::estimatedRowSize() const {
  return std::nullopt;
}

PagefileReader::PagefileReader(
    const ReaderOptions& options,
    std::unique_ptr<dwio::common::BufferedInput> input)
    : options_(options) {
}

std::unique_ptr<dwio::common::RowReader> PagefileReader::createRowReader(
    const RowReaderOptions& opts) const {
  return createPagefileRowReader(opts);
}

std::unique_ptr<DwrfRowReader> PagefileReader::createPagefileRowReader(
    const RowReaderOptions& opts) const {
  auto rowReader = std::make_unique<DwrfRowReader>(readerBase_, opts);
  if (opts.getEagerFirstStripeLoad()) {
    // Load the first stripe on construction so that readers created in
    // background have a reader tree and can preload the first
    // stripe. Also the reader tree needs to exist in order to receive
    // adaptation from a previous reader.
    rowReader->loadCurrentStripe();
  }
  return rowReader;
}

std::unique_ptr<PagefileReader> PagefileReader::create(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const ReaderOptions& options) {
  return std::make_unique<PagefileReader>(options, std::move(input));
}

void registerPagefileReaderFactory() {
  dwio::common::registerReaderFactory(std::make_shared<PagefileReaderFactory>());
}

void unregisterPagefileReaderFactory() {
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::PAGEFILE);
}

} // namespace facebook::velox::pagefile
