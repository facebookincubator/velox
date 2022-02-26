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

#pragma once

#include "velox/common/base/AsyncSource.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/dwrf/reader/ColumnReader.h"
#include "velox/dwio/dwrf/reader/DwrfReaderShared.h"

namespace facebook::velox::dwrf {

class DwrfRowReader : public DwrfRowReaderShared {
 protected:
  void resetColumnReaderImpl() override {
    columnReader_.reset();
  }

  void createColumnReaderImpl(StripeStreams& stripeStreams) override {
    columnReader_ = (columnReaderFactory_ ? columnReaderFactory_.get()
                                          : ColumnReaderFactory::baseFactory())
                        ->build(
                            getColumnSelector().getSchemaWithId(),
                            getReader().getSchemaWithId(),
                            stripeStreams);
  }

  void seekImpl() override {
    columnReader_->skip(currentRowInStripe);
  }

 public:
  /**
   * Constructor that lets the user specify additional options.
   * @param contents of the file
   * @param options options for reading
   */
  DwrfRowReader(
      const std::shared_ptr<ReaderBase>& reader,
      const dwio::common::RowReaderOptions& options)
      : DwrfRowReaderShared{reader, options} {}

  ~DwrfRowReader() override = default;

  // Returns number of rows read. Guaranteed to be less then or equal
  // to size. May initiate prefetch of stripes beyond the current one.
  uint64_t next(uint64_t size, VectorPtr& result) override;

  void updateRuntimeStats(
      dwio::common::RuntimeStatistics& stats) const override {
    if (delegate_) {
      stats.skippedStrides += delegate_->skippedStrides_;
    } else {
      stats.skippedStrides += skippedStrides_;
    }
  }

  ColumnReader* columnReader() {
    if (delegate_) {
      return delegate_->columnReader_.get();
    } else {
      return columnReader_.get();
    }
  }

  void resetFilterCaches() override;

  bool moveAdaptation(RowReader& other) override;

  bool allPrefetchIssued() const override;

 private:
  using StripeReaderSource = AsyncSource<DwrfRowReader>;

  // Gets next rows within 'this'.
  uint64_t nextInStripe(uint64_t size, VectorPtr& result);

  // Asynchronously makes a DwrfRowReader for 'stripeIndex'.
  void preloadStripe(int32_t stripeIndex);

  // Returns a single-stripe DwrfRowReader for 'stripeIndex'. This
  // must have been prepared by preloadStripe() for the same stripe
  // index.
  std::unique_ptr<DwrfRowReader> readerForStripe(int32_t stripeIndex);

  void checkSkipStrides(const StatsContext& context, uint64_t strideSize);

  std::unique_ptr<ColumnReader> columnReader_;
  std::vector<uint32_t> stridesToSkip_;

  // Number of skipped strides.
  int64_t skippedStrides_{0};

  // The RowReader for the current stripe. If set, calls are delegated
  // to 'delegate_'. Multiple delegates are prefetched concurrently
  // for consecutive stripes and then used in turn. This allows
  // multiple stripes worth of read ahead.
  std::unique_ptr<DwrfRowReader> delegate_;

  // Indicates that next() needs to take a new delegate to read a new
  // stripe. The delegate for the previous stripe must stay live for
  // serving up lazy loads even if scan is at end of stripe.
  bool startWithNewDelegate_{false};

  // Map from stripe number to prepared stripe reader.
  folly::F14FastMap<int32_t, std::shared_ptr<StripeReaderSource>>
      prefetchedStripeReaders_;

  // Set to true after clearing filter caches, i.e.  adding a dynamic
  // filter. Causes filters to be re-evaluated against stride stats on
  // next stride instead of next stripe.
  bool recomputeStridesToSkip_{false};
};

class DwrfReader : public DwrfReaderShared {
 public:
  /**
   * Constructor that lets the user specify additional options.
   * @param contents of the file
   * @param options options for reading
   * @param fileLength the length of the file in bytes
   * @param postscriptLength the length of the postscript in bytes
   */
  DwrfReader(
      const dwio::common::ReaderOptions& options,
      std::unique_ptr<dwio::common::InputStream> input)
      : DwrfReaderShared{options, std::move(input)} {}

  ~DwrfReader() override = default;

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options = {}) const override;

  std::unique_ptr<DwrfRowReader> createDwrfRowReader(
      const dwio::common::RowReaderOptions& options = {}) const;

  /**
   * Create a reader to the for the dwrf file.
   * @param stream the stream to read
   * @param options the options for reading the file
   */
  static std::unique_ptr<DwrfReader> create(
      std::unique_ptr<dwio::common::InputStream> stream,
      const dwio::common::ReaderOptions& options);

  friend class E2EEncryptionTest;
};

class DwrfReaderFactory : public dwio::common::ReaderFactory {
 public:
  DwrfReaderFactory() : ReaderFactory(dwio::common::FileFormat::ORC) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::InputStream> stream,
      const dwio::common::ReaderOptions& options) override {
    return DwrfReader::create(std::move(stream), options);
  }
};

void registerDwrfReaderFactory();

void unregisterDwrfReaderFactory();

} // namespace facebook::velox::dwrf
