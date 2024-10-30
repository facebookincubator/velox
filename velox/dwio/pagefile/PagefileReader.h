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

#include "folly/Executor.h"
#include "folly/synchronization/Baton.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/UnitLoader.h"

namespace facebook::velox::pagefile {

class ColumnReader;

class PagefileRowReader : public dwio::common::RowReader {
 public:
  /**
   * Constructor that lets the user specify additional options.
   * @param contents of the file
   * @param options options for reading
   */
  PagefileRowReader(const dwio::common::RowReaderOptions& options);

  ~PagefileRowReader() override = default;

  /**
   * Fetch the next portion of rows.
   * @param size Max number of rows to read
   * @param result output vector
   * @param mutation The mutation to be applied during the read, null means no
   *  mutation
   * @return number of rows scanned in the file (including any rows filtered out
   *  or deleted in mutation), 0 if there are no more rows to read.
   */
  uint64_t next(
      uint64_t size,
      velox::VectorPtr& result,
      const Mutation* mutation = nullptr) override;

  /**
   * Return the next row number that will be scanned in the next next() call,
   * kAtEnd when at end of file.  This row number is relative to beginning of
   * the file (0 for the first row), including all rows in the file, no matter
   * whether it's deleted or filtered during the previous next() call.
   *
   * This function is mainly used to compute the bit mask used for mutation.
   * Given a list of row numbers in the file, we can calculate the offset of
   * each rows in the bit mask based on value returned from this call.
   */
  virtual int64_t nextRowNumber() = 0;

  /**
   * Given the max number of rows to read, return the actual number of rows that
   * will be scanned, including any rows to be deleted or filtered.  Return
   * kAtEnd when at end of file.  This is also used to compute the bit mask used
   * in mutation.
   */
  int64_t nextReadSize(uint64_t size) override;

  /**
   * Update current reader statistics. The set of updated values is
   * implementation specific and depends on a format of a file being read.
   * @param stats stats to update
   */
  void updateRuntimeStats(RuntimeStatistics& stats) const override;

  /**
   * This method should be called whenever filter is modified in a ScanSpec
   * object passed to Reader::createRowReader to create this object.
   */
  void resetFilterCaches() override;

  /**
   * Get an estimated row size basing on available statistics. Can
   * differ from the actual row size due to variable-length values.
   * @return Estimate of the row size or std::nullopt if cannot estimate.
   */
  std::optional<size_t> estimatedRowSize() const override;

  // Returns true if the expected IO for 'this' is scheduled. If this
  // is true it makes sense to prefetch the next split.
  bool allPrefetchIssued() const override;

  /**
   * Returns a vector of PrefetchUnit objects describing all the prefetch units
   * owned by this RowReader. For example, a returned vector {{50, func1}, {50,
   * func2}} would represent a RowReader which has 2 prefetch units (for
   * example, a stripe for dwrf and alpha file formats). Each prefetch unit has
   * 50 rows, and func1 and func2 represent callables which will run the
   * prefetch and report a FetchResult. The FetchResult reports if the prefetch
   * was completed by the caller, if the prefetch was in progress when the
   * function was called or if the prefetch was already completed, as a result
   * of i.e. calling next and having the main thread load the stripe.
   * @return std::nullopt if the reader implementation does not support
   * prefetching.
   */
  std::optional<std::vector<PrefetchUnit>> prefetchUnits() override;
};

class PagefileReader : public dwio::common::Reader {
 public:
  /**
   * Constructor that lets the user specify reader options and input stream.
   */
  PagefileReader(
      const dwio::common::ReaderOptions& options,
      std::unique_ptr<dwio::common::BufferedInput> input);

  ~PagefileReader() override = default;

  /**
   * Get statistics for a specified column.
   * @param index column index
   * @return column statisctics
   */
  std::unique_ptr<ColumnStatistics> columnStatistics(
      uint32_t index) const override {
    return nullptr;
  }

  /**
   * Get the file schema.
   * @return file schema
   */
  const velox::RowTypePtr& rowType() const override {

  }

  /**
   * Get the file schema attributed with type and column ids.
   * @return file schema
   */
  const std::shared_ptr<const TypeWithId>& typeWithId() const override {

  }

  std::optional<uint64_t> numberOfRows() const override {
    // TODO : Add logic here
    return std::nullopt;
  }

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options = {}) const override;

  std::unique_ptr<PagefileReader> createPagefileReader(
      const dwio::common::RowReaderOptions& options = {}) const;

  /**
   * Create a reader to the for the pagefile format file.
   * @param input the stream to read
   * @param options the options for reading the file
   */
  static std::unique_ptr<PagefileReader> create(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options);

 private:
  const dwio::common::ReaderOptions options_;
};

class PagefileReaderFactory : public dwio::common::ReaderFactory {
 public:
  PagefileReaderFactory() : ReaderFactory(dwio::common::FileFormat::PAGEFILE) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options) override {
    return PagefileReader::create(std::move(input), options);
  }
};

void registerPagefileReaderFactory();

void unregisterPagefileReaderFactory();

} // namespace facebook::velox::pagefile
