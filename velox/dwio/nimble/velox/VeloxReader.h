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
#pragma once

#include <optional>
#include "folly/container/F14Set.h"
#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/ExecutorBarrier.h"
#include "velox/dwio/common/FlatMapHelper.h"
#include "velox/dwio/common/UnitLoader.h"
#include "velox/dwio/nimble/common/MetricsLogger.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/velox/FieldReader.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/StreamLabels.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

// The VeloxReader reads (a projection from) a file into a velox VectorPtr.
// The current implementation only uses FlatVector rather than any of the
// fancier vectors (dictionary, etc), and only supports the types needed for ML.

namespace facebook::nimble {

struct VeloxReadParams : public FieldReaderParams {
  uint64_t fileRangeStartOffset = 0;
  uint64_t fileRangeEndOffset = std::numeric_limits<uint64_t>::max();

  // Optional reader decoding executor. When supplied, decoding into a Velox
  // vector will be parallelized by this executor, if the column type supports
  // parallel decoding.
  std::shared_ptr<folly::Executor> decodingExecutor{};

  // Metric logger with pro-populated access info.
  std::shared_ptr<MetricsLogger> metricsLogger{};

  // Report the number of stripes that will be read (consider the given range).
  std::function<void(uint32_t)> stripeCountCallback{};

  // Report the Wall time (ms) that we're blocked waiting on IO.
  std::function<void(std::chrono::high_resolution_clock::duration)>
      blockedOnIoCallback{};

  // Report the Wall time (us) that we spend decoding.
  std::function<void(std::chrono::high_resolution_clock::duration)>
      decodingTimeCallback{};

  // Factory with the algorithm to load stripes (units).
  // If nullptr we'll use the default one, that doesn't pre-load stripes.
  std::shared_ptr<velox::dwio::common::UnitLoaderFactory> unitLoaderFactory{};

  /// Resolves External shared integer dictionaries referenced by the file.
  std::shared_ptr<const ExternalDictionaryResolver> externalDictionaryResolver;

  /// Creates an encoding for one serialized stream chunk.
  using StreamEncodingFactory = std::function<std::unique_ptr<Encoding>(
      velox::memory::MemoryPool&,
      std::string_view,
      std::function<void*(uint32_t)>)>;

  // Used strictly for backward compatible migrations where the Encoding
  // implementations might have different read implementations, but
  // identical/backward compatible write behavior.
  StreamEncodingFactory encodingFactory =
      [](velox::memory::MemoryPool& pool,
         std::string_view data,
         const std::function<void*(uint32_t)>& stringBufferFactory)
      -> std::unique_ptr<Encoding> {
    return legacy::EncodingFactory().create(
        pool, data, stringBufferFactory, Encoding::Options{});
  };
};

class VeloxReader {
 public:
  static constexpr uint64_t kConservativeEstimatedRowSize = 1L << 20; // 1MB

  VeloxReader(
      velox::ReadFile* file,
      velox::memory::MemoryPool& pool,
      std::shared_ptr<const velox::dwio::common::ColumnSelector> selector =
          nullptr,
      const VeloxReadParams& params = {});

  VeloxReader(
      std::shared_ptr<velox::ReadFile> file,
      velox::memory::MemoryPool& pool,
      std::shared_ptr<const velox::dwio::common::ColumnSelector> selector =
          nullptr,
      const VeloxReadParams& params = {});

  VeloxReader(
      std::shared_ptr<const TabletReader> tabletReader,
      velox::memory::MemoryPool& pool,
      std::shared_ptr<const velox::dwio::common::ColumnSelector> selector =
          nullptr,
      VeloxReadParams params = {});

  ~VeloxReader();

  // Returns the estimated row size from the current stripe in bytes.
  uint64_t estimatedRowSize();

  // Fills |result| with up to rowCount new rows, returning whether any new rows
  // were read. |result| may be nullptr, in which case it will be allocated via
  // pool_. If it is not nullptr its type must match type_.
  bool next(uint64_t rowCount, velox::VectorPtr& result);

  const TabletReader& tabletReader() const;

  const std::shared_ptr<const velox::RowType>& type() const;

  const std::shared_ptr<const Type>& schema() const;

  const std::map<std::string, std::string>& metadata() const;

  velox::memory::MemoryPool& memoryPool() const {
    return pool_;
  }

  // Returns the current row number the reader is pointing to. If there are no
  // more rows to read in the file, this will return the last row number.
  uint64_t getRowNumber();

  // Seeks to |rowNumber| from the beginning of the file (row 0).
  // If |rowNumber| is greater than the number of rows in the file, the
  // seek will stop at the end of file and following reads will return
  // false.
  // Returns the current row number the reader is pointing to. If
  // |rowNumber| is greater than the number of rows in the file, this will
  // return the last row number.
  uint64_t seekToRow(uint64_t rowNumber);

  // Skips |numberOfRowsToSkip| rows from current row index.
  // If |numberOfRowsToSkip| is greater than the remaining rows in the
  // file, skip will stop at the end of file, and following reads will
  // return false.
  // Returns the number of rows skipped. If |numberOfRowsToSkip| is
  // greater than the remaining rows in the file, this will return the
  // total number of rows skipped, until reaching the end of file.
  uint64_t skipRows(uint64_t numberOfRowsToSkip);

  // Loads the next stripe if any
  void loadStripeIfAny();

 private:
  // Loads the next stripe's streams.
  void loadNextStripe();

  // Returns the base encoding factory unless the stream has an alphabet.
  VeloxReadParams::StreamEncodingFactory createStreamEncodingFactory(
      uint32_t valueStreamId,
      std::unique_ptr<StreamLoader> dictionaryStream) const;

  // True if the file contain zero rows.
  bool isEmptyFile() const {
    return ((lastRow_ - firstRow_) == 0);
  }

  // Skips |rowsToSkip| in the currently loaded stripe. |rowsToSkip| must be
  // less than rowsRemaining_.
  void skipInCurrentStripe(uint64_t rowsToSkip);

  // Skips over multiple stripes, starting from a stripe with index
  // |startStripeIndex|. Keeps skipping stripes for as long as remaining rows to
  // skip is greater than the next stripe's row count.
  // This method does not load the last stripe, or skips inside the last stripe.
  // Returns the total number of rows skipped.
  uint64_t skipStripes(uint32_t startStripeIndex, uint64_t rowsToSkip);

  std::unique_ptr<velox::dwio::common::UnitLoader> getUnitLoader();

  uint32_t getUnitIndex(uint32_t stripeIndex) const;

  uint32_t getCurrentRowInStripe() const;

  velox::memory::MemoryPool& pool_;
  std::shared_ptr<const TabletReader> tabletReader_;
  std::optional<StripeIdentifier> stripeIdentifier_;
  const VeloxReadParams parameters_;
  std::shared_ptr<const Type> schema_;
  std::shared_ptr<const velox::RowType> type_;
  std::vector<uint32_t> offsets_;
  std::vector<uint64_t> cummulativeStripeRowCount_;
  folly::F14FastMap<offset_size, std::unique_ptr<Decoder>> decoders_;
  std::unique_ptr<FieldReaderFactory> rootFieldReaderFactory_;
  std::unique_ptr<FieldReader> rootReader_;
  mutable std::optional<std::map<std::string, std::string>> metadata_;
  uint32_t firstStripe_;
  uint32_t lastStripe_;
  uint64_t firstRow_;
  uint64_t lastRow_;

  // Reading state for reader
  uint32_t nextStripe_{0};
  uint64_t rowsRemainingInStripe_{0};

  // stripe currently loaded.
  std::optional<uint32_t> loadedStripe_;

  // Row size estimation cache so that for the same stripe no repeated
  // estimation is done.
  uint64_t cachedRowSizeEstimation_{0};
  std::optional<uint32_t> cachedRowSizeEstimationStripeIdx_;

  std::unique_ptr<velox::dwio::common::ExecutorBarrier> barrier_;

  // Right now each reader is considered its own session if not passed from
  // writer option.
  std::shared_ptr<MetricsLogger> logger_;

  std::unique_ptr<velox::dwio::common::UnitLoader> unitLoader_;

  friend class VeloxReaderHelper;
};

} // namespace facebook::nimble
