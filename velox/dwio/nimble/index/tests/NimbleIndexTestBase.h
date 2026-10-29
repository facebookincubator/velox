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

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "folly/coro/BlockingWait.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/nimble/common/HybridFlatMap.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/NimbleIndexProjector.h"
#include "velox/dwio/nimble/index/NimbleIterator.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/tablet/BlockCache.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::test {

using namespace velox;
using namespace velox::dwio::common;
using index::ClusterIndexConfigBuilder;
using Subfield = velox::common::Subfield;

serde::TabletChunkHeader readEmbeddedTabletChunkHeader(
    const folly::IOBuf& slice);
RowRange readEmbeddedRowRange(const folly::IOBuf& slice);
std::optional<std::string> readEmbeddedResumeKey(const folly::IOBuf& slice);
folly::IOBuf coalesceChunkSlice(const folly::IOBuf& slice);
HybridFlatMap makeHybridFlatMapConfig(
    const std::vector<std::vector<std::string>>& explicitGroups);
NimbleIndexProjector::RowProjectionResult projectRows(
    NimbleIndexProjector& projector,
    const NimbleIndexProjector::Request& request,
    const NimbleIndexProjector::Options& options);

// Adapts the coroutine iterator API to synchronous assertions in this test.
class TestNimbleIterator final {
 public:
  explicit TestNimbleIterator(std::unique_ptr<NimbleIterator> iterator)
      : iterator_{std::move(iterator)} {}

  void seek(const velox::serializer::EncodedKeyBounds& bounds) {
    iterator_->seek(bounds);
  }

  void seek(std::string encodedKey) {
    iterator_->seek(std::move(encodedKey));
  }

  void seekToFirst() {
    iterator_->seekToFirst();
  }

  void next() {
    iterator_->nextRow();
  }

  bool hasNext() const {
    return iterator_->hasNext();
  }

  std::string_view key() const {
    return iterator_->key();
  }

  std::string_view value() {
    return iterator_->value();
  }

  bool valid() const {
    return iterator_->valid();
  }

  NimbleIterator::Stats stats() const {
    return iterator_->stats();
  }

 private:
  std::unique_ptr<NimbleIterator> iterator_;
};

struct TestParam {
  bool cacheMetadata;
  bool pinMetadata;
  // Applies to the iterators and cursors that the iterator suites create.
  bool cacheEncodingView{true};

  std::string debugString() const {
    return fmt::format(
        "cacheMetadata={}, pinMetadata={}, cacheEncodingView={}",
        cacheMetadata,
        pinMetadata,
        cacheEncodingView);
  }
};

// Shares indexed files and reader-cache configuration across the test suites.
class NimbleIndexProjectorTestBase : public ::testing::Test {
 protected:
  static void SetUpTestCase();

  void SetUp() override;

  void TearDown() override;

  // Rows per batch produced by singleFeatureBatch().
  static constexpr vector_size_t kFeatureRowsPerBatch = 4;

  static RowTypePtr singleFeatureRowType();

  // Builds a batch whose flat map carries exactly one key, so writing several
  // with different keys yields stripes that project nothing for a projection
  // pinned to one of them.
  RowVectorPtr singleFeatureBatch(int64_t keyBase, StringView featureKey);

  // Restricts default writer candidates to encodings supported by stream views.
  static EncodingSelectionPolicyCreator encodingViewSelectionPolicy();

  // Writes sorted data with an index on the key column.
  void writeData(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<std::string>& indexColumns,
      const folly::F14FastMap<std::string, std::set<std::string>>&
          flatMapColumns = {},
      uint64_t stripeSize = 4 << 10 /* 4KB */,
      bool enableStreamDeduplication = true,
      bool compactRowCountEncoding = false,
      CompressionType chunkCompressionType = CompressionType::Uncompressed,
      bool skipConstantFlatMapInMapStreams = false,
      bool enableChunking = true,
      bool enableStreamChecksums = false,
      StripeGroup::EncodingLayout stripeGroupEncodingLayout =
          StripeGroup::EncodingLayout::kRaw,
      std::optional<uint32_t> metadataFlushThreshold = std::nullopt);

  // Returns an ascending, duplicate-free cluster index over `indexColumns`.
  static std::shared_ptr<const index::IndexConfig> makeClusterIndexConfig(
      const std::vector<std::string>& indexColumns);

  // Writes `batches` into sinkData_ under `options` and drops the readers
  // holding the previous file.
  void writeBatches(
      const std::vector<RowVectorPtr>& batches,
      WriterOptions options);

  // Replaces the file in sinkData_ with `data` and drops the readers holding
  // the previous file.
  void setFile(std::string data);

  // Drops the cached tablet readers, which hold the previous file, and the key
  // encoder, which the next file's key columns may not match.
  void resetReaders();

  // Opens the file in sinkData_ directly, without a projector.
  std::shared_ptr<TabletReader> openSinkTablet();

 public:
  static std::vector<TestParam> getTestParams();

  // Returns getTestParams() with cacheEncodingView set to both values, for the
  // suites that create iterators and cursors.
  static std::vector<TestParam> getIteratorTestParams();

  // Names an instance of the iterator suites after its parameters.
  static std::string iteratorTestParamName(
      const ::testing::TestParamInfo<TestParam>& info);

 protected:
  // Flips a byte inside every stream of stripe 0 of the file currently in
  // sinkData_, so the damage lands in stream payload rather than metadata
  // regardless of which streams a projection selects. Unprojected streams are
  // never read, so corrupting them is inert.
  void corruptProjectedStreams();

  // Flips the checksum byte of every stream trailer in stripe 0 of the file in
  // sinkData_, leaving the streams themselves intact.
  void corruptStreamTrailers();

  // Flips, for every non-empty stream of stripe 0, the byte `byteInStream`
  // picks, relative to the stream's start. Deduplicated streams share their
  // bytes, and flipping a byte twice would undo it, so each is flipped once.
  template <typename ByteInStream>
  void corruptStripe0Streams(ByteInStream byteInStream) {
    auto tablet = openSinkTablet();
    const auto stripe = tablet->stripeIdentifier(0);
    std::vector<TabletReader::StreamMetadata> locations(
        tablet->streamCount(stripe));
    tablet->streamLocations(stripe, locations);

    std::set<uint64_t> byteOffsets;
    for (const auto& location : locations) {
      if (location.size == 0) {
        continue;
      }
      byteOffsets.insert(
          tablet->stripeOffset(0) + location.offset + byteInStream(location));
    }
    ASSERT_FALSE(byteOffsets.empty());
    for (const auto byteOffset : byteOffsets) {
      ASSERT_LT(byteOffset, sinkData_.size());
      sinkData_[byteOffset] = static_cast<char>(sinkData_[byteOffset] ^ 0xFF);
    }
  }

  // Returns how many non-empty streams of stripe 0 of the file in sinkData_
  // point at bytes another stream already holds.
  size_t countDeduplicatedStreams();

  // Rewrites, in place, the kind of the first stream trailer field the file in
  // sinkData_ declares, so the trailer reads as one this reader does not know.
  void setFirstStreamTrailerFieldKind(uint8_t kind);

  velox::FileHandle makeFileHandle();

  velox::FileHandle makeFileHandle(
      std::shared_ptr<velox::ReadFile> readFile,
      std::string_view id);

  std::unique_ptr<NimbleIndexProjector> createProjector(
      const std::vector<Subfield>& projectedSubfields,
      bool setDataIoStats = true,
      bool setMetadataIoStats = true,
      bool setIndexIoStats = true);

  // Creates the iterator with the cacheEncodingView of the test parameter.
  template <typename IteratorFactory>
  std::unique_ptr<NimbleIterator> createIterator(
      IteratorFactory iteratorFactory,
      const std::vector<Subfield>& projectedSubfields,
      NimbleIterator::Options options) {
    options.cacheEncodingView = currentTestParam().cacheEncodingView;
    auto projector = createProjector(projectedSubfields);
    return iteratorFactory(std::move(projector), leafPool_.get(), options);
  }

  // Returns the decoder of the tests, which makes one entry of each row: the
  // cluster-index key of the row, and the CompactRow of the row as the value.
  static NimbleIterator::EntryDecoder compactRowDecoder();

  // Creates a row iterator over the entries of 'compactRowDecoder()'.
  std::unique_ptr<NimbleIterator> createRowIterator(
      const std::vector<Subfield>& projectedSubfields,
      const NimbleIterator::Options& options);

  // Creates a row iterator over the entries that 'entryDecoder' decodes.
  std::unique_ptr<NimbleIterator> createEntryIterator(
      const std::vector<Subfield>& projectedSubfields,
      const NimbleIterator::EntryDecoder& entryDecoder,
      const NimbleIterator::Options& options);

  std::unique_ptr<NimbleIndexProjector> createProjectorWithFileHandle(
      const std::vector<Subfield>& projectedSubfields,
      const velox::FileHandle& fileHandle,
      bool setDataIoStats = true,
      bool setMetadataIoStats = true,
      bool setIndexIoStats = true,
      bool cacheData = false,
      bool verifyStreamChecksums = false);

  void configureReaderOptions(
      dwio::common::ReaderOptions& readerOptions,
      bool setDataIoStats,
      bool setMetadataIoStats,
      bool setIndexIoStats,
      bool cacheData,
      bool verifyStreamChecksums);

  VectorPtr deserializeProjectedBatches(
      const NimbleIndexProjector& projector,
      const std::vector<std::string_view>& batches);

  VectorPtr deserializeProjectedSlice(
      const NimbleIndexProjector& projector,
      const folly::IOBuf& slice);

  // Stripe rows the `makeWindowedSlice` key bounds select.
  static constexpr uint32_t kWindowStart = 30;
  static constexpr uint32_t kWindowEnd = 60;
  static constexpr uint32_t kWindowRows = kWindowEnd - kWindowStart;

  // A projected slice whose header window is [kWindowStart, kWindowEnd).
  // Owns the projector and the projector result because the slice borrows
  // from both.
  struct WindowedSlice {
    std::unique_ptr<NimbleIndexProjector> projector;
    NimbleIndexProjector::SerializedResult result;
    // The full stripe's `value` column, for building expectations.
    std::vector<int32_t> values;

    const folly::IOBuf& slice() const {
      return result.responses[0].slices[0];
    }
  };

  // Writes a 100-row stripe keyed on `key`, projects `value`, and returns
  // the single slice the key bounds window to [kWindowStart, kWindowEnd).
  WindowedSlice makeWindowedSlice();

  // Asserts the slice really carries the expected header window, so the
  // composition tests cannot pass vacuously on an unwindowed slice.
  void expectWindow(const WindowedSlice& windowed);

  // Same, but narrows within the slice's header window using the per-batch
  // rowRanges overload. `rowRanges` are relative to that window.
  VectorPtr deserializeProjectedSlice(
      const NimbleIndexProjector& projector,
      const folly::IOBuf& slice,
      const std::vector<RowRange>& rowRanges);

  // Compares decoded output rows with an expected vector. The Deserializer
  // honors the header's row range via skip+next at decode time, so output
  // contains exactly the requested rows starting at index 0.
  // `assertEqualVectors` internally asserts equal size.
  void expectVectorRows(const VectorPtr& output, const VectorPtr& expected);

  // Creates encoded key bounds for a point lookup on a single int64 key.
  velox::serializer::EncodedKeyBounds makePointLookup(
      const RowTypePtr& keyType,
      const std::vector<std::string>& indexColumns,
      int64_t key);

  // Creates encoded key bounds for a range scan on a single int64 key.
  // The range is [lowerKey, upperKey) (lower inclusive, upper exclusive).
  velox::serializer::EncodedKeyBounds makeRangeLookup(
      const RowTypePtr& keyType,
      const std::vector<std::string>& indexColumns,
      int64_t lowerKey,
      int64_t upperKey);

  // Writes deterministic data with keys 0..numBatches*rowsPerBatch-1 across
  // multiple stripes (one stripe per batch).
  void writeResumeKeyTestData(int rowsPerBatch = 100, int numBatches = 10);

  std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> indexIoStats_{
      std::make_shared<io::IoStatistics>()};

  void ensureTabletReaderCache();

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> ioExecutor_;
  std::string sinkData_;
  std::unique_ptr<velox::serializer::KeyEncoder> keyEncoder_;
  std::unique_ptr<TabletReaderCache> tabletReaderCache_;
  // Lazily looked up from the process-wide BlockCache registry the first time
  // a projector is created with caching enabled. Non-owning; the registry
  // owns the wrapper and its lifetime matches the backend AsyncDataCache.
  BlockCache* blockCache_{nullptr};

  virtual TestParam currentTestParam() const = 0;

  virtual velox::cache::AsyncDataCache* dataCache() const;
};

class NimbleIndexProjectorTest
    : public NimbleIndexProjectorTestBase,
      public ::testing::WithParamInterface<TestParam> {
 protected:
  static void SetUpTestCase();

  static void TearDownTestCase();

  void TearDown() override;

  TestParam currentTestParam() const override;

  velox::cache::AsyncDataCache* dataCache() const override;

 private:
  inline static std::shared_ptr<velox::cache::AsyncDataCache> dataCache_;
};

} // namespace facebook::nimble::test
