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

#include "velox/dwio/nimble/selective/ReaderBase.h"

#include "velox/buffer/Buffer.h"
#include "velox/common/file/File.h"
#include "velox/dwio/common/InputStream.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/tablet/Constants.h"

namespace facebook::nimble {

using namespace facebook::velox;

namespace {
const std::string kSchemaSectionString(kSchemaSection);

class SeekableStreamLoader final : public StreamLoader {
 public:
  SeekableStreamLoader(
      std::unique_ptr<dwio::common::SeekableInputStream> input,
      uint64_t size,
      memory::MemoryPool* pool)
      : input_{std::move(input)} {
    const void* data;
    int32_t length;
    const bool hasData = input_->Next(&data, &length);
    NIMBLE_CHECK(hasData, "Shared dictionary stream ended early.");
    NIMBLE_CHECK_GT(length, 0);
    const auto contiguousSize = static_cast<uint64_t>(length);
    NIMBLE_CHECK_LE(contiguousSize, size);
    if (contiguousSize == size) {
      stream_ = {static_cast<const char*>(data), size};
      return;
    }

    buffer_ = AlignedBuffer::allocateExact<char>(size, pool);
    std::memcpy(buffer_->asMutable<char>(), data, contiguousSize);
    uint64_t copied{contiguousSize};
    while (copied < size) {
      const bool hasMoreData = input_->Next(&data, &length);
      NIMBLE_CHECK(hasMoreData, "Shared dictionary stream ended early.");
      NIMBLE_CHECK_GT(length, 0);
      const auto chunkSize = static_cast<uint64_t>(length);
      NIMBLE_CHECK_LE(copied + chunkSize, size);
      std::memcpy(buffer_->asMutable<char>() + copied, data, chunkSize);
      copied += chunkSize;
    }
    stream_ = {buffer_->as<char>(), buffer_->size()};
  }

  const std::string_view getStream() const final {
    return stream_;
  }

 private:
  // Keeps a contiguous range returned by Next() alive.
  const std::unique_ptr<dwio::common::SeekableInputStream> input_;
  // Allocated only when the input spans multiple ranges.
  BufferPtr buffer_;
  std::string_view stream_;
};

std::shared_ptr<const facebook::nimble::Type> loadSchema(
    const TabletReader& tablet) {
  auto section = tablet.loadOptionalSection(kSchemaSectionString);
  NIMBLE_CHECK(section.has_value());
  return SchemaDeserializer::deserialize(section->content().data());
}

TypePtr getFileSchema(
    const dwio::common::ReaderOptions& options,
    const TypePtr& fileSchema) {
  if (options.columnMappingMode() == dwio::common::ColumnMappingMode::kName ||
      !options.fileSchema()) {
    return fileSchema;
  }
  return dwio::common::Reader::updateColumnNames(
      fileSchema, options.fileSchema());
}

// Reads a small file fully in one storage round trip and returns a
// BufferedInput backed by an owning in-memory ReadFile, so footer and stripe
// reads are served from memory. Returns the original input unchanged when the
// file is empty, over the threshold, or an AsyncDataCache is present.
std::unique_ptr<velox::dwio::common::BufferedInput> maybePreloadInput(
    std::unique_ptr<velox::dwio::common::BufferedInput> input,
    const velox::dwio::common::ReaderOptions& options) {
  const auto readFile = input->getReadFile();
  const auto fileSize = readFile->size();
  if (fileSize == 0 || fileSize > options.filePreloadThreshold() ||
      options.cache() != nullptr) {
    return input;
  }
  auto buffer = velox::AlignedBuffer::allocateExact<char>(
      fileSize, &options.memoryPool());
  velox::dwio::common::ReadFileInputStream(
      readFile,
      velox::dwio::common::MetricsLog::voidLog(),
      options.dataIoStats().get())
      .read(
          buffer->asMutable<char>(),
          fileSize,
          /*offset=*/0,
          velox::dwio::common::LogType::FILE);
  return std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(std::move(buffer)),
      options.memoryPool());
}
} // namespace

std::shared_ptr<ReaderBase> ReaderBase::create(
    std::unique_ptr<velox::dwio::common::BufferedInput> input,
    const velox::dwio::common::ReaderOptions& options) {
  input = maybePreloadInput(std::move(input), options);

  auto tabletOptions = TabletReader::configureOptions(options);

  auto tablet = TabletReader::create(
      input->getReadFile(), &options.memoryPool(), tabletOptions);

  auto* pool = &options.memoryPool();
  const auto& randomSkip = options.randomSkip();
  const auto& scanSpec = options.scanSpec();
  const auto nimbleSchema = loadSchema(*tablet);
  auto fileSchema =
      asRowType(getFileSchema(options, convertToVeloxType(*nimbleSchema)));

  return std::shared_ptr<ReaderBase>(new ReaderBase(
      std::move(input),
      std::move(tablet),
      randomSkip,
      scanSpec,
      nimbleSchema,
      std::move(fileSchema),
      pool));
}

std::shared_ptr<ReaderBase> ReaderBase::create(
    std::unique_ptr<velox::dwio::common::BufferedInput> input,
    const std::shared_ptr<CachedTabletReader>& cachedTablet,
    const velox::dwio::common::ReaderOptions& options) {
  NIMBLE_CHECK_NOT_NULL(cachedTablet);
  auto* pool = &options.memoryPool();
  const auto& randomSkip = options.randomSkip();
  const auto& scanSpec = options.scanSpec();
  auto fileSchema =
      asRowType(getFileSchema(options, cachedTablet->veloxSchema()));

  // Aliased onto the entry so this ReaderBase owns it, not just the tablet.
  // The entry owns the IoStatistics the tablet writes its metadata and index
  // reads into, and retiring it stops those totals being watched, so an entry
  // that retired while this reader was still going would lose the rest.
  auto tablet =
      std::shared_ptr<TabletReader>(cachedTablet, cachedTablet->tablet().get());

  return std::shared_ptr<ReaderBase>(new ReaderBase(
      std::move(input),
      std::move(tablet),
      randomSkip,
      scanSpec,
      cachedTablet->nimbleSchema(),
      std::move(fileSchema),
      pool));
}

ReaderBase::ReaderBase(
    std::unique_ptr<velox::dwio::common::BufferedInput> input,
    std::shared_ptr<TabletReader> tablet,
    const std::shared_ptr<velox::random::RandomSkipTracker>& randomSkip,
    const std::shared_ptr<velox::common::ScanSpec>& scanSpec,
    std::shared_ptr<const Type> nimbleSchema,
    velox::RowTypePtr fileSchema,
    velox::memory::MemoryPool* pool)
    : input_{std::move(input)},
      tablet_{std::move(tablet)},
      pool_{pool},
      randomSkip_{randomSkip},
      scanSpec_{scanSpec},
      nimbleSchema_{std::move(nimbleSchema)},
      fileSchema_{std::move(fileSchema)},
      fileColumnStats_{[&]() -> std::vector<std::unique_ptr<ColumnStatistics>> {
        auto statsSection =
            tablet_->loadOptionalSection(std::string(kVectorizedStatsSection));
        if (!statsSection.has_value()) {
          return {};
        }
        auto fileStats =
            VectorizedFileStats::deserialize(statsSection->content(), *pool_);
        if (!fileStats) {
          return {};
        }
        return fileStats->toColumnStatistics(fileSchema_, nimbleSchema_);
      }()} {}

void LazyInput::load() {
  if (!loaded_) {
    input_->load(velox::dwio::common::LogType::STREAM_BUNDLE);
    loaded_ = true;
  }
}

LazyInput* StripeStreams::createLazyInput() {
  NIMBLE_CHECK_NULL(lazyInput_);
  lazyInput_ = std::make_unique<LazyInput>(readerBase_->input().clone());
  return lazyInput_.get();
}

std::optional<common::Region> StripeStreams::streamRegion(int streamId) const {
  NIMBLE_CHECK(stripeIdentifier_.has_value());
  const auto& tablet = readerBase_->tablet();
  if (streamId >= tablet.streamCount(*stripeIdentifier_)) {
    return std::nullopt;
  }
  const auto size = tablet.streamSize(*stripeIdentifier_, streamId);
  if (size == 0) {
    return std::nullopt;
  }
  common::Region region;
  region.offset = tablet.stripeOffset(stripe_) +
      tablet.streamOffset(*stripeIdentifier_, streamId);
  region.length = size;
  return region;
}

std::unique_ptr<dwio::common::SeekableInputStream> StripeStreams::enqueue(
    int streamId,
    bool lazyColumnIo) {
  const auto region = streamRegion(streamId);
  if (!region.has_value()) {
    return nullptr;
  }
  dwio::common::StreamIdentifier sid(streamId);
  auto& input =
      lazyColumnIo ? *lazyInput_->bufferedInput() : readerBase_->input();
  auto valueInput = input.enqueue(*region, &sid);

  const auto dictionaryStreamId =
      readerBase_->tablet().stripeDictionaryStreamId(streamId);
  if (!dictionaryStreamId.has_value()) {
    return valueInput;
  }
  const auto dictionaryStreamOffset = dictionaryStreamId.value();
  const auto dictionaryStreamRegion =
      streamRegion(static_cast<int>(dictionaryStreamOffset));
  // A value stream can use a stripe dictionary in some stripes and direct
  // encoding in others. Direct-encoded stripes omit the dictionary stream.
  if (!dictionaryStreamRegion.has_value()) {
    return valueInput;
  }
  // Each projected value stream is enqueued once and owns a distinct stripe
  // dictionary stream.
  const auto valueStreamId = static_cast<uint32_t>(streamId);
  NIMBLE_DCHECK(
      !dictionaryInputs_.contains(valueStreamId),
      "Stripe dictionary stream is already enqueued.");
  dwio::common::StreamIdentifier dictionarySid{
      static_cast<int>(dictionaryStreamOffset)};
  dictionaryInputs_.emplace(
      valueStreamId,
      DictionaryInput{
          input.enqueue(*dictionaryStreamRegion, &dictionarySid),
          dictionaryStreamRegion->length});
  return valueInput;
}

DictionaryAlphabetLoader StripeStreams::dictionaryAlphabetLoader(
    uint32_t valueStreamId) {
  NIMBLE_CHECK(stripeIdentifier_.has_value());
  auto inputIt = dictionaryInputs_.find(valueStreamId);
  if (inputIt != dictionaryInputs_.end()) {
    auto dictionaryInput = std::move(inputIt->second);
    dictionaryInputs_.erase(inputIt);
    return [_dictionaryInput = std::move(dictionaryInput),
            readerBase = readerBase_]() mutable {
      std::shared_ptr<const StreamLoader> dictionaryStreamOwner =
          std::make_shared<SeekableStreamLoader>(
              std::move(_dictionaryInput.input),
              _dictionaryInput.size,
              readerBase->pool());
      return SharedDictionaryAlphabet::create(
          dictionaryStreamOwner->getStream(),
          dictionaryStreamOwner,
          readerBase->pool());
    };
  }
  auto dictionaryAlphabet =
      readerBase_->tablet().resolveDictionaryAlphabet(valueStreamId);
  if (dictionaryAlphabet == nullptr) {
    return nullptr;
  }
  return [_dictionaryAlphabet = std::move(dictionaryAlphabet)] {
    return _dictionaryAlphabet;
  };
}

std::vector<std::optional<StreamLocation>> StripeStreams::locateStreams(
    std::span<const uint32_t> streamIds) const {
  NIMBLE_CHECK(stripeIdentifier_.has_value());

  const auto& tablet = readerBase_->tablet();
  const auto streamCount = tablet.streamCount(*stripeIdentifier_);
  // File offset where this stripe's data begins.
  const auto stripeOffset = tablet.stripeOffset(stripe_);

  std::vector<std::optional<StreamLocation>> locations(streamIds.size());
  for (size_t i = 0; i < streamIds.size(); ++i) {
    const auto streamId = streamIds[i];
    if (streamId >= streamCount) {
      continue;
    }
    const auto streamSize = tablet.streamSize(*stripeIdentifier_, streamId);
    if (streamSize == 0) {
      continue;
    }
    const auto streamOffset = tablet.streamOffset(*stripeIdentifier_, streamId);
    locations[i] = StreamLocation{
        streamId, common::Region{stripeOffset + streamOffset, streamSize}};
  }
  return locations;
}

std::shared_ptr<index::StreamIndex> StripeStreams::streamIndex(
    int streamId) const {
  NIMBLE_CHECK(stripeIdentifier_.has_value());

  const auto& chunkStats = stripeIdentifier_->chunkStats();
  if (chunkStats == nullptr) {
    return nullptr;
  }
  // A stream absent from this stripe group (e.g. added by later schema
  // evolution or a flat-map feature not present in this group) has no chunk
  // index. Gate on streamCount as streamRegion()/locateStreams() do, so an
  // out-of-range stream id does not reach streamSize() (which check-fails on
  // out-of-range).
  if (streamId >= readerBase_->tablet().streamCount(*stripeIdentifier_)) {
    return nullptr;
  }
  const uint32_t streamSize =
      readerBase_->tablet().streamSize(*stripeIdentifier_, streamId);
  return chunkStats->createStreamIndex(stripe_, streamId, streamSize);
}

} // namespace facebook::nimble
