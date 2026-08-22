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
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "folly/container/F14Map.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/nimble/common/Checksum.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/index/ClusterIndexFactory.h"
#include "velox/dwio/nimble/index/IndexSerialization.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FooterGenerated.h"
#include "velox/dwio/nimble/tablet/IndexGenerated.h"
#include "velox/dwio/nimble/tablet/SharedDictionaryReader.h"
#include "velox/dwio/nimble/tablet/StripeGroup.h"

#include "flatbuffers/flatbuffers.h"
#include "folly/io/Cursor.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

namespace facebook::nimble {

namespace {

// Tries to create a MetadataBuffer from the speculative read buffer if the
// metadata section fits within it. Returns nullptr if not covered.
std::unique_ptr<MetadataBuffer> tryExtractFromBuffer(
    std::string_view footerBuf,
    uint64_t footerOffset,
    const MetadataSection& section,
    velox::memory::MemoryPool* pool) {
  if (footerBuf.empty() || section.offset() < footerOffset) {
    return nullptr;
  }
  const uint64_t bufOffset = section.offset() - footerOffset;
  NIMBLE_CHECK_LE(
      bufOffset + section.size(),
      footerBuf.size(),
      "Metadata section exceeds speculative read buffer");
  return std::make_unique<MetadataBuffer>(MetadataBuffer::decompress(
      footerBuf.substr(bufOffset, section.size()),
      section.compressionType(),
      pool));
}

// Ensures the buffer has at least the requested capacity.
void ensureBuffer(
    velox::BufferPtr& buffer,
    uint64_t size,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK(
      buffer == nullptr || buffer->unique(), "Cannot reuse shared buffer");
  if (buffer == nullptr || buffer->size() < size) {
    buffer = velox::AlignedBuffer::allocateExact<char>(size, pool);
  }
}

} // namespace

TabletReader::Options TabletReader::configureOptions(
    const velox::dwio::common::ReaderOptions& options) {
  Options tabletOptions;
  tabletOptions.maxFooterIoBytes = options.footerSpeculativeIoSize();
  tabletOptions.preloadOptionalSections = {
      std::string(kSchemaSection), std::string(kVectorizedStatsSection)};
  tabletOptions.loadClusterIndex = options.loadClusterIndex();
  tabletOptions.preloadIndex = options.preloadIndex();
  tabletOptions.loadChunkStats = options.loadChunkStats();
  if (tabletOptions.loadChunkStats) {
    tabletOptions.preloadOptionalSections.emplace_back(kChunkStatsSection);
  }
  tabletOptions.pinMetadata = options.pinMetadata();
  tabletOptions.cacheMetadata = options.cacheMetadata();
  tabletOptions.pinIndex = options.pinIndex();
  tabletOptions.cacheIndex = options.cacheIndex();
  tabletOptions.fileHandle = options.fileHandle();
  tabletOptions.cache = options.cache();
  tabletOptions.ioOptions = options;
  if (const auto& formatOptions = options.formatSpecificOptions()) {
    if (auto nimbleOptions =
            std::dynamic_pointer_cast<const NimbleReaderOptions>(
                formatOptions)) {
      tabletOptions.externalDictionaryResolver =
          nimbleOptions->externalDictionaryResolver;
    }
  }

  // TODO(T272495998): Temporary shim. Remove once HiveDataSource plumbs a
  // real `metadataIoStats` through `ReaderOptions` that the operator /
  // connector exposes for observability. Until then, synthesize a default
  // when the caller does not supply one, so that callers reaching a reader
  // via Velox's HiveDataSource (which builds ReaderOptions without setting IO
  // stats) continue to work rather than tripping the constructor's check.
  // Only the sliced copy is touched, so the caller's options are unchanged.
  if (tabletOptions.ioOptions->metadataIoStats() == nullptr) {
    tabletOptions.ioOptions->setMetadataIoStats(
        std::make_shared<velox::io::IoStatistics>());
  }
  return tabletOptions;
}

// Here's the layout of the tablet:
//
// stripe 1 streams
// stripe 2 streams
// ...
// stripe k streams
// footer
//
// where the footer is a flatbuffer payload, as described here:
// velox/dwio/nimble/tablet/footer.fbs
// followed by fixed payload:
// 4 bytes footer size + 1 byte footer compression type +
// 1 byte checksum type + 8 bytes checksum +
// 2 bytes major version + 2 bytes minor version +
// 2 bytes magic number.
namespace {

template <typename T>
const T* asFlatBuffersRoot(std::string_view content) {
  return flatbuffers::GetRoot<T>(content.data());
}

inline const serialization::Footer* footerRoot(const MetadataBuffer& footer) {
  return asFlatBuffersRoot<serialization::Footer>(footer.content());
}

inline const serialization::Stripes* stripesRoot(
    const MetadataBuffer& stripes) {
  return asFlatBuffersRoot<serialization::Stripes>(stripes.content());
}

inline void validateOptionalSections(
    const serialization::OptionalMetadataSections* sections) {
  NIMBLE_CHECK_NOT_NULL(sections, "Optional sections is null.");
  NIMBLE_CHECK(sections->names(), "Optional sections names is null.");
  NIMBLE_CHECK(sections->offsets(), "Optional sections offsets is null.");
  NIMBLE_CHECK(sections->sizes(), "Optional sections sizes is null.");
  NIMBLE_CHECK(
      sections->compression_types(),
      "Optional sections compression_types is null.");
  NIMBLE_CHECK_EQ(
      sections->names()->size(),
      sections->offsets()->size(),
      "Optional sections names and offsets size mismatch.");
  NIMBLE_CHECK_EQ(
      sections->names()->size(),
      sections->sizes()->size(),
      "Optional sections names and sizes size mismatch.");
  NIMBLE_CHECK_EQ(
      sections->names()->size(),
      sections->compression_types()->size(),
      "Optional sections names and compression_types size mismatch.");
}

// Converts a FlatBuffers section (with offset/size/compression_type fields)
// to a MetadataSection.
template <typename T>
MetadataSection toMetadataSection(const T* section) {
  const auto uncompressedSize = section->uncompressed_size();
  return MetadataSection{
      section->offset(),
      section->size(),
      static_cast<CompressionType>(section->compression_type()),
      uncompressedSize > 0 ? std::optional<uint32_t>(uncompressedSize)
                           : std::nullopt};
}

size_t copyTo(const folly::IOBuf& source, void* target, size_t size) {
  NIMBLE_DCHECK_LE(
      source.computeChainDataLength(), size, "Target buffer too small.");
  size_t offset = 0;
  for (const auto& chunk : source) {
    std::copy(chunk.begin(), chunk.end(), static_cast<char*>(target) + offset);
    offset += chunk.size();
  }

  return offset;
}

} // namespace

TabletReader::TabletReader(
    std::shared_ptr<velox::ReadFile> readFile,
    MemoryPool& pool,
    const Options& options)
    : pool_{&pool},
      file_{std::move(readFile)},
      loadClusterIndex_{options.loadClusterIndex},
      clusterIndexName_{options.clusterIndexName},
      loadChunkStats_{options.loadChunkStats},
      loadDenseIndexes_{options.loadDenseIndexes},
      ioOptions_{[&]() {
        NIMBLE_CHECK(options.ioOptions.has_value(), "ioOptions is not set.");
        NIMBLE_CHECK_NOT_NULL(
            options.ioOptions->metadataIoStats(), "metadataIoStats is null.");
        return *options.ioOptions;
      }()},
      indexOptions_{[&]() {
        // cacheIndex takes effect only when fileHandle and cache are provided.
        const bool cacheEnabled =
            options.cacheIndex && options.fileHandle != nullptr;
        if (cacheEnabled) {
          NIMBLE_CHECK_NOT_NULL(options.cache, "cacheIndex requires cache");
        }
        return index::IndexLookup::Options{
            .file = file_,
            .ioOptions = &ioOptions_,
            .fileHandle = cacheEnabled ? options.fileHandle : nullptr,
            .cache = cacheEnabled ? options.cache : nullptr,
            // preloadIndex requires pinIndex; auto-promote so the caller only
            // needs to flip one knob.
            .pinIndex = options.pinIndex || options.preloadIndex,
            .preloadIndex = options.preloadIndex};
      }()},
      metadataInput_{[&]() {
        MetadataInput::Options metadataOptions{
            .pool = &pool,
            .ioStats = ioOptions_.metadataIoStats(),
            .maxCoalesceDistance = ioOptions_.maxCoalesceDistance(),
            .maxCoalesceBytes = ioOptions_.maxCoalesceBytes(),
            .executor = ioOptions_.ioExecutor().get()};
        // cacheMetadata takes effect only when fileHandle and cache are
        // provided.
        if (options.cacheMetadata && options.fileHandle != nullptr) {
          NIMBLE_CHECK_NOT_NULL(options.cache, "cacheMetadata requires cache");
          metadataOptions.maxCacheEntrySize = options.maxCacheEntrySize;
          metadataOptions.fileHandle = options.fileHandle;
          metadataOptions.cache = options.cache;
        }
        return MetadataInput::create(file_.get(), metadataOptions);
      }()},
      stripeGroupCache_{
          [this](uint32_t stripeGroupIndex) {
            return loadStripeGroup(stripeGroupIndex);
          },
          options.pinMetadata},
      chunkStatsCache_{
          [this](uint32_t stripeGroupIndex) {
            return loadChunkStatsGroup(stripeGroupIndex);
          },
          options.pinMetadata} {
  NIMBLE_CHECK_EQ(
      static_cast<const void*>(&ioOptions_.memoryPool()),
      static_cast<const void*>(pool_),
      "ioOptions pool must match the provided pool");
  init(options);
}

TabletReader::~TabletReader() = default;

bool TabletReader::hasFileOrExternalDictionaries() const {
  return sharedDictionaryReaderFactory_ != nullptr &&
      sharedDictionaryReaderFactory_->hasFileOrExternalDictionaries();
}

bool TabletReader::hasStripeDictionaries() const {
  return sharedDictionaryReaderFactory_ != nullptr &&
      sharedDictionaryReaderFactory_->hasStripeDictionaries();
}

std::optional<uint32_t> TabletReader::stripeDictionaryStreamId(
    uint32_t valueStreamId) const {
  return sharedDictionaryReaderFactory_ == nullptr
      ? std::nullopt
      : sharedDictionaryReaderFactory_->dictionaryStreamId(valueStreamId);
}

folly::F14FastMap<uint32_t, uint32_t> TabletReader::stripeDictionaryStreamIds(
    std::span<const uint32_t> valueStreamIds) const {
  return sharedDictionaryReaderFactory_ == nullptr
      ? folly::F14FastMap<uint32_t, uint32_t>{}
      : sharedDictionaryReaderFactory_->dictionaryStreamIds(valueStreamIds);
}

std::shared_ptr<const SharedDictionaryAlphabet>
TabletReader::resolveDictionaryAlphabet(uint32_t valueStreamId) const {
  return sharedDictionaryReaderFactory_ == nullptr
      ? nullptr
      : sharedDictionaryReaderFactory_->resolveAlphabet(valueStreamId);
}

void TabletReader::init(const Options& options) {
  fileSize_ = file_->size();
  NIMBLE_CHECK_FILE(
      fileSize_ >= Postscript::kSize,
      "Corrupted file. File size {} is smaller than postscript size {}.",
      velox::succinctBytes(fileSize_),
      velox::succinctBytes(Postscript::kSize));

  if (initFromCache(options)) {
    stats_.footerCacheHit = true;
    return;
  }

  uint64_t footerIoSize{0};
  uint64_t footerOffset{0};
  velox::BufferPtr footerBuf;

  loadFooter(options.maxFooterIoBytes, footerBuf, footerIoSize, footerOffset);
  NIMBLE_CHECK_NOT_NULL(footer_);

  const auto footerView = footerBuf != nullptr
      ? std::string_view{footerBuf->as<char>(), footerBuf->size()}
      : std::string_view{};

  // Parse optional sections metadata from footer before enqueueing.
  initOptionalSections();

  // Collect stripes + optional sections. Try extracting from footerBuf
  // first (speculative mode), batch-load the rest via coalesced IO.
  std::vector<LoadSection> sections;
  collectStripesSection(footerView, footerOffset, sections);
  collectOptionalSections(options, footerView, footerOffset, sections);

  loadSections(sections);

  initStripes(footerView, footerOffset);
  initFeatures();
  initSharedDictionaries(options);
  initIndexDescriptors();
  initClusterIndex();
  initChunkStats(footerView, footerOffset);
  initDenseIndexes();

  cacheMetadata(footerView, footerOffset);
}

void TabletReader::loadFooter(
    uint64_t maxFooterIoBytes,
    velox::BufferPtr& footerBuf,
    uint64_t& footerIoSize,
    uint64_t& footerOffset) {
  // Postscript and footer are not cached individually — they are cached
  // together at the synthetic key fileSize_ by cacheMetadata().
  if (maxFooterIoBytes > 0) {
    // Speculative mode: one big read covers PS + footer + more.
    // Extract PS and footer from the buffer — no separate reads.
    NIMBLE_CHECK_GE(
        maxFooterIoBytes,
        Postscript::kSize,
        "maxFooterIoBytes must be at least Postscript::kSize");
    footerIoSize = std::min(maxFooterIoBytes, fileSize_);
    footerOffset = fileSize_ - footerIoSize;
    ensureBuffer(footerBuf, footerIoSize, pool_);
    metadataInput_->readRaw(
        footerOffset, footerBuf->size(), footerBuf->asMutable<char>());

    ps_ = Postscript::parse(
        std::string_view{
            footerBuf->as<char>() + footerIoSize - Postscript::kSize,
            Postscript::kSize});

    // If the speculative read didn't cover the full footer, re-read.
    const uint64_t requiredSize = ps_.footerSize() + Postscript::kSize;
    if (requiredSize > footerIoSize) {
      stats_.footerBufferUnderread =
          static_cast<int64_t>(requiredSize - footerIoSize);
      footerIoSize = requiredSize;
      footerOffset = fileSize_ - footerIoSize;
      ensureBuffer(footerBuf, footerIoSize, pool_);
      metadataInput_->readRaw(
          footerOffset, footerBuf->size(), footerBuf->asMutable<char>());
    } else {
      stats_.footerBufferOverread =
          static_cast<int64_t>(footerIoSize - requiredSize);
    }

    const uint64_t footerBufOffset =
        footerIoSize - Postscript::kSize - ps_.footerSize();
    footer_ = std::make_unique<MetadataBuffer>(MetadataBuffer::decompress(
        std::string_view{
            footerBuf->as<char>() + footerBufOffset, ps_.footerSize()},
        ps_.footerCompressionType(),
        pool_));
  } else {
    // Adaptive mode: read PS first, then footer.
    ensureBuffer(footerBuf, Postscript::kSize, pool_);
    metadataInput_->readRaw(
        fileSize_ - Postscript::kSize,
        Postscript::kSize,
        footerBuf->asMutable<char>());
    ps_ = Postscript::parse(
        std::string_view{footerBuf->as<char>(), Postscript::kSize});

    footerIoSize = ps_.footerSize() + Postscript::kSize;
    footerOffset = fileSize_ - footerIoSize;

    ensureBuffer(footerBuf, ps_.footerSize(), pool_);
    metadataInput_->readRaw(
        footerOffset, ps_.footerSize(), footerBuf->asMutable<char>());
    footer_ = std::make_unique<MetadataBuffer>(MetadataBuffer::decompress(
        std::move(footerBuf), ps_.footerCompressionType(), pool_));
  }
}

bool TabletReader::loadFooterFromCache() {
  if (!metadataInput_->cached()) {
    return false;
  }
  auto cachedFooter = metadataInput_->findCachedMetadata(fileSize_);
  if (cachedFooter == nullptr) {
    return false;
  }

  const auto content = cachedFooter->content();
  NIMBLE_CHECK_GE(
      content.size(), Postscript::kSize, "Cached footer+PS entry too small");

  ps_ = Postscript::parse(
      content.substr(content.size() - Postscript::kSize, Postscript::kSize));

  const auto footerSize = content.size() - Postscript::kSize;
  if (ps_.footerCompressionType() == CompressionType::Uncompressed) {
    NIMBLE_CHECK_EQ(
        ps_.footerSize(),
        footerSize,
        "Cached footer size mismatch with postscript");
  } else {
    NIMBLE_CHECK_LE(
        ps_.footerSize(),
        footerSize,
        "Cached compressed footer size exceeds decompressed size");
  }

  footer_ = cachedFooter->slice(0, footerSize);
  return true;
}

bool TabletReader::initFromCache(const Options& options) {
  if (!loadFooterFromCache()) {
    return false;
  }
  initOptionalSections();

  // Collect stripes + optional sections together for coalesced IO.
  // Empty files (0 rows) have no stripes section but still need optional
  // sections loaded (e.g., schema describes column types even with no rows).
  std::vector<LoadSection> sections;
  collectStripesSection(sections);
  collectOptionalSections(options, sections);

  loadSections(sections);

  initStripes();
  initFeatures();
  initSharedDictionaries(options);
  initIndexDescriptors();
  initClusterIndex();
  initChunkStats();
  initDenseIndexes();
  return true;
}

void TabletReader::cacheMetadata(
    std::string_view footerBuf,
    uint64_t footerOffset) {
  if (!metadataInput_->cached()) {
    return;
  }
  if (footerBuf.empty()) {
    return;
  }

  auto cacheSection = [&](const MetadataSection& section) {
    if (section.offset() < footerOffset) {
      return;
    }
    const uint64_t bufOffset = section.offset() - footerOffset;
    if (bufOffset + section.size() > footerBuf.size()) {
      return;
    }
    std::string_view range = footerBuf.substr(bufOffset, section.size());
    if (section.compressionType() == CompressionType::Uncompressed) {
      metadataInput_->cacheMetadata(section.offset(), {&range, 1});
    } else {
      auto decompressed =
          MetadataBuffer::decompress(range, section.compressionType(), pool_);
      std::string_view view{decompressed->as<char>(), decompressed->size()};
      metadataInput_->cacheMetadata(section.offset(), {&view, 1});
    }
  };

  // Cache decompressed footer + PS at synthetic offset fileSize_.
  // Store uncompressed so loadFooterFromCache can slice without decompressing.
  {
    const auto footerContent = footer_->content();
    const auto psData = ps_.serialize();
    std::array<std::string_view, 2> ranges{footerContent, psData};
    metadataInput_->cacheMetadata(fileSize_, ranges);
  }

  const auto* footer = footerRoot(*footer_);

  if (auto* stripesSection = footer->stripes()) {
    cacheSection(toMetadataSection(stripesSection));
  }

  if (auto* stripeGroups = footer->stripe_groups();
      stripeGroups != nullptr && stripeGroups->size() > 0) {
    cacheSection(toMetadataSection(stripeGroups->Get(0)));
  }

  if (auto featuresIt = optionalSections_.find(std::string{kFeaturesSection});
      featuresIt != optionalSections_.end()) {
    cacheSection(featuresIt->second);
  }

  if (sharedDictionaryReaderFactory_ != nullptr) {
    auto sharedDictionaryIt =
        optionalSections_.find(std::string{kDictionarySection});
    NIMBLE_CHECK(sharedDictionaryIt != optionalSections_.end());
    cacheSection(sharedDictionaryIt->second);
  }

  if (clusterIndex_ != nullptr || denseIndexRegistry_ != nullptr) {
    auto indexIt = optionalSections_.find(std::string{kIndexSection});
    NIMBLE_CHECK(indexIt != optionalSections_.end());
    cacheSection(indexIt->second);
  }

  if (chunkStats_ != nullptr) {
    auto chunkStatsIt = optionalSections_.find(std::string{kChunkStatsSection});
    NIMBLE_CHECK(chunkStatsIt != optionalSections_.end());
    cacheSection(chunkStatsIt->second);

    const auto& firstSection = chunkStats_->groupMetadata(0);
    if (firstSection.size() > 0) {
      cacheSection(firstSection);
    }
  }
}

void TabletReader::loadStripes(
    velox::BufferPtr& footerBuf,
    uint64_t& footerIoSize,
    uint64_t& footerOffset,
    const Options& options) {
  NIMBLE_CHECK_NOT_NULL(footer_);
  NIMBLE_CHECK_NULL(stripes_);

  const auto* footer = footerRoot(*footer_);
  auto* stripesSection = footer->stripes();
  if (stripesSection == nullptr) {
    NIMBLE_CHECK_EQ(footer->row_count(), 0);
    return;
  }

  // Compute the lowest offset we need in the buffer.
  uint64_t requiredOffset = stripesSection->offset();
  const auto updateOffset = [&](std::string_view sectionName) {
    auto it = optionalSections_.find(std::string{sectionName});
    if (it != optionalSections_.end()) {
      requiredOffset = std::min(requiredOffset, it->second.offset());
    }
  };
  if (options.loadClusterIndex || options.loadDenseIndexes) {
    updateOffset(kIndexSection);
  }
  if (options.loadChunkStats) {
    updateOffset(kChunkStatsSection);
  }
  const uint64_t requiredSize = fileSize_ - requiredOffset;
  if (requiredSize > footerIoSize) {
    footerIoSize = requiredSize;
    footerOffset = fileSize_ - footerIoSize;
    ensureBuffer(footerBuf, footerIoSize, pool_);
    metadataInput_->readRaw(
        footerOffset, footerBuf->size(), footerBuf->asMutable<char>());
  }
  const auto footerView = footerBuf != nullptr
      ? std::string_view{footerBuf->as<char>(), footerBuf->size()}
      : std::string_view{};
  const auto stripesMetadataSection = toMetadataSection(stripesSection);
  auto fromBuf = tryExtractFromBuffer(
      footerView, footerOffset, stripesMetadataSection, pool_);
  if (fromBuf != nullptr) {
    stripes_ = std::move(fromBuf);
  } else {
    stripes_ = readMetadata(stripesMetadataSection);
  }
}

void TabletReader::collectStripesSection(
    std::string_view footerBuf,
    uint64_t footerOffset,
    std::vector<LoadSection>& loadSections) {
  const auto* footer = footerRoot(*footer_);
  auto* stripesSection = footer->stripes();
  if (stripesSection == nullptr) {
    return;
  }
  const auto section = toMetadataSection(stripesSection);
  auto fromBuf = tryExtractFromBuffer(footerBuf, footerOffset, section, pool_);
  if (fromBuf != nullptr) {
    stripes_ = std::move(fromBuf);
  } else {
    loadSections.emplace_back(
        LoadSection{LoadSection::Type::kStripes, {}, section});
  }
}

void TabletReader::collectOptionalSections(
    const Options& options,
    std::string_view footerBuf,
    uint64_t footerOffset,
    std::vector<LoadSection>& loadSections) {
  auto optionalSectionCache = optionalSectionsCache_.wlock();
  for (const auto& name : preloadSectionNames(options)) {
    auto it = optionalSections_.find(name);
    if (it == optionalSections_.end()) {
      continue;
    }
    auto fromBuf =
        tryExtractFromBuffer(footerBuf, footerOffset, it->second, pool_);
    if (fromBuf != nullptr) {
      optionalSectionCache->insert({name, std::move(fromBuf)});
    } else {
      loadSections.emplace_back(
          LoadSection{LoadSection::Type::kOptionalSection, name, it->second});
    }
  }
}

void TabletReader::loadSections(std::vector<LoadSection>& loadSections) {
  if (loadSections.empty()) {
    return;
  }
  std::vector<MetadataSection> sections;
  sections.reserve(loadSections.size());
  for (const auto& p : loadSections) {
    sections.emplace_back(p.section);
  }
  auto results = metadataInput_->load(sections);
  auto optionalSectionCache = optionalSectionsCache_.wlock();
  for (size_t i = 0; i < loadSections.size(); ++i) {
    auto buffer = std::make_unique<MetadataBuffer>(std::move(*results[i]));
    if (loadSections[i].type == LoadSection::Type::kStripes) {
      NIMBLE_CHECK_NULL(stripes_, "Stripes already loaded");
      stripes_ = std::move(buffer);
    } else {
      auto [_, inserted] = optionalSectionCache->insert(
          {std::move(loadSections[i].name), std::move(buffer)});
      NIMBLE_CHECK(inserted, "Optional section already loaded");
    }
  }
}

void TabletReader::initStripes(
    std::string_view footerBuf,
    uint64_t footerOffset) {
  const auto* footer = footerRoot(*footer_);
  NIMBLE_CHECK_EQ(tabletRowCount_, 0);
  tabletRowCount_ = footer->row_count();
  if (stripes_ == nullptr) {
    NIMBLE_CHECK_EQ(tabletRowCount_, 0);
    return;
  }

  const auto* stripes = stripesRoot(*stripes_);

  stripeCount_ = stripes->row_counts()->size();
  NIMBLE_CHECK_GT(stripeCount_, 0, "Unexpected stripe count");
  NIMBLE_CHECK_EQ(
      stripeCount_, stripes->offsets()->size(), "Unexpected stripe count");
  NIMBLE_CHECK_EQ(
      stripeCount_, stripes->sizes()->size(), "Unexpected stripe count");
  NIMBLE_CHECK_EQ(
      stripeCount_,
      stripes->group_indices()->size(),
      "Unexpected stripe count");
  stripeOffsets_ = stripes->offsets()->data();

  // Build prefix sum for O(log n) rowToStripe lookup.
  const auto* rowCounts = stripes->row_counts()->data();
  stripeRows_.resize(stripeCount_ + 1);
  stripeRows_[0] = 0;
  for (uint32_t i = 0; i < stripeCount_; ++i) {
    stripeRows_[i + 1] = stripeRows_[i] + rowCounts[i];
  }

  // Eagerly pin the first stripe group if covered by the speculative buffer.
  const auto* stripeGroups = footer->stripe_groups();
  NIMBLE_CHECK_NOT_NULL(stripeGroups, "Stripe groups is null.");
  NIMBLE_CHECK_EQ(
      stripeGroups->size(),
      *stripes->group_indices()->rbegin() + 1,
      "Unexpected stripe group count");
  if (stripeGroups->size() == 1) {
    auto metadataBuffer = tryExtractFromBuffer(
        footerBuf,
        footerOffset,
        toMetadataSection(stripeGroups->Get(0)),
        pool_);
    if (metadataBuffer != nullptr) {
      stripeGroupCache_.pin(
          0,
          std::make_shared<StripeGroup>(
              0, *stripes_, std::move(metadataBuffer), pool_));
    }
  }
}

void TabletReader::initOptionalSections() {
  const auto* footer = footerRoot(*footer_);

  auto* optionalSections = footer->optional_sections();
  if (optionalSections == nullptr) {
    return;
  }

  validateOptionalSections(optionalSections);

  optionalSections_.reserve(optionalSections->names()->size());
  const auto* uncompressedSizes = optionalSections->uncompressed_sizes();
  for (auto i = 0; i < optionalSections->names()->size(); ++i) {
    optionalSections_.insert(
        std::make_pair(
            optionalSections->names()->GetAsString(i)->str(),
            MetadataSection{
                optionalSections->offsets()->Get(i),
                optionalSections->sizes()->Get(i),
                static_cast<CompressionType>(
                    optionalSections->compression_types()->Get(i)),
                uncompressedSizes != nullptr
                    ? std::optional<uint32_t>(uncompressedSizes->Get(i))
                    : std::nullopt}));
  }
}

void TabletReader::initFeatures() {
  auto section =
      loadOptionalSection(std::string{kFeaturesSection}, /*keepCache=*/true);
  if (!section.has_value()) {
    return;
  }

  features_ = FileFeatures::deserialize(section->content());
}

void TabletReader::initSharedDictionaries(const Options& options) {
  auto section =
      loadOptionalSection(std::string{kDictionarySection}, /*keepCache=*/false);
  if (!section.has_value()) {
    return;
  }

  sharedDictionaryReaderFactory_ = SharedDictionaryReaderFactory::create(
      section->content(),
      options.externalDictionaryResolver,
      /*tabletReader=*/this,
      /*pool=*/pool_);
}

std::vector<std::string> TabletReader::preloadSectionNames(
    const Options& options) const {
  std::vector<std::string> names;
  names.reserve(options.preloadOptionalSections.size() + 3);
  auto addName = [&names](std::string name) {
    if (std::find(names.begin(), names.end(), name) == names.end()) {
      names.emplace_back(std::move(name));
    }
  };
  for (const auto& name : options.preloadOptionalSections) {
    addName(name);
  }
  if (options.loadClusterIndex || options.loadDenseIndexes) {
    addName(std::string{kIndexSection});
  }
  addName(std::string{kFeaturesSection});
  addName(std::string{kDictionarySection});
  return names;
}

std::shared_ptr<TabletReader> TabletReader::create(
    std::shared_ptr<velox::ReadFile> readFile,
    MemoryPool* pool,
    const Options& options) {
  NIMBLE_CHECK_NOT_NULL(pool);
  return std::shared_ptr<TabletReader>(
      new TabletReader(std::move(readFile), *pool, options));
}

uint64_t TabletReader::calculateChecksum(
    velox::memory::MemoryPool& pool,
    velox::ReadFile* readFile,
    uint64_t chunkSize) {
  const auto postscriptStart = readFile->size() - Postscript::kSize;
  Vector<char> postscript(&pool, Postscript::kSize);
  readFile->pread(postscriptStart, Postscript::kSize, postscript.data());
  const ChecksumType checksumType = *reinterpret_cast<ChecksumType*>(
      postscript.data() + kPostscriptChecksumedSize);

  const auto checksum = ChecksumFactory::create(checksumType);
  Vector<char> buffer(&pool);
  uint64_t sizeToRead =
      readFile->size() - Postscript::kSize + kPostscriptChecksumedSize;
  uint64_t offset{0};
  while (sizeToRead > 0) {
    const auto sizeOneRead = std::min(chunkSize, sizeToRead);
    buffer.resize(sizeOneRead);
    const std::string_view bufferRead =
        readFile->pread(offset, sizeOneRead, buffer.data());
    checksum->update(bufferRead);
    sizeToRead -= sizeOneRead;
    offset += sizeOneRead;
  }
  return checksum->getChecksum();
}

namespace {
// LoadTask describes the task of PReading a region and splitting it into the
// streams within (described by StreamTask).

struct StreamTask {
  // The stream index from the input indices
  uint32_t index;

  // Byte offset for the stream, relative to the beginning of the stripe
  uint32_t offset;

  // Size of the stream
  uint32_t length;
};

struct LoadTask {
  // Relative to first byte in file.
  uint64_t readStart;
  uint32_t readLength;
  std::vector<StreamTask> streamTasks;
};

class PreloadedStreamLoader : public StreamLoader {
 public:
  explicit PreloadedStreamLoader(Vector<char>&& stream)
      : stream_{std::move(stream)} {}

  const std::string_view getStream() const override {
    return {stream_.data(), stream_.size()};
  }

 private:
  const Vector<char> stream_;
};

struct RegionHash {
  size_t operator()(const velox::common::Region& region) const {
    return folly::hash::hash_combine(
        std::hash<size_t>{}(region.offset), std::hash<size_t>{}(region.length));
  }
};

struct RegionEqual {
  bool operator()(
      const velox::common::Region& lhs,
      const velox::common::Region& rhs) const {
    return lhs.offset == rhs.offset && lhs.length == rhs.length;
  }
};

} // namespace

uint32_t TabletReader::stripeGroupIndex(uint32_t stripeIndex) const {
  return stripesRoot(*stripes_)->group_indices()->Get(stripeIndex);
}

std::unique_ptr<MetadataBuffer> TabletReader::readMetadata(
    const MetadataSection& section) const {
  NIMBLE_CHECK_GT(section.size(), 0, "Metadata section size must be non-zero");
  auto results = metadataInput_->load({&section, 1});
  NIMBLE_CHECK_EQ(results.size(), 1);
  return std::make_unique<MetadataBuffer>(std::move(*results[0]));
}

std::shared_ptr<StripeGroup> TabletReader::loadStripeGroup(
    uint32_t stripeGroupIndex) const {
  const auto* footer = footerRoot(*footer_);
  auto stripeGroupInfo = footer->stripe_groups()->Get(stripeGroupIndex);

  const auto section = toMetadataSection(stripeGroupInfo);
  return std::make_shared<StripeGroup>(
      stripeGroupIndex, *stripes_, readMetadata(section), pool_);
}

std::shared_ptr<StripeGroup> TabletReader::stripeGroup(
    uint32_t stripeGroupIndex) const {
  return stripeGroupCache_.getOrCreate(stripeGroupIndex);
}

TabletReader::StripeGroupMetadata TabletReader::loadStripeGroupMetadata(
    uint32_t stripeGroupIndex) const {
  const auto* footer = footerRoot(*footer_);
  const auto groupSection =
      toMetadataSection(footer->stripe_groups()->Get(stripeGroupIndex));

  const bool hasChunkStats = this->hasChunkStats(stripeGroupIndex);

  std::vector<MetadataSection> sections;
  sections.emplace_back(groupSection);
  if (hasChunkStats) {
    sections.emplace_back(chunkStats_->groupMetadata(stripeGroupIndex));
  }

  auto results = metadataInput_->load(sections);

  auto groupMetadata = std::make_unique<MetadataBuffer>(std::move(*results[0]));

  StripeGroupMetadata result;
  result.stripeGroup = std::make_shared<StripeGroup>(
      stripeGroupIndex, *stripes_, std::move(groupMetadata), pool_);
  if (hasChunkStats) {
    auto chunkStatsMetadata =
        std::make_unique<MetadataBuffer>(std::move(*results[1]));
    result.chunkStats = ChunkStatsGroup::create(
        result.stripeGroup->firstStripe(),
        result.stripeGroup->stripeCount(),
        std::move(chunkStatsMetadata));
  }
  return result;
}

uint32_t TabletReader::streamOffset(
    const StripeIdentifier& stripe,
    uint32_t streamId) const {
  NIMBLE_DCHECK_LT(stripe.stripeId(), stripeCount_, "Stripe is out of range.");
  return stripe.stripeGroup()->streamOffset(stripe.stripeId(), streamId);
}

uint32_t TabletReader::streamSize(
    const StripeIdentifier& stripe,
    uint32_t streamId) const {
  NIMBLE_DCHECK_LT(stripe.stripeId(), stripeCount_, "Stripe is out of range.");
  // Mirrors streamOffset: StripeGroup::streamSize already returns 0 for an
  // out-of-range streamId, so delegate without a redundant guard here.
  return stripe.stripeGroup()->streamSize(stripe.stripeId(), streamId);
}

void TabletReader::streamOffsets(
    const StripeIdentifier& stripe,
    std::span<uint32_t> out) const {
  NIMBLE_DCHECK_LT(stripe.stripeId(), stripeCount_, "Stripe is out of range.");
  stripe.stripeGroup()->streamOffsets(stripe.stripeId(), out);
}

void TabletReader::streamSizes(
    const StripeIdentifier& stripe,
    std::span<uint32_t> out) const {
  NIMBLE_DCHECK_LT(stripe.stripeId(), stripeCount_, "Stripe is out of range.");
  stripe.stripeGroup()->streamSizes(stripe.stripeId(), out);
}

uint32_t TabletReader::streamCount(const StripeIdentifier& stripe) const {
  NIMBLE_DCHECK_LT(stripe.stripeId(), stripeCount_, "Stripe is out of range.");
  return stripe.stripeGroup()->streamCount();
}

StripeIdentifier TabletReader::stripeIdentifier(uint32_t stripeIndex) const {
  NIMBLE_CHECK_LT(stripeIndex, stripeCount_, "Stripe is out of range.");
  const auto stripeGroupIndex = this->stripeGroupIndex(stripeIndex);

  const bool hasChunkStats = this->hasChunkStats(stripeGroupIndex);

  // Check which metadata is already cached.
  const bool stripeGroupCached =
      stripeGroupCache_.hasCacheEntry(stripeGroupIndex);
  const bool chunkStatsCached =
      hasChunkStats && chunkStatsCache_.hasCacheEntry(stripeGroupIndex);

  // If all needed metadata is cached, load each individually (cache hits).
  const bool allCached =
      stripeGroupCached && (!hasChunkStats || chunkStatsCached);
  if (allCached) {
    return StripeIdentifier{
        stripeIndex,
        stripeGroup(stripeGroupIndex),
        hasChunkStats ? chunkStats(stripeGroupIndex) : nullptr};
  }

  // At least one is not cached — use coalesced loading.
  auto loaded = loadStripeGroupMetadata(stripeGroupIndex);

  // Inject into caches using custom builder that returns the already-loaded
  // objects. This ensures the cache holds the reference.
  auto cachedStripeGroup = stripeGroupCache_.getOrCreate(
      stripeGroupIndex,
      [&loaded](uint32_t) { return std::move(loaded.stripeGroup); });

  std::shared_ptr<ChunkStatsGroup> cachedChunkStats;
  if (hasChunkStats) {
    cachedChunkStats = chunkStatsCache_.getOrCreate(
        stripeGroupIndex,
        [&loaded](uint32_t) { return std::move(loaded.chunkStats); });
  }

  return StripeIdentifier{
      stripeIndex, std::move(cachedStripeGroup), std::move(cachedChunkStats)};
}

std::vector<std::unique_ptr<StreamLoader>> TabletReader::load(
    const StripeIdentifier& stripe,
    std::span<const uint32_t> streamIdentifiers,
    std::function<std::string_view(uint32_t)> streamLabel) const {
  NIMBLE_CHECK_LT(stripe.stripeId(), stripeCount_, "Stripe is out of range.");

  const uint64_t stripeOffset = this->stripeOffset(stripe.stripeId());
  const auto& stripeGroup = stripe.stripeGroup();
  const uint32_t stripeId = stripe.stripeId();
  const uint32_t streamsToLoad = streamIdentifiers.size();

  std::vector<std::unique_ptr<StreamLoader>> streams(streamsToLoad);
  folly::F14FastMap<
      velox::common::Region,
      std::vector<uint32_t>,
      RegionHash,
      RegionEqual>
      regionToStreamIndices;
  std::vector<velox::common::Region> uniqueRegions;
  regionToStreamIndices.reserve(streamsToLoad);
  uniqueRegions.reserve(streamsToLoad);

  for (uint32_t i = 0; i < streamsToLoad; ++i) {
    const uint32_t streamIdentifier = streamIdentifiers[i];
    if (streamIdentifier >= stripeGroup->streamCount()) {
      streams[i] = nullptr;
      continue;
    }

    const uint32_t streamSize =
        stripeGroup->streamSize(stripeId, streamIdentifier);
    if (streamSize == 0) {
      streams[i] = nullptr;
      continue;
    }

    const auto streamStart =
        stripeOffset + stripeGroup->streamOffset(stripeId, streamIdentifier);

    auto it = regionToStreamIndices.emplace(
        velox::common::Region{streamStart, streamSize},
        std::vector<uint32_t>{});
    if (it.second) {
      uniqueRegions.emplace_back(
          streamStart, streamSize, streamLabel(streamIdentifier));
    }
    it.first->second.push_back(i);
  }
  if (!uniqueRegions.empty()) {
    std::vector<folly::IOBuf> iobufs(uniqueRegions.size());
    file_->preadv(uniqueRegions, {iobufs.data(), iobufs.size()});
    NIMBLE_DCHECK_EQ(
        iobufs.size(), uniqueRegions.size(), "Buffer size mismatch.");
    for (uint32_t i = 0; i < uniqueRegions.size(); ++i) {
      // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
      const auto size = iobufs[i].computeChainDataLength();
      const auto& streamIndices = regionToStreamIndices[uniqueRegions[i]];
      for (uint32_t streamIndex : streamIndices) {
        Vector<char> vector{pool_, size};
        copyTo(iobufs[i], vector.data(), vector.size());
        streams[streamIndex] =
            std::make_unique<PreloadedStreamLoader>(std::move(vector));
      }
    }
  }

  return streams;
}

uint64_t TabletReader::totalStreamSize(
    const StripeIdentifier& stripe,
    std::span<const uint32_t> streamIdentifiers) const {
  NIMBLE_CHECK_LT(stripe.stripeId(), stripeCount_, "Stripe is out of range.");
  const auto& stripeGroup = stripe.stripeGroup();
  const uint32_t stripeId = stripe.stripeId();
  uint64_t streamSizeSum{0};
  for (const auto streamId : streamIdentifiers) {
    if (streamId >= stripeGroup->streamCount()) {
      continue;
    }
    streamSizeSum += stripeGroup->streamSize(stripeId, streamId);
  }
  return streamSizeSum;
}

std::optional<MetadataSection> TabletReader::stripesMetadata() const {
  const auto* footer = footerRoot(*footer_);
  auto* stripes = footer->stripes();
  if (stripes == nullptr) {
    return std::nullopt;
  }
  return toMetadataSection(stripes);
}

std::vector<MetadataSection> TabletReader::stripeGroupsMetadata() const {
  std::vector<MetadataSection> groupsMetadata;
  const auto* footer = footerRoot(*footer_);
  auto* stripeGroups = footer->stripe_groups();
  if (stripeGroups == nullptr) {
    return groupsMetadata;
  }
  groupsMetadata.reserve(stripeGroups->size());
  std::transform(
      stripeGroups->cbegin(),
      stripeGroups->cend(),
      std::back_inserter(groupsMetadata),
      [](const auto& stripeGroup) { return toMetadataSection(stripeGroup); });
  return groupsMetadata;
}

bool TabletReader::hasOptionalSection(const std::string& name) const {
  const auto it = optionalSections_.find(name);
  return it != optionalSections_.end();
}

std::optional<Section> TabletReader::loadOptionalSection(
    const std::string& name,
    bool keepCache) const {
  NIMBLE_CHECK(!name.empty(), "Optional section name cannot be empty.");
  auto it = optionalSections_.find(name);
  if (it == optionalSections_.end()) {
    return std::nullopt;
  }

  {
    auto optionalSectionsCache = optionalSectionsCache_.wlock();
    auto itCache = optionalSectionsCache->find(name);
    if (itCache != optionalSectionsCache->end()) {
      if (keepCache) {
        return Section{itCache->second->clone()};
      } else {
        auto metadata = std::move(itCache->second);
        optionalSectionsCache->erase(itCache);
        return Section{std::move(*metadata)};
      }
    }
  }
  if (it->second.size() == 0) {
    return Section{
        MetadataBuffer{velox::AlignedBuffer::allocate<char>(0, pool_)}};
  }
  return Section{std::move(*readMetadata(it->second))};
}

bool TabletReader::hasChunkStatsSection() const {
  return hasOptionalSection(std::string{kChunkStatsSection});
}

bool TabletReader::hasIndexSection() const {
  return hasOptionalSection(std::string{kIndexSection});
}

const index::IndexDescriptor* TabletReader::findIndexDescriptor(
    index::IndexFamily family,
    std::string_view indexName) const {
  NIMBLE_CHECK(!indexName.empty(), "Index name cannot be empty");
  for (const auto& descriptor : indexDescriptors_) {
    if (descriptor.family == family && descriptor.name == indexName) {
      return &descriptor;
    }
  }
  return nullptr;
}

void TabletReader::initIndexDescriptors() {
  if (!loadClusterIndex_ && !loadDenseIndexes_) {
    return;
  }
  NIMBLE_CHECK(indexDescriptors_.empty(), "Indexes already initialized");

  auto indexSection =
      loadOptionalSection(std::string{kIndexSection}, /*keepCache=*/true);
  if (!indexSection.has_value()) {
    return;
  }

  const auto* root = flatbuffers::GetRoot<serialization::IndexRoot>(
      indexSection->content().data());
  NIMBLE_CHECK_NOT_NULL(root);
  const auto* indexes = root->indexes();
  NIMBLE_CHECK_NOT_NULL(indexes);
  NIMBLE_CHECK_GT(indexes->size(), 0u, "Index section contains no descriptors");

  for (const auto* descriptor : *indexes) {
    NIMBLE_CHECK_NOT_NULL(descriptor);
    const auto* name = descriptor->name();
    NIMBLE_CHECK_NOT_NULL(name, "Index descriptor name is missing");
    const auto* indexRoot = descriptor->root();
    NIMBLE_CHECK_NOT_NULL(indexRoot, "Index descriptor root is missing");
    auto indexDescriptor = index::IndexDescriptor{
        .family = index::toIndexFamily(descriptor->family()),
        .name = name->str(),
        .root = toMetadataSection(indexRoot)};
    if (indexDescriptor.family == index::IndexFamily::Cluster) {
      for (const auto& existing : indexDescriptors_) {
        NIMBLE_CHECK(
            existing.family != index::IndexFamily::Cluster ||
                existing.name != indexDescriptor.name,
            "Nimble file contains multiple cluster indexes named '{}'",
            indexDescriptor.name);
      }
    }
    indexDescriptors_.emplace_back(std::move(indexDescriptor));
  }
}

void TabletReader::initClusterIndex() {
  if (!loadClusterIndex_) {
    return;
  }
  NIMBLE_CHECK_NULL(clusterIndex_, "Index already initialized.");

  if (!hasIndexSection()) {
    return;
  }

  const auto* clusterIndex =
      findIndexDescriptor(index::IndexFamily::Cluster, clusterIndexName_);
  if (clusterIndex == nullptr) {
    return;
  }

  clusterIndex_ = index::clusterIndexFactory(clusterIndex->name)
                      .createReader(
                          Section{std::move(*readMetadata(clusterIndex->root))},
                          pool_,
                          indexOptions_);
}

const index::IndexLookup* TabletReader::denseIndex(
    const std::vector<std::string>& columns) const {
  if (denseIndexRegistry_ == nullptr) {
    return nullptr;
  }
  return denseIndexRegistry_->findIndex(columns);
}

const index::IndexLookup* TabletReader::denseIndex(
    std::string_view name,
    const std::vector<std::string>& columns) const {
  if (denseIndexRegistry_ == nullptr) {
    return nullptr;
  }
  return denseIndexRegistry_->findIndex(name, columns);
}

void TabletReader::initDenseIndexes() {
  if (!loadDenseIndexes_) {
    return;
  }
  std::vector<index::DenseIndexRegistry::Entry> entries;
  for (const auto& descriptor : indexDescriptors_) {
    if (descriptor.family != index::IndexFamily::Dense) {
      continue;
    }
    entries.emplace_back(
        index::DenseIndexRegistry::Entry{
            descriptor.name,
            Section{std::move(*readMetadata(descriptor.root))}});
  }
  denseIndexRegistry_ = index::DenseIndexRegistry::create(
      std::move(entries), indexOptions_, pool_);
}

void TabletReader::initChunkStats(
    std::string_view footerBuf,
    uint64_t footerOffset) {
  if (!loadChunkStats_) {
    return;
  }
  if (!hasChunkStatsSection()) {
    return;
  }

  auto section =
      loadOptionalSection(std::string{kChunkStatsSection}, /*keepCache=*/false);
  NIMBLE_CHECK(section.has_value(), "Failed to load chunk stats section.");

  auto chunkStats = ChunkStats::create(std::move(section.value()));
  if (chunkStats->numGroups() > 0) {
    chunkStats_ = std::move(chunkStats);
  }

  // Eagerly pin the first chunk stats group if covered by the speculative
  // buffer.
  if (chunkStats_ != nullptr && chunkStats_->numGroups() == 1) {
    const auto& groupMetadata = chunkStats_->groupMetadata(0);
    if (groupMetadata.size() > 0) {
      auto metadataBuffer =
          tryExtractFromBuffer(footerBuf, footerOffset, groupMetadata, pool_);
      if (metadataBuffer != nullptr) {
        chunkStatsCache_.pin(
            0,
            ChunkStatsGroup::create(
                firstStripe(0), stripeCount(0), std::move(metadataBuffer)));
      }
    }
  }
}

uint32_t TabletReader::firstStripe(uint32_t stripeGroupIndex) const {
  const auto* stripesRoot =
      asFlatBuffersRoot<serialization::Stripes>(stripes_->content());
  const auto* groupIndices = stripesRoot->group_indices();
  NIMBLE_CHECK_NOT_NULL(groupIndices);

  for (uint32_t s = 0; s < stripeCount_; ++s) {
    if (groupIndices->Get(s) == stripeGroupIndex) {
      return s;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("No stripes found for group {}", stripeGroupIndex));
}

uint32_t TabletReader::rowToStripe(uint64_t row) const {
  NIMBLE_CHECK_GT(stripeCount_, 0u);
  NIMBLE_CHECK_LT(
      row,
      stripeRows_[stripeCount_],
      "File row {} exceeds total row count {}",
      row,
      stripeRows_[stripeCount_]);
  // Binary search on the prefix sum array.
  // upper_bound finds the first element > row, then subtract 1 to get the
  // stripe containing row.
  const auto it = std::upper_bound(
      stripeRows_.begin(), stripeRows_.begin() + stripeCount_ + 1, row);
  return static_cast<uint32_t>(it - stripeRows_.begin() - 1);
}

uint32_t TabletReader::stripeCount(uint32_t stripeGroupIndex) const {
  const auto* stripesRoot =
      asFlatBuffersRoot<serialization::Stripes>(stripes_->content());
  const auto* groupIndices = stripesRoot->group_indices();
  NIMBLE_CHECK_NOT_NULL(groupIndices);

  uint32_t count = 0;
  for (uint32_t s = 0; s < stripeCount_; ++s) {
    if (groupIndices->Get(s) == stripeGroupIndex) {
      ++count;
    }
  }
  return count;
}

std::shared_ptr<ChunkStatsGroup> TabletReader::chunkStats(
    uint32_t stripeGroupIndex) const {
  return chunkStatsCache_.getOrCreate(stripeGroupIndex);
}

std::shared_ptr<ChunkStatsGroup> TabletReader::loadChunkStatsGroup(
    uint32_t stripeGroupIndex) const {
  NIMBLE_CHECK_NOT_NULL(chunkStats_, "Chunk stats not initialized.");
  NIMBLE_CHECK_LT(
      stripeGroupIndex,
      chunkStats_->numGroups(),
      "Chunk stats group out of range");

  const auto& section = chunkStats_->groupMetadata(stripeGroupIndex);
  if (section.size() == 0) {
    return nullptr;
  }

  return ChunkStatsGroup::create(
      this->firstStripe(stripeGroupIndex),
      this->stripeCount(stripeGroupIndex),
      readMetadata(section));
}

} // namespace facebook::nimble
