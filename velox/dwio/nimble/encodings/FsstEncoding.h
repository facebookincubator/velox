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

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#endif
#pragma push_macro("fsst_create")
#pragma push_macro("fsst_duplicate")
#pragma push_macro("fsst_export")
#pragma push_macro("fsst_destroy")
#pragma push_macro("fsst_import")
#pragma push_macro("fsst_decoder")
#pragma push_macro("fsst_compress")
#pragma push_macro("fsst_decompress")
#define fsst_create nimble_fsst_create
#define fsst_duplicate nimble_fsst_duplicate
#define fsst_export nimble_fsst_export
#define fsst_destroy nimble_fsst_destroy
#define fsst_import nimble_fsst_import
#define fsst_decoder nimble_fsst_decoder
#define fsst_compress nimble_fsst_compress
#define fsst_decompress nimble_fsst_decompress
#include <fsst.h>
#pragma pop_macro("fsst_decompress")
#pragma pop_macro("fsst_compress")
#pragma pop_macro("fsst_decoder")
#pragma pop_macro("fsst_import")
#pragma pop_macro("fsst_destroy")
#pragma pop_macro("fsst_export")
#pragma pop_macro("fsst_duplicate")
#pragma pop_macro("fsst_create")
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "folly/Portability.h"
#include "folly/lang/Bits.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

/// Encoding for string data using FSST (Fast Static Symbol Table) compression.
///
/// FSST compresses strings by mapping frequent byte sequences (1-8 bytes) to
/// single-byte codes via a trained symbol table. Each string is compressed
/// independently, enabling random-access decompression without touching
/// neighboring strings.
///
/// Validates lengths and blob bounds lazily as rows are consumed, avoiding a
/// scan of the stream during construction. Reports malformed unread rows only
/// if a later read or skip reaches them.
/// Callers requiring eager structural validation can call skip() in bounded
/// batches on this encoding instance, then reset() before reading. Validating
/// decompression itself requires materializing every row.
///
/// Binary layout:
/// - Encoding::kPrefixSize bytes: standard Encoding prefix
/// - 1 byte: compression type for the concatenated FSST output
/// - varint: serialized FSST symbol table size
/// - N bytes: serialized FSST symbol table (~2KB typical)
/// - varint: lengths encoding size
/// - M bytes: nested encoding of compressed string lengths
/// - K bytes: concatenated FSST output, optionally compressed by the normal
///   Nimble encoding-layer compression policy
///
/// Only supports std::string_view data type.
class FsstEncoding final
    : public TypedEncoding<std::string_view, std::string_view> {
 public:
  using cppDataType = std::string_view;
  using physicalType = std::string_view;

  FsstEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  ~FsstEncoding() override {
    this->releaseBuffer(uncompressedBlob_);
  }

  FsstEncoding(const FsstEncoding&) = delete;
  FsstEncoding& operator=(const FsstEncoding&) = delete;
  FsstEncoding(FsstEncoding&&) = delete;
  FsstEncoding& operator=(FsstEncoding&&) = delete;

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Returns the estimated encoded size used by encoding selection.
  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<std::string_view>& statistics,
      const Encoding::Options& options);

  /// Returns the nested compressed-lengths encoding from serialized FSST data.
  static std::string_view lengthsEncoding(
      std::string_view encoding,
      const Encoding::Options& options = {});

  /// Captures FSST's nested compressed-lengths encoding layout.
  static void captureNestedEncoding(
      std::string_view encoding,
      std::vector<std::optional<const EncodingLayout>>& children,
      const Encoding::Options& options = {});

  /// Serialized-format helpers shared with FsstEncodingView.
  struct Header {
    /// Secondary compression applied to the concatenated FSST output.
    CompressionType compressionType{CompressionType::Uncompressed};
    /// Serialized FSST symbol table.
    std::string_view symbolTable;
    /// Nested encoding for per-row compressed string sizes.
    std::string_view lengths;
    /// Serialized blob, which may have secondary encoding-layer compression.
    std::string_view blob;
  };

  /// Parses the serialized FSST header at offset within encoding.
  static Header parseHeader(std::string_view encoding, size_t offset);

  /// Validates the common prefix before the base Encoding constructor parses
  /// it using the unchecked EncodingPrefix helpers.
  static std::string_view validateEncodedPrefix(
      std::string_view encoding,
      const Encoding::Options& options);

  /// Validates a serialized FSST symbol table before calling fsst_import(),
  /// whose upstream API does not accept an input-buffer length.
  static void validateSymbolTable(std::string_view symbolTable);

  /// Validates compressed lengths against blob bounds and FSST escape framing.
  /// Returns the number of compressed bytes covered by lengths.
  static size_t validateCompressedLengths(
      std::span<const uint32_t> lengths,
      std::string_view blob,
      size_t blobOffset);

  std::string debugString(int offset) const final;

 private:
  // Approximate serialized symbol table size in bytes (~2KB typical).
  static constexpr uint32_t kSymbolTableOverhead = 2048;

  // Maximum bytes represented by one FSST symbol.
  static constexpr size_t kMaxSymbolLength = 8;

  static constexpr size_t kStringPageSize = 256 * 1024;

  struct StringPageSlot {
    // Non-owning page address returned by stringBufferFactory_.
    char* data;
    // Number of writable bytes in the page.
    size_t capacity;
  };

  // Maps a compressed payload to its cached decompressed string for this call.
  struct DecodeCacheEntry {
    // References the encoded blob, whose symbol table is fixed for this call.
    std::string_view compressed;
    // References decoded bytes in the existing output pages.
    std::string_view decoded;
    // Hash of the compressed bytes, used to confirm lookup matches.
    uint64_t hash{0};
  };

  // Stores lookup metadata separately from the string views.
  struct DecodeCacheBucket {
    // -1 marks an unused bucket. Other values are zero-based entries_ indices.
    int32_t entryIndex{-1};
    // Rejects hash mismatches before reading the entry.
    uint32_t hashFingerprint{0};
  };

  // Stores encoded and decoded string references while checking for repeats.
  struct DecodeCacheProbeEntry {
    // References the encoded payload without copying it.
    const char* compressedData;
    // Delimits the encoded payload for binary-safe comparisons.
    size_t compressedSize;
    // References decoded bytes in output pages for reuse on a cache hit.
    const char* decodedData;
    // Delimits the decoded value, including any embedded null bytes.
    size_t decodedSize;
    // Avoids hashing the payload again when initializing the full cache.
    uint64_t hash;
  };

  // Caches decompressed strings keyed by their compressed payloads within one
  // encoding read call, using that encoding's fixed symbol table.
  class DecodeCache {
   public:
    // Starts sampling when this read can reuse cached values.
    DecodeCache(FsstEncoding& encoding, uint32_t numCandidates);
    // Resets this call's entries while retaining metadata capacity.
    ~DecodeCache();

    DecodeCache(const DecodeCache&) = delete;
    DecodeCache& operator=(const DecodeCache&) = delete;
    DecodeCache(DecodeCache&&) = delete;
    DecodeCache& operator=(DecodeCache&&) = delete;

    // Whether this call still samples or reuses cached decompressed strings.
    bool enabled() const {
      return enabled_;
    }

    // Returns stable decoded bytes, reusing earlier payloads when available.
    // Kept inline so materialize and templated visitors can dispatch each value
    // without a separate call; sampling and full-table lookups stay out of
    // line.
    FOLLY_ALWAYS_INLINE std::string_view decode(std::string_view compressed) {
      if (compressed.empty()) {
        // Empty strings have no decompression work to reuse or cache.
        return encoding_.decompressToStringBuffer(compressed);
      }
      if (sampling_) {
        return decodeWhileSampling(compressed);
      }
      if (numEntries_ == maxEntries_) {
        if (!velox::bits::isBitSet(
                hashFilter_.data(), hashFilterIndex(compressed))) {
          return encoding_.decompressToStringBuffer(compressed);
        }
        return decodeAtCapacity(compressed);
      }
      return decodeAndCache(compressed);
    }

   private:
    // Limits the unique prefix examined before the first repeated payload.
    static constexpr size_t kProbeEntryLimit = 64;
    // Keeps the probe table less than half full.
    static constexpr size_t kNumProbeBuckets = 2 * kProbeEntryLimit + 1;
    // The 64-bit golden ratio multiplier spreads suffix bits across the
    // filter index.
    static constexpr uint64_t kHashFilterMultiplier = 0x9e3779b97f4a7c15ULL;

    // Builds the full lookup table after the sample contains a duplicate.
    void initializeCache();

    // Tracks initial payloads and builds the full cache on the first repeat.
    std::string_view decodeWhileSampling(std::string_view compressed);

    // Reuses cached bytes or adds a new entry.
    std::string_view decodeAndCache(std::string_view compressed);

    // Reuses cached bytes and decodes misses without adding entries.
    std::string_view decodeAtCapacity(std::string_view compressed);

    // Computes a filter bit index to avoid full lookups on definite misses.
    // Kept inline for cache lookups at capacity.
    FOLLY_ALWAYS_INLINE size_t
    hashFilterIndex(std::string_view compressed) const {
      const auto size = compressed.size();
      const auto suffix = size >= sizeof(uint64_t)
          ? folly::loadUnaligned<uint64_t>(
                compressed.data() + size - sizeof(uint64_t))
          : velox::bits::loadPartialWord(
                reinterpret_cast<const uint8_t*>(compressed.data()),
                static_cast<int32_t>(size));
      const auto fingerprint =
          (suffix ^ static_cast<uint64_t>(size)) * kHashFilterMultiplier;
      return fingerprint >> hashFilterShift_;
    }

    // Finds an exact payload match or an empty bucket for insertion.
    DecodeCacheBucket& findBucket(std::string_view compressed, uint64_t hash);

    // Decodes a new payload and stores it in the cache.
    std::string_view decodeWithCacheMiss(
        std::string_view compressed,
        uint64_t hash,
        DecodeCacheBucket& bucket);

    // Provides the symbol table, decoded pages and memory pool for this call.
    FsstEncoding& encoding_;
    // Entries are assigned before their bucket indices are set.
    Vector<DecodeCacheEntry>& entries_;
    // Compact lookup metadata avoids reading entries_ on most misses.
    Vector<DecodeCacheBucket>& buckets_;
    // Unset bits rule out cached payloads; set bits require an exact lookup.
    Vector<uint64_t>& hashFilter_;
    // Selects the high fingerprint bits used to index the hash filter.
    int32_t hashFilterShift_{0};
    // Only entries below numProbeEntries_ are read, after all fields are set.
    // @lint-ignore CLANGTIDY cppcoreguidelines-pro-type-member-init
    std::array<DecodeCacheProbeEntry, kProbeEntryLimit> probeEntries_;
    // Zero means empty; other values are one-based probeEntries_ indices.
    // Initialized only when this cache is enabled.
    // @lint-ignore CLANGTIDY cppcoreguidelines-pro-type-member-init
    std::array<uint16_t, kNumProbeBuckets> probeBuckets_;
    // Bounds the number of distinct payloads retained during this call.
    size_t maxEntries_{0};
    // Entries below this index have been fully assigned and can be read.
    size_t numProbeEntries_{0};
    // Counts distinct payloads retained during sampling or full-cache lookup.
    size_t numEntries_{0};
    // Sampling is enabled only for reads with at least two candidate values.
    bool enabled_{false};
    // Stops sampling when the first duplicate initializes the full table.
    bool sampling_{true};
  };

  struct CompressedValues {
    explicit CompressedValues(velox::memory::MemoryPool* pool);

    // Serialized FSST symbol table buffer. Kept alive until final
    // serialization.
    velox::BufferPtr symbolTableBuffer;
    unsigned char* symbolTableData{nullptr};
    size_t symbolTableSize{0};

    // Concatenated FSST output storage. compressedPtrs point into this buffer.
    Vector<unsigned char> compressedBuffer;
    Vector<size_t> compressedLengths;
    Vector<unsigned char*> compressedPtrs;

    size_t totalInputSize{0};
    size_t totalCompressedSize{0};
  };

  // Checks that a sequential read remains within the row range.
  void checkReadRange(uint32_t rowCount, const char* operation) const;

  // Verifies the blob cursor when all rows have been consumed.
  void checkFinalBlobPosition() const;

  // Materializes rows while reusing decoded bytes within this call.
  void materializeWithDecodeCache(
      uint32_t numRows,
      std::span<std::string_view> output);

  // Reads selected rows while reusing decoded bytes within this call.
  template <typename V, typename Skip, typename ReadCompressed>
  void readWithDecodeCache(
      V& visitor,
      ReadWithVisitorParams& params,
      Skip skip,
      ReadCompressed readCompressed);

  // Trains FSST and compresses each input string independently.
  static CompressedValues compressValues(
      std::span<const physicalType> values,
      velox::memory::MemoryPool* pool);

  // Checks the compression target using the final serialized FSST size.
  static bool meetsCompressionTarget(
      uint64_t uncompressedSize,
      uint64_t encodedSize,
      double compressionTargetRatio);

  // Encodes the per-row FSST-compressed lengths as the nested lengths stream.
  static std::string_view encodeCompressedLengths(
      EncodingSelection<physicalType>& selection,
      std::span<const size_t> compressedLengths,
      Buffer& buffer,
      const Encoding::Options& options);

  // Encodes the original values with TrivialEncoding after FSST misses its
  // compression target.
  static std::string_view encodeTrivialFallback(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options);

  // Decompresses a validated row payload into stable page storage, using
  // scratch space when the page cannot hold the maximum possible expansion.
  // Returned views never reference decompressBuffer_. Output pages are owned by
  // stringBufferFactory_ and stay at fixed addresses until the caller releases
  // them; reset() may reuse their contents after a read has completed.
  // The caller must first validate the containing range with
  // validateCompressedLengths().
  std::string_view decompressToStringBuffer(std::string_view compressed);

  // Compares the bytes represented by a row's symbols with a byte range.
  bool testCompressedRange(
      std::string_view compressed,
      const velox::common::BytesRange& filter) const;

  // Returns writable storage when the current page has enough room.
  std::optional<std::span<char>> availablePageRange(size_t requiredBytes) const;

  // Ensures at least requiredBytes remain in the current page and returns the
  // first writable byte.
  char* ensurePage(size_t requiredBytes);

  const std::function<void*(uint32_t)> stringBufferFactory_;

  // FSST decoder (symbol table for decompression).
  fsst_decoder_t decoder_{};

  // Nested encoding for compressed string lengths.
  std::unique_ptr<Encoding> lengths_;

  // Unwrapped FSST string blob and the current byte offset within it.
  velox::BufferPtr uncompressedBlob_;
  std::string_view blob_;
  size_t blobOffset_{0};

  // Current row index.
  uint32_t row_{0};

  // Scratch buffer for materializing lengths.
  Vector<uint32_t> lengthBuffer_;

  // Current string page for stable storage of decompressed values.
  char* currentPage_{nullptr};
  size_t pageCapacityBytes_{0};
  size_t pageUsedBytes_{0};
  // Factory-allocated pages retained as non-owning slots for reuse after reset.
  std::vector<StringPageSlot> stringPages_;
  // Slot containing currentPage_.
  size_t currentPageIndex_{0};

  // Scratch buffer used when the remaining page capacity cannot hold the
  // maximum possible expansion of a compressed string.
  Vector<char> decompressBuffer_;

  // Retains entry capacity; cache contents are cleared after each call.
  Vector<DecodeCacheEntry> decodeCacheEntries_;
  // Retains bucket capacity to avoid allocating on each promotion.
  Vector<DecodeCacheBucket> decodeCacheBuckets_;
  // Retains hash filter capacity to avoid allocating on each promotion.
  Vector<uint64_t> decodeCacheHashFilter_;
};

template <typename V, typename Skip, typename ReadCompressed>
FOLLY_NOINLINE void FsstEncoding::readWithDecodeCache(
    V& visitor,
    ReadWithVisitorParams& params,
    Skip skip,
    ReadCompressed readCompressed) {
  DecodeCache decodeCache{*this, static_cast<uint32_t>(visitor.numRows())};
  detail::readWithVisitorSlow(visitor, params, skip, [&] {
    const auto compressed = readCompressed();
    return decodeCache.enabled() ? decodeCache.decode(compressed)
                                 : decompressToStringBuffer(compressed);
  });
}

template <typename V>
void FsstEncoding::readWithVisitor(V& visitor, ReadWithVisitorParams& params) {
  if (visitor.numRows() == 0) {
    checkReadRange(0, "Reading past end of FSST encoding.");
    checkFinalBlobPosition();
    return;
  }

  // Pre-materialize all compressed lengths needed for this read.
  const auto endRow = visitor.rowAt(visitor.numRows() - 1);
  auto numSelected = endRow + 1 - params.numScanned;
  if (auto& nulls = visitor.reader().nullsInReadRange()) {
    numSelected -= velox::bits::countNulls(
        nulls->template as<uint64_t>(), params.numScanned, endRow + 1);
  }
  NIMBLE_CHECK_GE(numSelected, 0, "Invalid FSST visitor row range.");
  checkReadRange(
      static_cast<uint32_t>(numSelected), "Reading past end of FSST encoding.");
  lengthBuffer_.resize(numSelected);
  lengths_->materialize(numSelected, lengthBuffer_.data());
  validateCompressedLengths(
      {lengthBuffer_.data(), lengthBuffer_.size()}, blob_, blobOffset_);

  auto* lengths = lengthBuffer_.data();
  const auto skip = [&](auto toSkip) {
    const auto compressedBytes =
        std::accumulate(lengths, lengths + toSkip, static_cast<size_t>(0));
    row_ += toSkip;
    blobOffset_ += compressedBytes;
    lengths += toSkip;
  };
  const auto readCompressed = [&] {
    const auto compressedLen = *lengths++;
    const auto compressed = blob_.substr(blobOffset_, compressedLen);
    ++row_;
    blobOffset_ += compressedLen;
    return compressed;
  };

  // Deterministic filter-only ranges do not need decoded strings.
  if constexpr (requires { typename V::FilterType; }) {
    if constexpr (
        (std::is_same_v<typename V::FilterType, velox::common::BytesRange> ||
         std::is_same_v<
             typename V::FilterType,
             velox::common::NegatedBytesRange>) &&
        std::is_same_v<typename V::Extract, velox::dwio::common::DropValues>) {
      if (visitor.filter().isDeterministic()) {
        detail::readWithVisitorSlow(
            visitor,
            params,
            skip,
            readCompressed,
            [&](std::string_view compressed, bool& atEnd) {
              const bool passesFilter = [&] {
                if constexpr (std::is_same_v<
                                  typename V::FilterType,
                                  velox::common::NegatedBytesRange>) {
                  return !testCompressedRange(
                      compressed, *visitor.filter().getNonNegated());
                } else {
                  return testCompressedRange(compressed, visitor.filter());
                }
              }();
              if (passesFilter) {
                visitor.filterPassed(std::string_view{});
              } else {
                visitor.filterFailed();
              }
              visitor.addRowIndex(1);
              atEnd = visitor.atEnd();
            });
        checkFinalBlobPosition();
        return;
      }
    }
  }

  if (!options_.fsstDecodeCacheEnabled()) {
    detail::readWithVisitorSlow(visitor, params, skip, [&] {
      return decompressToStringBuffer(readCompressed());
    });
  } else {
    readWithDecodeCache(visitor, params, skip, readCompressed);
  }
  checkFinalBlobPosition();
}

} // namespace facebook::nimble
