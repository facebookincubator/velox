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
/// Binary layout:
/// - Encoding::kPrefixSize bytes: standard Encoding prefix
/// - varint: serialized FSST symbol table size
/// - N bytes: serialized FSST symbol table (~2KB typical)
/// - varint: lengths encoding size
/// - M bytes: nested encoding of compressed string lengths
/// - K bytes: compressed string blob (concatenated FSST-compressed strings)
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

  std::string debugString(int offset) const final;

 private:
  // Approximate serialized symbol table size in bytes (~2KB typical).
  static constexpr uint32_t kSymbolTableOverhead = 2048;

  // Maximum bytes a compressed FSST code can expand to.
  static constexpr size_t kMaxSymbolLength = 8;

  static constexpr size_t kStringPageSize = 256 * 1024;

  struct StringPageSlot {
    // Non-owning page address returned by stringBufferFactory_.
    char* data;
    // Number of writable bytes in the page.
    size_t capacity;
  };

  struct Header {
    // Serialized FSST symbol table.
    std::string_view symbolTable;

    // Nested encoding for per-row compressed string sizes.
    std::string_view lengths;

    // Concatenated FSST-compressed string data.
    std::string_view blob;
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

  // Parses the serialized FSST header at offset within encoding.
  static Header parseHeader(std::string_view encoding, size_t offset);

  // Validates the common prefix before the base Encoding constructor parses
  // it using the unchecked EncodingPrefix helpers.
  static std::string_view validateEncodedPrefix(
      std::string_view encoding,
      const Encoding::Options& options);

  // Validates a serialized FSST symbol table before calling fsst_import(),
  // whose upstream API does not accept an input-buffer length.
  static void validateSymbolTable(std::string_view symbolTable);

  // Validates compressed lengths against blob bounds and FSST escape framing.
  // Returns the number of compressed bytes covered by lengths.
  static size_t validateCompressedLengths(
      std::span<const uint32_t> lengths,
      std::string_view blob,
      size_t blobOffset);

  // Validates the complete nested lengths stream and compressed blob.
  void validateLengthsAndBlob();

  // Checks that a sequential read remains within the row range.
  void checkReadRange(uint32_t rowCount, const char* operation) const;

  // Verifies the blob cursor when all rows have been consumed.
  void checkFinalBlobPosition() const;

  // Trains FSST and compresses each input string independently.
  static CompressedValues compressValues(
      std::span<const physicalType> values,
      velox::memory::MemoryPool* pool);

  // Checks whether the final FSST encoding meets the compression target.
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

  // Decompresses a single compressed string and copies the result into the
  // string buffer page. Returns a stable string_view.
  std::string_view decompressToStringBuffer(std::string_view compressed);

  void ensurePage(size_t requiredBytes);

  const std::function<void*(uint32_t)> stringBufferFactory_;

  // FSST decoder (symbol table for decompression).
  fsst_decoder_t decoder_{};

  // Nested encoding for compressed string lengths.
  std::unique_ptr<Encoding> lengths_;

  // Compressed string blob and the current byte offset within it.
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

  // Scratch buffer for decompression output.
  Vector<char> decompressBuffer_;
};

template <typename V>
void FsstEncoding::readWithVisitor(V& visitor, ReadWithVisitorParams& params) {
  if (visitor.numRows() == 0) {
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
  auto* lengths = lengthBuffer_.data();
  validateCompressedLengths(
      {lengthBuffer_.data(), lengthBuffer_.size()}, blob_, blobOffset_);

  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) {
        const auto compressedBytes =
            std::accumulate(lengths, lengths + toSkip, static_cast<size_t>(0));
        row_ += toSkip;
        blobOffset_ += compressedBytes;
        lengths += toSkip;
      },
      [&] {
        const auto compressedLen = *lengths++;
        auto result =
            decompressToStringBuffer(blob_.substr(blobOffset_, compressedLen));
        ++row_;
        blobOffset_ += compressedLen;
        return result;
      });
  checkFinalBlobPosition();
}

} // namespace facebook::nimble
