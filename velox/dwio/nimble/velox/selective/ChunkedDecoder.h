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

#include <functional>
#include <optional>
#include <string_view>
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingTrait.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/velox/selective/NimbleData.h"
#include "velox/dwio/nimble/velox/selective/ReaderBase.h"

namespace facebook::nimble {

/// Invoked when a chunk boundary is crossed during skip, ensureLoaded, or
/// multi-chunk dictionary index reads. Returns true to continue processing
/// subsequent chunks, or false to stop (e.g., when the new chunk's encoding
/// is not dictionary-compatible and requires flat fallback).
using ChunkBoundaryCallback = std::function<bool()>;

/// Default chunk boundary callback that always returns true (continue
/// processing). Used when the caller does not need to react to chunk
/// transitions.
inline const ChunkBoundaryCallback kNoopAtChunkBoundary = [] { return true; };

/// Controls how accumulated string buffers are delivered to the reader after
/// a chunk read. By default (nullptr), replaces the reader's string buffers.
/// Callers that need to preserve existing string buffers (e.g., reactive
/// fallback from dict to flat mid-read) pass an appending handler.
using SaveStringBufferCallback =
    std::function<void(std::vector<velox::BufferPtr>&&)>;

class ChunkedDecoderTestHelper;

class ChunkedDecoder {
 public:
  ChunkedDecoder(
      std::unique_ptr<velox::dwio::common::SeekableInputStream> input,
      std::shared_ptr<index::StreamIndex> streamIndex,
      bool decodeValuesWithNulls,
      const EncodingFactory* encodingFactory,
      velox::memory::MemoryPool* pool,
      bool stringDecoderZeroCopy = false,
      velox::dwio::common::DecodingStats* decodingStats = nullptr,
      DictionaryAlphabetLoader dictionaryAlphabetLoader = nullptr)
      : input_{std::move(input)},
        pool_{pool},
        decodeValuesWithNulls_{decodeValuesWithNulls},
        encodingFactory_(encodingFactory),
        stringDecoderZeroCopy_{stringDecoderZeroCopy},
        streamIndex_{std::move(streamIndex)},
        streamRowCount_{
            streamIndex_ ? std::optional<uint32_t>(streamIndex_->rowCount())
                         : std::nullopt},
        decodingStats_{decodingStats},
        dictionaryAlphabetLoader_{std::move(dictionaryAlphabetLoader)} {
    NIMBLE_CHECK_NOT_NULL(input_);
    NIMBLE_CHECK_NOT_NULL(encodingFactory_);
  }

  /// Skips non-null values. The optional onChunkBoundary callback is invoked
  /// whenever the skip crosses into a new chunk. This allows callers (e.g.,
  /// StringColumnReader) to invalidate cached state like mergedAlphabet_.
  void skip(
      int64_t numValues,
      const ChunkBoundaryCallback& onChunkBoundary = kNoopAtChunkBoundary);

  /// Decode nulls or in-map buffer values.
  void nextBools(uint64_t* data, int64_t count, const uint64_t* incomingNulls);

  /// Decode index type values, which usually needed to be read fully.
  void nextIndices(int32_t* data, int64_t count, const uint64_t* incomingNulls);

  const velox::BufferPtr& getPreloadedValues() {
    NIMBLE_CHECK(
        decodeValuesWithNulls_,
        "Only Array and Map types preload offset values with nulls");
    return nullableValues_;
  }

  /// Reads values using a visitor pattern, handling chunk transitions
  /// automatically. The optional saveStringBuffersFn callback controls how
  /// accumulated string buffers are delivered to the reader. By default,
  /// replaces the reader's string buffers. Callers that need to preserve
  /// existing string buffers (e.g., flat encoding fallback after a partial dict
  /// read, where abandonDictionaryEncoding already placed string buffers)
  /// pass an appending handler. readDictionaryIndices does not need this
  /// because it reads dict indices (int32_t), not string values — no string
  /// buffers are produced during dict index reads.
  /// @param readOffset Starting position in nullsInReadRange_ for
  ///   reading from the middle of a batch (e.g., flat encoding fallback after a
  ///   partial dict read). Defaults to 0.
  template <typename DecoderVisitor>
  void readWithVisitor(
      DecoderVisitor& visitor,
      velox::vector_size_t readOffset = 0,
      const SaveStringBufferCallback& saveStringBuffersFn = nullptr) {
    NIMBLE_CHECK(
        readOffset == 0 || stringDecoderZeroCopy_,
        "readOffset is only valid for dict→flat encoding fallback reads");
    // The visitor always carries the complete output row set. When a read is
    // resumed mid-range (readOffset > 0 — e.g. the dict→flat abandon-dictionary
    // fallback), the caller advances the visitor's rowIndex past the
    // already-read prefix instead of slicing the row set, so null prep still
    // sees the full range for buffer sizing and the return-nulls mode.
    const velox::RowSet rows(visitor.rows(), visitor.numRows());
    NIMBLE_DCHECK(std::is_sorted(rows.begin(), rows.end()));
    const auto numRows = visitor.numRows();
    ReadWithVisitorParams params{};
    params.numScanned = readOffset;
    // readOffset > 0 means a single read range is being decoded in segments
    // across multiple readWithVisitor calls (e.g. the dict→flat
    // abandon-dictionary fallback resuming at the abandoned chunk boundary), so
    // the lazy null-prep hooks below must prepare resultNulls_ for the WHOLE
    // read range, not just this segment. An earlier segment already prepared it
    // for the full output IFF that segment emitted nulls (reader hasNulls()):
    // seed the guard to skip re-preparing in that case — appending to, rather
    // than clobbering, the earlier segment's nulls — and leave it unset when
    // the earlier segment produced no nulls, so a nullable chunk in this
    // segment prepares the buffer.
    bool resultNullsPrepared{readOffset > 0 && visitor.reader().hasNulls()};
    // Allocate one extra byte (8 bits) for nulls to handle multi-chunk
    // reading, where each chunk we need to align the result nulls to byte
    // boundary then shift.
    params.prepareResultNulls = [&resultNullsPrepared, &visitor, &rows] {
      if (FOLLY_UNLIKELY(!resultNullsPrepared)) {
        // Pass the real read-range null state so prepareNulls short-circuits
        // (no allocate/clear) when the range carries no nulls. Only mark the
        // buffer prepared when prepareNulls actually allocated it: on a
        // null-free call it no-ops, so a later chunk that does carry nulls must
        // still trigger a real prepare.
        const bool hasNulls = visitor.reader().rawNullsInReadRange() != nullptr;
        visitor.reader().prepareNulls(rows, hasNulls, /*extraRows=*/8);
        resultNullsPrepared = hasNulls;
      }
    };
    params.setReturnNullsMode = [&visitor, &rows] {
      visitor.reader().setReturnNullsMode(rows);
    };
    bool readerNullsMade{false};
    if (auto& nulls = visitor.reader().nullsInReadRange()) {
      // For flat map child columns, NimbleData::readNulls() may alias inMap_
      // and nullsInReadRange_ to the same buffer (when the child has no
      // separate nulls stream). Since makeReaderNulls() returns a mutable
      // pointer that NullableEncoding will overwrite with value-level nulls,
      // we must copy-on-first-write to avoid corrupting the in-map bitmap.
      if (static_cast<const NimbleData&>(visitor.reader().formatData())
              .inMap() == nulls->template as<uint64_t>()) {
        params.makeReaderNulls =
            [&readerNullsMade, &nulls, pool = pool_]() mutable {
              if (FOLLY_UNLIKELY(!readerNullsMade)) {
                nulls = velox::AlignedBuffer::copy(pool, nulls);
                readerNullsMade = true;
              }
              return nulls->template asMutable<uint64_t>();
            };
      } else {
        params.makeReaderNulls = [&readerNullsMade, &nulls] {
          readerNullsMade = true;
          return nulls->template asMutable<uint64_t>();
        };
      }
      dispatchReadWithVisitorImpl<true>(
          visitor, nulls->template as<uint64_t>(), params, saveStringBuffersFn);
    } else {
      params.makeReaderNulls = [&readerNullsMade, this, numRows, &visitor] {
        if (FOLLY_UNLIKELY(!readerNullsMade)) {
          auto& nulls = visitor.reader().nullsInReadRange();
          if (!nulls) {
            nulls = velox::AlignedBuffer::allocate<bool>(
                visitor.rowAt(numRows - 1) + 1,
                pool_,
                /*value=*/velox::bits::kNotNull);
          }
          readerNullsMade = true;
        }
        return visitor.reader()
            .nullsInReadRange()
            ->template asMutable<uint64_t>();
      };
      dispatchReadWithVisitorImpl<false>(
          visitor, /*nulls=*/nullptr, params, saveStringBuffersFn);
    }
  }

  /// Decode values from a nullable encoding.
  /// Template on data type T to support int64_t, uint32_t, uint16_t, etc.
  /// Populates nulls buffer and data buffer.
  template <typename T>
  void decodeNullable(
      uint64_t* nulls,
      T* data,
      int64_t count,
      const uint64_t* incomingNulls) {
    const int64_t totalNumValues = !incomingNulls
        ? count
        : velox::bits::countNonNulls(incomingNulls, 0, count);

    if (incomingNulls != nullptr) {
      if (totalNumValues == 0) {
        if (nulls) {
          velox::bits::fillBits(nulls, 0, count, velox::bits::kNull);
        }
        return;
      }

      velox::bits::Bitmap scatterBitmap{
          incomingNulls, static_cast<uint32_t>(count)};
      // These are subranges in the scatter bit map for each materialize
      // call.
      uint32_t offset{0};
      uint32_t endOffset{0};
      for (int64_t i = 0; i < totalNumValues;) {
        if (FOLLY_UNLIKELY(remainingValues_ == 0)) {
          loadNextChunk();
        }

        const auto numValues = std::min(totalNumValues - i, remainingValues_);
        endOffset = velox::bits::findSetBit(
            static_cast<const char*>(scatterBitmap.bits()),
            offset,
            scatterBitmap.size(),
            numValues + 1);
        velox::bits::Bitmap localBitmap{scatterBitmap.bits(), endOffset};
        auto nonNullCount = encoding_->materializeNullable(
            numValues, data, [&]() { return nulls; }, &localBitmap, offset);
        if (nulls && nonNullCount == endOffset - offset) {
          velox::bits::fillBits(
              nulls, offset, endOffset, velox::bits::kNotNull);
        }
        advancePosition(numValues);
        i += numValues;
        offset = endOffset;
      }
    } else {
      for (int64_t i = 0; i < totalNumValues;) {
        if (FOLLY_UNLIKELY(remainingValues_ == 0)) {
          loadNextChunk();
        }

        const auto numValues = std::min(totalNumValues - i, remainingValues_);
        const auto nonNullCount = encoding_->materializeNullable(
            numValues, data, [&]() { return nulls; }, nullptr, i);
        if (nulls && nonNullCount == numValues) {
          velox::bits::fillBits(nulls, i, i + numValues, velox::bits::kNotNull);
        }
        advancePosition(numValues);
        i += numValues;
      }
    }
  }

  /// A temporary solution to estimate row size before column statistics are
  /// implemented.  This is based on the first chunk so is only accurate when
  /// there is a single chunk in the stripe, or all the first chunks of all
  /// streams are aligned at the same row count (unlikely).
  ///
  /// TODO: Remove after column statistics are added.
  std::optional<size_t> estimateRowCount() const;

  /// A temporary solution to estimate row size, similar to estimateRowCount().
  ///
  /// TODO: Remove after column statistics are added.
  std::optional<size_t> estimateStringDataSize() const;

  // Ensures a chunk with remaining values is loaded in the chunked decoder.
  // Loads the first chunk if none has been loaded yet, or advances to the
  // next chunk if the current one is fully consumed. Called before inspecting
  // the current encoding (e.g., currentEncoding, dictionaryConvertible) so
  // that the caller sees the encoding and remainingValues of the chunk that
  // will actually be read.
  void ensureLoaded(
      const ChunkBoundaryCallback& onChunkBoundary = kNoopAtChunkBoundary) {
    if (encoding_ != nullptr && remainingValues_ != 0) {
      return;
    }

    if (hasMoreChunks()) {
      loadNextChunk(onChunkBoundary);
    }
  }

  /// Returns the current encoding, or nullptr if no chunk has been loaded yet.
  const Encoding* currentEncoding() const {
    return encoding_.get();
  }

  /// Returns true if the current chunk's encoding supports dictionary index
  /// reading. Delegates to Encoding::dictionaryEnabled() which handles Dict,
  /// Nullable→Dict, and MC→Dict via virtual dispatch.
  /// Caller must call ensureLoaded() first.
  bool dictionaryConvertible() const;

  int64_t remainingValues() const {
    return remainingValues_;
  }

  // Reads dictionary indices across chunks, invoking onChunkBoundary at each
  // chunk transition. The callback can extend the merged alphabet and refresh
  // cached state. The callback returns true to continue reading with the new
  // chunk's dictionary, or false to stop (e.g., when the new chunk is not
  // dictionary-compatible and requires flat fallback).
  //
  // Uses a dedicated loop instead of readWithVisitorImpl to avoid the
  // testNulls-based chunk transition in computeNextRowIndex, which counts
  // non-nulls across chunk boundaries and produces incorrect index reads.
  //
  // Returns true if all requested rows were consumed. Returns false if
  // reading stopped early at a chunk boundary (the callback returned false);
  // in that case it advances the reader's readOffset_ to the boundary so the
  // flat encoding fallback resumes the decode there.
  template <typename DictionaryVisitor>
  bool readDictionaryIndices(
      DictionaryVisitor& visitor,
      const ChunkBoundaryCallback& onChunkBoundary) {
    NIMBLE_CHECK(
        stringDecoderZeroCopy_,
        "Dictionary indices reading requires non-legacy encoding path");
    NIMBLE_CHECK_NOT_NULL(encoding_);

    const velox::RowSet rows(visitor.rows(), visitor.numRows());
    NIMBLE_DCHECK(std::is_sorted(rows.begin(), rows.end()));
    ReadWithVisitorParams params{};
    bool resultNullsPrepared{false};
    // Allocate one extra byte (8 bits) for nulls to handle multi-chunk
    // reading, where each chunk we need to align the result nulls to byte
    // boundary then shift.
    params.prepareResultNulls = [&resultNullsPrepared, &visitor, &rows] {
      if (FOLLY_UNLIKELY(!resultNullsPrepared)) {
        // Pass the real read-range null state so prepareNulls short-circuits
        // (no allocate/clear) when the range carries no nulls. Only mark the
        // buffer prepared when prepareNulls actually allocated it: on a
        // null-free call it no-ops, so a later chunk that does carry nulls must
        // still trigger a real prepare.
        const bool hasNulls = visitor.reader().rawNullsInReadRange() != nullptr;
        visitor.reader().prepareNulls(rows, hasNulls, /*extraRows=*/8);
        resultNullsPrepared = hasNulls;
      }
    };
    params.setReturnNullsMode = [&visitor, &rows] {
      visitor.reader().setReturnNullsMode(rows);
    };
    bool readerNullsMade{false};
    if (auto& nulls = visitor.reader().nullsInReadRange()) {
      // For flat map child columns, NimbleData::readNulls() may alias inMap_
      // and nullsInReadRange_ to the same buffer (when the child has no
      // separate nulls stream). Since makeReaderNulls() returns a mutable
      // pointer that NullableEncoding will overwrite with value-level nulls,
      // we must copy-on-first-write to avoid corrupting the in-map bitmap.
      if (static_cast<const NimbleData&>(visitor.reader().formatData())
              .inMap() == nulls->template as<uint64_t>()) {
        params.makeReaderNulls =
            [&readerNullsMade, &nulls, pool = pool_]() mutable {
              if (FOLLY_UNLIKELY(!readerNullsMade)) {
                nulls = velox::AlignedBuffer::copy(pool, nulls);
                readerNullsMade = true;
              }
              return nulls->template asMutable<uint64_t>();
            };
      } else {
        params.makeReaderNulls = [&readerNullsMade, &nulls] {
          readerNullsMade = true;
          return nulls->template asMutable<uint64_t>();
        };
      }
      return readDictionaryIndicesImpl<true>(
          visitor, nulls->template as<uint64_t>(), onChunkBoundary, params);
    } else {
      const auto numRows = visitor.numRows();
      params.makeReaderNulls = [&readerNullsMade, this, numRows, &visitor] {
        if (FOLLY_UNLIKELY(!readerNullsMade)) {
          auto& nulls = visitor.reader().nullsInReadRange();
          if (!nulls) {
            nulls = velox::AlignedBuffer::allocate<bool>(
                visitor.rowAt(numRows - 1) + 1,
                pool_,
                /*value=*/velox::bits::kNotNull);
          }
          readerNullsMade = true;
        }
        return visitor.reader()
            .nullsInReadRange()
            ->template asMutable<uint64_t>();
      };
      return readDictionaryIndicesImpl<false>(
          visitor, /*nulls=*/nullptr, onChunkBoundary, params);
    }
  }

 private:
  // For regular cases, kHasNulls is false because NullableEncoding handles
  // the nulls from the outside. But when dealing with flatmap types, inMap
  // streams are not part of the data stream but interact with the value
  // streams just like nulls. Not using this parameter causes a bug for
  // flatmap data shapes.

  // Dedicated chunk iteration loop for dictionary index reads. Unlike
  // readWithVisitorImpl, this handles chunk transitions explicitly at the
  // top of the loop (for both kHasNulls and !kHasNulls paths) instead of
  // relying on testNulls which counts non-nulls across chunk boundaries.
  template <bool kHasNulls, typename V>
  bool readDictionaryIndicesImpl(
      V& visitor,
      const uint64_t* nulls,
      const ChunkBoundaryCallback& onChunkBoundary,
      ReadWithVisitorParams& params) {
    constexpr bool kExtractToReader = std::
        is_same_v<typename V::Extract, velox::dwio::common::ExtractToReader>;
    const auto numRows = visitor.numRows();

    std::vector<velox::BufferPtr> stringBuffers;
    SCOPE_EXIT {
      if (stringDecoderZeroCopy_) {
        visitor.reader().setStringBuffers(std::move(stringBuffers));
      }
    };

    // For kHasNulls, count the value-stream rows this read will consume, once
    // up front, so the per-chunk-exhaustion guard below is an O(1) decrement
    // rather than a bitmap re-scan each time a chunk runs out.
    //
    // The nulls bitmap here carries only structural nulls (incoming struct
    // nulls plus flatmap in-map), populated whole-range by
    // prepareRead()/readNulls() before this read. NullableEncoding merges
    // value-level nulls only into the already-decoded prefix
    // (materializeNullsForVisitor), never the remaining range, so over
    // [numScanned, endRow) every structural non-null is exactly one
    // value-stream row to consume; countNonNulls is thus an exact total.
    //
    // The !kHasNulls path has no null bitmap: every remaining row is a value,
    // so it always loads (like the flat !kHasNulls path) and does not use this
    // counter.
    int64_t remainingValues = 0;
    if constexpr (kHasNulls) {
      const auto endRow = V::dense
          ? params.numScanned + (numRows - visitor.rowIndex())
          : visitor.rowAt(numRows - 1) + 1;
      remainingValues = static_cast<int64_t>(
          velox::bits::countNonNulls(nulls, params.numScanned, endRow));
    }

    while (visitor.rowIndex() < numRows) {
      // Load the next chunk only when a value-stream row still remains in this
      // read range, mirroring the flat path's testNulls guard
      // (c > 0 && remainingValues_ == 0). An all-null run leaves
      // remainingValues == 0: skip the load; computeNextRowIndex then
      // emits the null rows with numNonNulls == 0. If a value is expected but
      // no chunk remains, loadNextChunk throws on the metadata/bitmap mismatch
      // (corruption).
      //
      // The load is driven from this loop (computeNextRowIndex runs with
      // kReadAllChunks=false) rather than reused from the flat testNulls path.
      // Both paths read one chunk per pass and stop at its boundary; the
      // difference is where the boundary load lives. In the flat path it lives
      // inside the testNulls bit-scan and is bare: no seam at which
      // to run onChunkBoundary
      // (merge the per-chunk alphabet
      // + refresh scan state) or to abandon to the flat fallback when the next
      // chunk is not dictionary-compatible. Hoisting the load here gives those
      // a clean seam.
      if (FOLLY_UNLIKELY(remainingValues_ == 0)) {
        // With nulls, load only if a value-stream row still remains; without a
        // null bitmap every remaining row is a value, so always load (as the
        // flat !kHasNulls path does).
        const bool needMoreValues = kHasNulls ? (remainingValues > 0) : true;
        if (needMoreValues) {
          loadNextChunk();
          if (!onChunkBoundary()) {
            // The new chunk is not dictionary-compatible. readOffset_ has not
            // moved during this read, so it still holds the read's start
            // offset; adding params.numScanned (the rows consumed up to this
            // chunk boundary) advances it to the boundary, where the flat
            // encoding fallback resumes the decode.
            //
            // TODO: Make the decoder the single owner of readOffset_ — also
            // advance it on the full-read path here instead of
            // readWithDictionary doing readOffset_ += endReadRow, so the update
            // is not split across the reader and the decoder.
            visitor.reader().setReadOffset(
                visitor.reader().readOffset() + params.numScanned);
            return false;
          }

          // onChunkBoundary() rebuilt the per-chunk filter dictionary in the
          // reader's scan state. Re-snapshot it into the visitor so the next
          // chunk's filter evaluation uses this chunk's dictionary, not the
          // stale copy captured when the visitor was constructed.
          visitor.refreshScanState();
        }
        // Otherwise all remaining rows are null; no chunk is needed.
      }

      velox::vector_size_t numNulls{0};
      velox::vector_size_t numNonNulls{0};
      const auto endRowIndex =
          computeNextRowIndex<kHasNulls, /*kReadAllChunks=*/false>(
              visitor, nulls, params, numRows, numNulls, numNonNulls);

      if (visitor.rowIndex() < endRowIndex) {
        visitor.setRows(velox::RowSet(visitor.rows(), endRowIndex));
        if (numNonNulls > 0) {
          callReadIndicesWithVisitor(*encoding_, visitor, params);
        } else if (!visitor.allowNulls()) {
          visitor.setRowIndex(endRowIndex);
        } else {
          bool emitNullsRowByRow = true;
          if constexpr (kExtractToReader) {
            params.prepareResultNulls();
            if (visitor.reader().returnReaderNulls()) {
              auto numValues = endRowIndex - visitor.rowIndex();
              visitor.setRowIndex(endRowIndex);
              visitor.addNumValues(numValues);
              emitNullsRowByRow = false;
            }
          }
          if (emitNullsRowByRow) {
            bool atEnd = false;
            while (!atEnd) {
              visitor.processNull(atEnd);
            }
          }
        }
        NIMBLE_CHECK_EQ(visitor.rowIndex(), endRowIndex);
      }
      advancePosition(numNonNulls);
      params.numScanned += numNulls + numNonNulls;
      if constexpr (kHasNulls) {
        remainingValues -= numNonNulls;
      }

      if (stringDecoderZeroCopy_) {
        stringBuffers.reserve(
            stringBuffers.size() + currentStringBuffers_.size());
        for (const auto& buf : currentStringBuffers_) {
          stringBuffers.emplace_back(buf);
        }
      }
    }
    return true;
  }

  template <bool kHasMask>
  bool testNulls(
      const uint64_t* nulls,
      int32_t i,
      uint64_t mask,
      velox::vector_size_t& numNulls,
      velox::vector_size_t& numNonNulls,
      velox::vector_size_t& outIndex) {
    auto word = nulls[i];
    if constexpr (kHasMask) {
      word &= mask;
    }
    auto c = __builtin_popcountll(word);
    if (FOLLY_UNLIKELY(c > 0 && remainingValues_ == 0)) {
      loadNextChunk();
    }
    if (c == 0 || numNonNulls + c < remainingValues_) {
      numNulls += (kHasMask ? __builtin_popcountll(mask) : 64) - c;
      numNonNulls += c;
      return true;
    }
    for (int j = 0; j < 64; ++j) {
      if constexpr (kHasMask) {
        if (!(mask & (1ull << j))) {
          continue;
        }
      }
      if (word & (1ull << j)) {
        if (++numNonNulls == remainingValues_) {
          outIndex = i * 64 + j;
          return false;
        }
      } else {
        ++numNulls;
      }
    }
    NIMBLE_UNREACHABLE();
  }

  velox::vector_size_t lastNonNullIndex(
      const uint64_t* nulls,
      velox::vector_size_t begin,
      velox::vector_size_t end,
      velox::vector_size_t& numNulls,
      velox::vector_size_t& numNonNulls) {
    velox::vector_size_t index;
    const bool allConsumed = velox::bits::testWords(
        begin,
        end,
        [&](int32_t i, uint64_t mask) {
          return testNulls<true>(nulls, i, mask, numNulls, numNonNulls, index);
        },
        [&](int32_t i) {
          return testNulls<false>(nulls, i, 0, numNulls, numNonNulls, index);
        });
    return allConsumed ? end - 1 : index;
  }

  // Computes the end row index for the current chunk's read range.
  //
  // When kReadAllChunks is true (default, used by readWithVisitorImpl),
  // the kHasNulls path calls lastNonNullIndex which may load new chunks via
  // testNulls when remainingValues_ reaches 0. This allows the flat read
  // path to transparently span chunks.
  //
  // When kReadAllChunks is false (used by readDictionaryIndicesImpl),
  // the kHasNulls path caps at the current chunk's boundary using
  // countNonNulls + findSetBit instead of lastNonNullIndex. This prevents
  // testNulls from loading new chunks, which would bypass the dict path's
  // onChunkBoundary callback.
  template <bool kHasNulls, bool kReadAllChunks = true, typename V>
  velox::vector_size_t computeNextRowIndex(
      const V& visitor,
      const uint64_t* nulls,
      const ReadWithVisitorParams& params,
      velox::vector_size_t numRows,
      velox::vector_size_t& numNulls,
      velox::vector_size_t& numNonNulls) {
    numNulls = 0;
    numNonNulls = 0;

    if constexpr (V::dense) {
      // Dense: row positions equal indices, so endRowIndex is computed
      // directly from numScanned without lower_bound.
      NIMBLE_DCHECK_EQ(visitor.currentRow(), params.numScanned);
      if constexpr (!kHasNulls) {
        numNonNulls =
            std::min<int64_t>(numRows - visitor.rowIndex(), remainingValues_);
        return visitor.rowIndex() + numNonNulls;
      }
      // Absolute end in the null bitmap. For a row set starting at 0 this
      // equals numRows. For continuation reads (numScanned > 0 at start),
      // this accounts for the offset so the bitmap is scanned correctly.
      const auto nullsEnd = params.numScanned + (numRows - visitor.rowIndex());
      if constexpr (kReadAllChunks) {
        return 1 +
            lastNonNullIndex(
                   nulls, params.numScanned, nullsEnd, numNulls, numNonNulls);
      }
      // !kReadAllChunks: cap at the current chunk boundary.
      numNonNulls = static_cast<velox::vector_size_t>(
          velox::bits::countNonNulls(nulls, params.numScanned, nullsEnd));
      velox::vector_size_t chunkEnd{0};
      if (numNonNulls <= remainingValues_) {
        chunkEnd = nullsEnd;
      } else {
        numNonNulls = remainingValues_;
        chunkEnd = static_cast<velox::vector_size_t>(
            1 +
            velox::bits::findSetBit(
                reinterpret_cast<const char*>(nulls),
                params.numScanned,
                nullsEnd,
                remainingValues_));
      }
      numNulls = (chunkEnd - params.numScanned) - numNonNulls;
      return visitor.rowIndex() + (chunkEnd - params.numScanned);
    } else {
      // Sparse: row positions are non-contiguous, so we need lower_bound
      // to map chunkEnd back to a row index.
      const auto endRow = visitor.rowAt(numRows - 1) + 1;
      velox::vector_size_t chunkEnd{0};
      if constexpr (!kHasNulls) {
        chunkEnd = params.numScanned + remainingValues_;
        numNonNulls =
            std::min<int64_t>(endRow - params.numScanned, remainingValues_);
      } else if constexpr (kReadAllChunks) {
        chunkEnd = 1 +
            lastNonNullIndex(
                       nulls, params.numScanned, endRow, numNulls, numNonNulls);
      } else {
        // !kReadAllChunks: cap at the current chunk boundary.
        numNonNulls = static_cast<velox::vector_size_t>(
            velox::bits::countNonNulls(nulls, params.numScanned, endRow));
        if (numNonNulls <= remainingValues_) {
          chunkEnd = endRow;
        } else {
          numNonNulls = remainingValues_;
          chunkEnd = static_cast<velox::vector_size_t>(
              1 +
              velox::bits::findSetBit(
                  reinterpret_cast<const char*>(nulls),
                  params.numScanned,
                  endRow,
                  remainingValues_));
        }
        numNulls = (chunkEnd - params.numScanned) - numNonNulls;
      }
      return std::lower_bound(
                 visitor.rows() + visitor.rowIndex(),
                 visitor.rows() + numRows,
                 chunkEnd) -
          visitor.rows();
    }
  }

  // Selects the encoding trait based on stringDecoderZeroCopy_ and
  // forwards to the fully-monomorphic readWithVisitorImpl.
  template <bool kHasNulls, typename V>
  void dispatchReadWithVisitorImpl(
      V& visitor,
      const uint64_t* nulls,
      ReadWithVisitorParams& params,
      const SaveStringBufferCallback& saveStringBuffersFn) {
    if (stringDecoderZeroCopy_) {
      readWithVisitorImpl<kHasNulls, DefaultEncodingTrait>(
          visitor, nulls, params, saveStringBuffersFn);
    } else {
      readWithVisitorImpl<kHasNulls, legacy::LegacyEncodingTrait>(
          visitor, nulls, params, saveStringBuffersFn);
    }
  }

  template <bool kHasNulls, typename EncodingTrait, typename V>
  void readWithVisitorImpl(
      V& visitor,
      const uint64_t* nulls,
      ReadWithVisitorParams& params,
      const SaveStringBufferCallback& saveStringBuffersFn) {
    constexpr bool kExtractToReader = std::
        is_same_v<typename V::Extract, velox::dwio::common::ExtractToReader>;
    const auto numRows = visitor.numRows();

    std::vector<velox::BufferPtr> stringBuffers;

    while (visitor.rowIndex() < numRows) {
      if constexpr (!kHasNulls) {
        if (FOLLY_UNLIKELY(remainingValues_ == 0)) {
          loadNextChunk();
        }
      }
      velox::vector_size_t numNulls;
      velox::vector_size_t numNonNulls;
      const auto endRowIndex = computeNextRowIndex<kHasNulls>(
          visitor, nulls, params, numRows, numNulls, numNonNulls);

      VLOG(1) << visitor.reader().fileType().fullName() << ":"
              << " rowIndex=" << visitor.rowIndex()
              << " endRowIndex=" << endRowIndex << " numNulls=" << numNulls
              << " numNonNulls=" << numNonNulls
              << " remainingValues_=" << remainingValues_;
      if (visitor.rowIndex() < endRowIndex) {
        visitor.setRows(velox::RowSet(visitor.rows(), endRowIndex));
        if (numNonNulls > 0) {
          EncodingTrait::callReadWithVisitor(*encoding_, visitor, params);
          // Some encodings (e.g., Constant via readWithVisitorSlow) do not
          // update numValues when DropValues (filter-only) is used. Sync
          // numValues with outputRows so that subsequent chunks whose
          // encodings (e.g., RLE bulkScan) use numValues as a base for
          // setNumRows do not truncate outputRows.
          if constexpr (V::kFilterOnly) {
            visitor.reader().setNumValues(visitor.reader().outputRows().size());
          }
        } else if (!visitor.allowNulls()) {
          visitor.setRowIndex(endRowIndex);
        } else {
          bool emitNullsRowByRow = true;
          if constexpr (kExtractToReader) {
            params.prepareResultNulls();
            if (visitor.reader().returnReaderNulls()) {
              auto numValues = endRowIndex - visitor.rowIndex();
              visitor.setRowIndex(endRowIndex);
              visitor.addNumValues(numValues);
              emitNullsRowByRow = false;
            }
          }
          if (emitNullsRowByRow) {
            bool atEnd = false;
            while (!atEnd) {
              visitor.processNull(atEnd);
            }
          }
        }
        NIMBLE_CHECK_EQ(visitor.rowIndex(), endRowIndex);
      }
      advancePosition(numNonNulls);
      params.numScanned += numNulls + numNonNulls;

      // Accumulate string buffers from each chunk. For non-string types,
      // currentStringBuffers_ will be empty.
      // Note: we can over queue the current string buffers by exactly
      // once, but that doesn't change the life cycle of the buffers for now.
      // Keeping this pattern for simplicity.
      for (const auto& buf : currentStringBuffers_) {
        stringBuffers.push_back(buf);
      }
    }

    // Flush accumulated string buffers to the reader.
    if (stringDecoderZeroCopy_) {
      if (saveStringBuffersFn) {
        saveStringBuffersFn(std::move(stringBuffers));
      } else {
        visitor.reader().setStringBuffers(std::move(stringBuffers));
      }
    }
  }

  // Returns true if the input stream has at least one more chunk header
  // available, indicating that loadNextChunk() can be called.
  bool hasMoreChunks() {
    return ensureInput(kChunkHeaderSize);
  }

  // Ensures that the input buffer has at least 'size' bytes available for
  // reading. If the current buffer doesn't have enough data, this method will
  // read more data from the underlying input stream until the requirement is
  // met or the stream is exhausted.
  // Returns true if the input buffer has at least 'size' bytes available,
  // false if the end of stream is reached before meeting the requirement.
  bool ensureInput(int size);

  // TODO: remove with row size estimate hacks.
  bool ensureInputIncremental_hack(int size, const char*& pos);

  // Loads the next chunk from the input stream and creates a new encoding.
  //
  // @param onChunkLoaded Callback invoked after loading. Returns true to
  //   continue reading, false to stop (e.g., the new chunk is not
  //   dictionary-compatible). Defaults to kNoopAtChunkBoundary.
  bool loadNextChunk(
      const ChunkBoundaryCallback& onChunkLoaded = kNoopAtChunkBoundary);

  void prepareInputBuffer(int32_t size);

  // Returns true if inputData_ points into inputBuffer_.
  bool fromInputBuffer() const {
    if (!inputBuffer_) {
      return false;
    }
    const char* bufStart = inputBuffer_->as<char>();
    return inputData_ >= bufStart &&
        inputData_ < bufStart + inputBuffer_->capacity();
  }

  // Seek to a specific chunk by offset.
  // Positions the decoder at the beginning of the chunk at the given offset.
  // The caller is responsible for updating rowPosition_ after this call.
  // @param offset The byte offset of the chunk in the stream
  void seekToChunk(
      uint32_t offset,
      const ChunkBoundaryCallback& onChunkBoundary = kNoopAtChunkBoundary);

  // Advances the position by the given number of values.
  // Updates both remainingValues_ and currentRow_.
  inline void advancePosition(int64_t numValues) {
    remainingValues_ -= numValues;
    NIMBLE_CHECK_GE(remainingValues_, 0);
    rowPosition_ += numValues;
    NIMBLE_CHECK_LE(
        rowPosition_,
        streamRowCount_.value_or(std::numeric_limits<uint32_t>::max()));
  }

  void skipWithIndex(
      int64_t numValues,
      const ChunkBoundaryCallback& onChunkBoundary);

  void skipWithoutIndex(
      int64_t numValues,
      const ChunkBoundaryCallback& onChunkBoundary);

  // Creates and caches the shared dictionary alphabet the first time a shared
  // dictionary chunk is loaded.
  void ensureSharedDictionaryAlphabet();

  // Returns true if this stream has a shared dictionary alphabet available or
  // pending lazy load.
  bool hasSharedDictionaryBinding() const {
    return dictionaryAlphabet_ != nullptr ||
        dictionaryAlphabetLoader_ != nullptr;
  }

  const std::unique_ptr<velox::dwio::common::SeekableInputStream> input_;
  velox::memory::MemoryPool* const pool_;
  // When true, decode nullable values (for array/map length streams that
  // encode nulls alongside values). When false, decode values without nulls
  // (standard case for scalar types).
  const bool decodeValuesWithNulls_;
  const EncodingFactory* const encodingFactory_;
  const bool stringDecoderZeroCopy_{false};
  // Optional stream index for accelerating skip operations
  const std::shared_ptr<index::StreamIndex> streamIndex_;
  // Total row count in the stream, set from stream index if available.
  const std::optional<uint32_t> streamRowCount_;

  // Pointer to the current position in the input buffer.
  // Points to the next byte to be read from the stream.
  const char* inputData_{nullptr};
  // Number of bytes remaining in the input buffer starting from inputData_.
  int64_t inputSize_{0};
  velox::BufferPtr inputBuffer_;
  velox::BufferPtr decompressed_;
  velox::BufferPtr nullableValues_;
  std::vector<velox::BufferPtr> currentStringBuffers_;

  std::unique_ptr<Encoding> encoding_;
  int64_t remainingValues_{0};
  mutable std::optional<size_t> rowCountEstimate_{std::nullopt};
  mutable std::optional<size_t> stringDataSizeEstimate_{std::nullopt};

  // Tracks the current row position in the stream.
  uint32_t rowPosition_{0};

  // Per-column decoding statistics. Owned by SplitStats; valid for
  // the lifetime of this decoder.
  velox::dwio::common::DecodingStats* const decodingStats_{nullptr};
  // Lazily resolves the alphabet bound to this stream. Reset after
  // dictionaryAlphabet_ is loaded.
  DictionaryAlphabetLoader dictionaryAlphabetLoader_;
  // Cached alphabet returned by dictionaryAlphabetLoader_.
  std::shared_ptr<const SharedDictionaryAlphabet> dictionaryAlphabet_;
  friend class ChunkedDecoderTestHelper;
};

} // namespace facebook::nimble
