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

#include "velox/dwio/nimble/velox/selective/StringColumnReader.h"

#include <folly/ScopeGuard.h>
#include <algorithm>
#include "velox/common/base/SimdUtil.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/ColumnVisitors.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingUtils.h"
#include "velox/dwio/nimble/velox/selective/NimbleData.h"
#include "velox/vector/DictionaryVector.h"

namespace facebook::nimble {

using namespace facebook::velox;

bool StringColumnReader::dictionaryPreservable() {
  return scanSpec_->valueHook() == nullptr &&
      formatData().stringDecoderZeroCopy() &&
      static_cast<const NimbleData&>(formatData())
          .nimblePreserveDictionaryEncoding();
}

uint64_t StringColumnReader::skip(uint64_t numValues) {
  numValues = SelectiveColumnReader::skip(numValues);
  // Clear cached dictionary state when skip crosses a chunk boundary,
  // since the new chunk may have a different alphabet.
  decoder_.skip(numValues, [this] {
    clearDictionaryState();
    return true;
  });
  return numValues;
}

void StringColumnReader::ensureDictionaryState() {
  if (hasDictionaryState()) {
    return;
  }
  dictionaryState_.alphabet = buildEncodingDictionaryAlphabet<std::string_view>(
      decoder_.currentEncoding());
}

void StringColumnReader::ensureFilterCache() {
  using FilterResult = velox::dwio::common::FilterResult;
  const auto alphabetSize = dictionaryState_.alphabet.size();
  const auto oldSize = dictionaryState_.filterCache.size();
  // The filter cache size and the merged-alphabet (dictionary) size increase
  // monotonically: the alphabet only grows by appending within a read and the
  // cache extends to match; on a chunk boundary both are cleared and rebuilt
  // together (the cache is never reused across a different chunk), and the
  // reader is rebuilt per stripe. So oldSize never exceeds alphabetSize — it is
  // either equal (same alphabet reused) or smaller (alphabet grew).
  NIMBLE_CHECK_LE(oldSize, alphabetSize);
  if (oldSize >= alphabetSize) {
    return;
  }
  dictionaryState_.filterCache.resize(alphabetSize);
  velox::simd::memset(
      dictionaryState_.filterCache.data() + oldSize,
      static_cast<uint8_t>(FilterResult::kUnknown),
      static_cast<int32_t>(alphabetSize - oldSize));
}

namespace {

// Compacts the non-null (index, row) pairs out of a materialized run. Writes
// the dictionary index and file row of each non-null position into
// outIndices/outRows (caller-sized to at least numValues) and returns the
// non-null count. A null 'nulls' bitmap means no nulls, so every entry is
// copied. Pure: touches no reader member state.
//
// 'indices'/'rows'/'nulls' are consumed only over [0, numValues). The caller
// passes the full read-range 'rows', which on the abandon path is longer than
// numValues (the materialized dictionary prefix); positions >= numValues belong
// to the non-dict chunk and are never read here.
int32_t extractNonNullEntries(
    const int32_t* indices,
    const velox::RowSet& rows,
    const uint64_t* nulls,
    velox::vector_size_t numValues,
    int32_t* outIndices,
    int32_t* outRows) {
  int32_t count = 0;
  for (velox::vector_size_t i = 0; i < numValues; ++i) {
    if (nulls != nullptr && velox::bits::isBitNull(nulls, i)) {
      continue;
    }
    outIndices[count] = indices[i];
    outRows[count] = rows[i];
    ++count;
  }
  return count;
}

// Filters a run of materialized dictionary indices through the merged alphabet
// and the per-index filterCache. Writes the passing indices to outputIndices
// and their rows to outputRows (both caller-sized, compacted to the survivors),
// records each newly resolved verdict into filterCache (kSuccess/kFailure), and
// returns the number of passing rows.
int32_t filterByCache(
    const int32_t* inputIndices,
    const int32_t* inputRows,
    int32_t numInput,
    const std::vector<std::string_view>& alphabet,
    const velox::common::Filter& filter,
    uint8_t* filterCache,
    int32_t* outputIndices,
    int32_t* outputRows) {
  // Delegate to the shared SIMD dictionary-filter kernel. Nimble's merged
  // alphabet is single-tier, so the per-index test resolves the dictionary
  // entry directly and applies the byte filter. The Nimble path always emits
  // both indices and rows, hence kFilterOnly=false.
  return velox::dwio::common::filterDictionaryRunSimd</*kFilterOnly=*/false>(
      inputIndices,
      numInput,
      inputRows,
      filterCache,
      outputRows,
      outputIndices,
      /*numValues=*/0,
      [&](int32_t dictIndex) {
        const auto& sv = alphabet[dictIndex];
        return filter.testBytes(sv.data(), static_cast<int32_t>(sv.size()));
      });
}

} // namespace

// TODO(T280425209): Delete this method (and its two call sites in
// filterDictionaryIndices) once these guards have baked in prod without firing
// — the fixup they replaced is dead on the dict-filter path.
void StringColumnReader::checkWritableResultNulls(velox::vector_size_t size) {
  // Invariant guard, not fixup. Picking the null source by read shape (dense:
  // nullsInReadRange_; sparse: resultNulls_) means the dict-filter path never
  // needs to convert the dense reader-nulls fast path into a compacted buffer:
  // the suppressed-filter bulk read leaves hasFilter() cached true, so
  // setReturnNullsMode always resolves returnReaderNulls_ to false here, and
  // prepareResultNulls has already allocated a unique, output-indexed
  // resultNulls_ sized for the read range. These are NIMBLE_CHECKs (prod
  // signals, not DCHECK): a fire means that invariant regressed and the
  // buffer-fixup this method used to perform must be restored before compacted
  // nulls get written into the wrong (fast-path) buffer.
  NIMBLE_CHECK(
      !returnReaderNulls_,
      "dict-filter path must not run on the dense reader-nulls fast path");
  NIMBLE_CHECK_NOT_NULL(resultNulls_);
  NIMBLE_CHECK_NOT_NULL(rawResultNulls_);
  NIMBLE_CHECK(
      resultNulls_->unique(),
      "resultNulls_ must be uniquely owned before writing compacted nulls");
  NIMBLE_CHECK_GE(
      resultNulls_->capacity(),
      velox::bits::nbytes(size) + velox::simd::kPadding);
}

void StringColumnReader::filterDictionaryIndices(
    const RowSet& rows,
    const velox::common::Filter* filter) {
  NIMBLE_CHECK_NOT_NULL(filter);
  ensureFilterCache();

  // readCount bounds every pass below to the dictionary indices actually
  // materialized into rawValues_. On the abandon path this is a prefix of the
  // read range: 'rows' extends past readCount into the non-dict chunk, but
  // those rows carry no dictionary index here (they are filtered by the flat
  // continuation in read()), so extractNonNullEntries/processNullAndPassingRows
  // only ever touch rawValues_/rows/nulls over [0, readCount).
  const auto readCount = numValues_;
  constexpr int32_t kSimdPadding = xsimd::batch<int32_t>::size;
  std::vector<int32_t> nonNullIndices(readCount + kSimdPadding, 0);
  std::vector<int32_t> nonNullRows(readCount + kSimdPadding, 0);
  // A dense read is exactly rows == [0, rows.size()) — the same shape
  // StringColumnReadWithVisitorHelper used to instantiate the dense/sparse
  // ColumnVisitor, so it matches where the bulk read stored the output nulls.
  // On the abandon path only rows[0, numValues_) were materialized, but the
  // null buffer was still chosen from the full range, so isDense is derived
  // from the full rows here as well.
  const bool isDense = !rows.empty() && rows.back() == rows.size() - 1;
  // The dict indices in rawValues_ are output-indexed, so the null check needs
  // an output-indexed bitmap. The deferred-filter bulk read leaves
  // anyNulls_/returnReaderNulls_ unset for the filtered read, so pick the
  // source by read shape rather than routing through resultNulls(): a dense
  // read keeps its nulls in nullsInReadRange_ (output == read range, so it is
  // output-aligned), while a sparse read has them in the resultNulls_ buffer
  // that readSparseMaterializedIndices built and compacted inline
  // (rawNullsInReadRange is file-indexed and wrong for the sparse case).
  const uint64_t* nulls;
  if (isDense) {
    nulls = rawNullsInReadRange();
  } else {
    const auto& resultNullsBuffer = resultNulls();
    nulls = resultNullsBuffer ? resultNullsBuffer->as<uint64_t>() : nullptr;
  }

  const auto numNonNull = extractNonNullEntries(
      reinterpret_cast<const int32_t*>(rawValues_),
      rows,
      nulls,
      readCount,
      nonNullIndices.data(),
      nonNullRows.data());

  if (nulls == nullptr) {
    // No nulls in range: filter the dictionary indices in place. There is no
    // null bitmap to realign.
    auto* outputRows = mutableOutputRows(numNonNull);
    numValues_ = filterByCache(
        nonNullIndices.data(),
        nonNullRows.data(),
        numNonNull,
        dictionaryState_.alphabet,
        *filter,
        dictionaryState_.filterCache.data(),
        reinterpret_cast<int32_t*>(rawValues_),
        outputRows);
    outputRows_.resize(numValues_);
    return;
  }

  // Nulls are present. The dict-filter path pre-compacts rawValues_/outputRows_
  // to the passing rows, which makes the framework's compactScalarValues a
  // no-op (rows.size() == numValues_) and skips its null move. So realign the
  // result null bitmap to the compacted output layout here; otherwise
  // getValues() would hand DictionaryVector a read-range-indexed bitmap against
  // compacted indices and misplace nulls.
  if (!filter->testNull()) {
    // The filter rejects nulls, so every null row is dropped and the
    // SIMD-filtered non-null passing rows are the entire output. Filter in
    // place, then mark the compacted output all-non-null.
    auto* outputRows = mutableOutputRows(numNonNull);
    numValues_ = filterByCache(
        nonNullIndices.data(),
        nonNullRows.data(),
        numNonNull,
        dictionaryState_.alphabet,
        *filter,
        dictionaryState_.filterCache.data(),
        reinterpret_cast<int32_t*>(rawValues_),
        outputRows);
    outputRows_.resize(numValues_);
    checkWritableResultNulls(readCount);
    // Clear through readCount, not numValues_: filterByCache compacted the
    // output, and the vacated tail [numValues_, readCount) still holds null
    // bits from the pre-compaction layout. On the dict->flat abandon path the
    // continuation appends over that tail with addValue(), which writes the
    // value but not the null bit, and ChunkedDecoder::readWithVisitor skips
    // prepareNulls() (and its buffer-wide clear) once the prefix has set
    // anyNulls_ — so nothing else would ever clear them.
    velox::bits::fillBits(rawResultNulls_, 0, readCount, velox::bits::kNotNull);
    return;
  }

  // The filter accepts nulls (e.g. IS NULL): retain every null row and merge
  // it, in ascending row order, with the SIMD-filtered non-null passing rows.
  // The non-null rows are filtered into temporaries because the merge writes
  // the interleaved result back into rawValues_/outputRows_ in row order.
  std::vector<int32_t> passIndices(numNonNull + kSimdPadding, 0);
  std::vector<int32_t> passRows(numNonNull + kSimdPadding, 0);
  const auto numPass = filterByCache(
      nonNullIndices.data(),
      nonNullRows.data(),
      numNonNull,
      dictionaryState_.alphabet,
      *filter,
      dictionaryState_.filterCache.data(),
      passIndices.data(),
      passRows.data());

  checkWritableResultNulls(readCount);
  auto* indices = reinterpret_cast<int32_t*>(rawValues_);
  auto* outputRows = mutableOutputRows(readCount);
  numValues_ = processNullAndPassingRows(
      nulls,
      rows,
      readCount,
      passIndices.data(),
      passRows.data(),
      numPass,
      indices,
      outputRows);
  outputRows_.resize(numValues_);
  // The merge wrote [0, numValues_) but compacted away the rest of
  // [0, readCount); clear the vacated tail for the same reason as the
  // reject-nulls branch above.
  velox::bits::fillBits(
      rawResultNulls_, numValues_, readCount, velox::bits::kNotNull);
}

velox::vector_size_t StringColumnReader::processNullAndPassingRows(
    const uint64_t* nulls,
    const velox::RowSet& rows,
    velox::vector_size_t readCount,
    const int32_t* passIndices,
    const int32_t* passRows,
    int32_t numPass,
    int32_t* indices,
    int32_t* outputRows) {
  velox::vector_size_t count = 0;
  int32_t passCursor = 0;
  for (velox::vector_size_t i = 0; i < readCount; ++i) {
    if (velox::bits::isBitNull(nulls, i)) {
      // Null row: passes because the filter accepts nulls. The index is
      // unused for a null position; write 0 as a safe placeholder. Record that
      // the output carries a null (the deferred-filter path removed the
      // reconciliation block that used to set anyNulls_).
      velox::bits::setNull(rawResultNulls_, count, /*isNull=*/true);
      anyNulls_ = true;
      indices[count] = 0;
      outputRows[count] = rows[i];
      ++count;
    } else if (passCursor < numPass && passRows[passCursor] == rows[i]) {
      // Non-null row that passed the byte filter.
      velox::bits::setNull(rawResultNulls_, count, /*isNull=*/false);
      indices[count] = passIndices[passCursor];
      outputRows[count] = rows[i];
      ++count;
      ++passCursor;
    }
  }
  return count;
}

void StringColumnReader::updateDictionaryIndices(
    vector_size_t alphabetSize,
    vector_size_t valueOffset) {
  if (alphabetSize > 0) {
    auto* values = reinterpret_cast<int32_t*>(rawValues_);
    for (auto i = valueOffset; i < numValues_; ++i) {
      values[i] += alphabetSize;
    }
  }
}

bool StringColumnReader::tryExtendDictionaryAtChunkBoundary(
    int32_t& alphabetOffset,
    vector_size_t& valueOffset) {
  // Offset the indices written by the previous chunk so they reference the
  // merged alphabet instead of the per-chunk alphabet. numValues_ is
  // incremented by the visitor (via process() / processNull()) as
  // readDictionaryIndices reads each row. At this point, numValues_ reflects
  // all rows read so far across chunks, and [valueOffset, numValues_) are
  // the indices from the just-completed chunk that need offsetting.
  updateDictionaryIndices(alphabetOffset, valueOffset);
  valueOffset = numValues_;

  // Check if the new chunk's encoding supports dictionary index reading.
  // If not, return false to signal the caller to abandon the dict path.
  if (!decoder_.dictionaryConvertible()) {
    return false;
  }

  // Extend the merged alphabet with the new chunk's dictionary entries.
  // alphabetOffset tracks the cumulative size so that the next chunk's
  // indices can be offset correctly.
  alphabetOffset = static_cast<int32_t>(dictionaryState_.alphabet.size());
  auto chunkAlphabet = buildEncodingDictionaryAlphabet<std::string_view>(
      decoder_.currentEncoding());
  dictionaryState_.alphabet.insert(
      dictionaryState_.alphabet.end(),
      chunkAlphabet.begin(),
      chunkAlphabet.end());
  dictionaryState_.alphabetVector.reset();
  crossChunkRead_ = true;
  return true;
}

bool StringColumnReader::readWithDictionary(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  // Contract: read() only enters the dictionary path when the dictionary
  // encoding is preservable — no value hook, the zero-copy (non-legacy)
  // encoding path, and preserve-dictionary enabled. Check it here so a caller
  // that forgets the gate fails loudly instead of silently mis-reading a
  // legacy/hook column through the (zero-copy) dictionary machinery.
  NIMBLE_CHECK(
      dictionaryPreservable(),
      "readWithDictionary entered when the dictionary is not preservable");

  // A cross-chunk merged alphabet is per-read: getValues() clears it (and snaps
  // its backing into the output DictionaryVector) once the read is consumed.
  // But the parent struct reader skips this column's getValues() when it
  // filters out all rows in a batch, leaving crossChunkRead_ set and a stale
  // merged alphabet whose backing buffers the next read's setStringBuffers()
  // will free. Clear it before reuse so ensureDictionaryState() rebuilds a
  // fresh, valid alphabet. Single-chunk reads keep their cached alphabet
  // (crossChunkRead_ stays false).
  if (crossChunkRead_) {
    clearDictionaryState();
    crossChunkRead_ = false;
  }

  prepareRead<int32_t>(offset, rows, incomingNulls);

  // Load the chunk that will actually be read, AFTER prepareRead's seekTo ->
  // skip. The skip advances the value decoder and can leave a stale chunk
  // current: the previous batch's exhausted chunk (no skip needed), or, when
  // the skip lands exactly on a chunk boundary, the last chunk it skipped
  // through. Loading here rather than before prepareRead ensures the
  // convertibility check below inspects the chunk that will be read, not a
  // stale one. The callback clears stale dictionary state so
  // ensureDictionaryState rebuilds from the new chunk.
  decoder_.ensureLoaded([this] {
    clearDictionaryState();
    return true;
  });

  const auto endReadRow = rows.back() + 1;
  // Check convertibility once, on that chunk. Only a genuinely non-dictionary
  // encoding (e.g. Trivial) fails it. When it is not convertible, no dictionary
  // indices were materialized; re-type the value buffer to StringView
  // (abandonDictionaryEncoding handles numValues_ == 0) so read()'s flat
  // continuation fills it. Crucially, read() must NOT run a second prepareRead
  // in this case — that would advance the null/in-map decoders a second time
  // and corrupt flatmap reads.
  if (!decoder_.dictionaryConvertible()) {
    abandonDictionaryEncoding(endReadRow);
    return false;
  }
  ensureDictionaryState();
  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::StringColumnReader::readWithDictionary",
      &dictionaryState_.alphabet);
  // Set to true when the dictionary read is abandoned at a non-dict chunk;
  // false when the whole range is read in dictionary mode. On abandon,
  // readDictionaryIndices has already advanced readOffset_ to the decoder's
  // parked position (the chunk boundary), where the flat fallback resumes.
  bool abandonDictionary = false;

  // Read dictionary indices, potentially across multiple chunks. At each
  // chunk boundary, onChunkBoundary checks if the new chunk is also
  // dictionary-convertible:
  //   - Returns true: the new chunk's alphabet is appended to the merged
  //     alphabet and reading continues in dict mode.
  //   - Returns false: the new chunk is not dict-compatible, so
  //     readDictionaryIndices stops and the reactive flat fallback below
  //     takes over.
  //
  // Save and suppress the filter during index reading so the bulk
  // materializeIndices path runs (not approach C's per-row process path).
  // The SIMD filterDictionaryIndices applies the filter post-hoc.
  std::unique_ptr<velox::common::Filter> savedFilter;
  if (scanSpec_->hasFilter()) {
    savedFilter = scanSpec_->filter()->clone();
    scanSpec_->setFilter(nullptr);
  }
  // If the read below throws, reinstall the suppressed filter so scanSpec_ is
  // not left permanently filterless, which would silently drop the pushed-down
  // filter on subsequent reads.
  auto filterGuard = folly::makeGuard([&] {
    if (savedFilter != nullptr) {
      scanSpec_->setFilter(std::move(savedFilter));
    }
  });

  // kEncodingHasNulls must be false: Nimble materializes nulls lazily during
  // decode and does not pre-populate nullsInReadRange_.
  dwio::common::StringColumnReadWithVisitorHelper<
      /*kEncodingHasNulls=*/false,
      /*kDictionary=*/false>(*this, rows)([&](auto visitor) {
    auto dictVisitor = visitor.toStringDictionaryColumnVisitor();

    int32_t alphabetOffset = 0;
    velox::vector_size_t valueOffset = numValues_;

    // At each chunk boundary, try to extend the merged alphabet with the
    // new chunk's dictionary. Returns true if the new chunk is
    // dict-compatible (alphabet extended, continue reading). Returns false
    // if the new chunk is not dict-compatible (stop, fall back to flat).
    auto onChunkBoundary = [&]() -> bool {
      return tryExtendDictionaryAtChunkBoundary(alphabetOffset, valueOffset);
    };

    // readDictionaryIndices returns false when the onChunkBoundary callback
    // returns false (new chunk is not dict-compatible), meaning the
    // dictionary path must be abandoned for the remaining rows.
    abandonDictionary =
        !decoder_.readDictionaryIndices(dictVisitor, onChunkBoundary);

    // Offset the final chunk's indices into the merged alphabet.
    updateDictionaryIndices(alphabetOffset, valueOffset);
  });
  // Reinstate the filter before the post-hoc SIMD filtering below needs it; the
  // guard then no-ops since savedFilter has been moved out.
  filterGuard.dismiss();
  if (savedFilter != nullptr) {
    scanSpec_->setFilter(std::move(savedFilter));
  }

  // Apply the filter to the just-materialized dict indices BEFORE the abandon
  // check so the dict portion is filtered on both paths: the normal path
  // (getValues) and the dict->flat fallback (abandonDictionaryEncoding).
  // filterDictionaryIndices derives its output-indexed null source from the
  // read shape (dense: nullsInReadRange_; sparse: resultNulls_). The no-filter
  // projected path needs no reconciliation — prepareResultNulls (called during
  // the bulk read) already established anyNulls_/returnReaderNulls_ via
  // setReturnNullsMode, since hasFilter() is genuinely false there (not
  // suppressed).
  if (scanSpec_->hasFilter()) {
    filterDictionaryIndices(rows, scanSpec_->filter());
  }

  if (!abandonDictionary) {
    readOffset_ += endReadRow;
    return true;
  }

  // Partial dict read: a non-dict chunk was encountered mid-read. Convert
  // the already-filtered dict indices to flat StringViews
  // (filterDictionaryIndices above ran for this path too). readOffset_ already
  // points at the chunk boundary (set by readDictionaryIndices), so read()
  // resumes the flat read there and slices the remaining rows past it.
  abandonDictionaryEncoding(endReadRow);
  return false;
}

void StringColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  // numReadRows is how many rows the dictionary portion produced before
  // abandoning: 0 when the dictionary is not preservable or the landed chunk
  // was non-convertible, or the chunk boundary when the dict path was abandoned
  // mid-read. It selects a fresh flat read vs a flat continuation below.
  velox::vector_size_t numReadRows = 0;
  // The dictionary encoding is preservable only without a value hook, on the
  // zero-copy (non-legacy) encoding path, and with preserve-dictionary enabled.
  // Deciding here (rather than inside readWithDictionary) lets
  // readWithDictionary own the single prepareRead: when it abandons it re-types
  // the value buffer via abandonDictionaryEncoding, so read() never issues a
  // second prepareRead — which would advance the null/in-map decoders twice and
  // corrupt flatmap reads.
  if (dictionaryPreservable()) {
    if (readWithDictionary(offset, rows, incomingNulls)) {
      return;
    }
    numReadRows = readOffset_ > offset
        ? static_cast<velox::vector_size_t>(readOffset_ - offset)
        : 0;
  } else {
    // Dictionary not preservable: full flat read. Clearing the dictionary state
    // converts the reader back to flat read mode, dropping anything a
    // dictionary-mode read (readWithDictionary) may have prepared, so this read
    // materializes plain StringViews.
    clearDictionaryState();
    prepareRead<std::string_view>(offset, rows, incomingNulls);
  }

  const auto endReadRow = rows.back() + 1;
  NIMBLE_CHECK_LE(numReadRows, endReadRow);

  // Always read against the complete original rows so the visitor — and hence
  // null prep — sees the full output range. On a continuation after an
  // abandoned dict read, numReadRows is the chunk boundary where the decoder is
  // parked; advance the visitor to the first row at/after it so the flat read
  // resumes there (the dict portion was already produced) instead of re-reading
  // the prefix.
  velox::vector_size_t startRowIndex = 0;
  if (numReadRows > 0) {
    startRowIndex = static_cast<velox::vector_size_t>(
        std::lower_bound(rows.data(), rows.data() + rows.size(), numReadRows) -
        rows.data());
  }

  dwio::common::StringColumnReadWithVisitorHelper<
      /*kEncodingHasNulls=*/false,
      /*kDictionary=*/false>(*this, rows)([&](auto visitor) {
    if (numReadRows == 0) {
      decoder_.readWithVisitor(visitor);
    } else {
      // Resume the flat read at the abandoned chunk boundary: skip the rows the
      // dict portion already produced, then append the new string buffers to
      // the existing ones (which include the dict portion's buffer from
      // abandonDictionaryEncoding).
      visitor.setRowIndex(startRowIndex);
      decoder_.readWithVisitor(
          visitor,
          /*readOffset=*/numReadRows,
          [this](std::vector<velox::BufferPtr>&& buffers) {
            for (auto& buffer : buffers) {
              stringBuffers_.push_back(std::move(buffer));
            }
          });
    }
  });

  readOffset_ = offset + endReadRow;
}

void StringColumnReader::ensureAlphabetVector() {
  if (dictionaryState_.alphabetVector != nullptr) {
    return;
  }
  const auto alphabetSize =
      static_cast<vector_size_t>(dictionaryState_.alphabet.size());
  auto values = AlignedBuffer::allocate<StringView>(alphabetSize, pool_);
  auto* rawValues = values->asMutable<StringView>();
  for (vector_size_t i = 0; i < alphabetSize; ++i) {
    rawValues[i] = StringView(
        dictionaryState_.alphabet[i].data(),
        dictionaryState_.alphabet[i].size());
  }
  auto buffers = stringBuffers_;
  dictionaryState_.alphabetVector = std::make_shared<FlatVector<StringView>>(
      pool_,
      fileType_->type(),
      BufferPtr(nullptr),
      alphabetSize,
      std::move(values),
      std::move(buffers));
}

void StringColumnReader::abandonDictionaryEncoding(vector_size_t endReadRow) {
  // numValues_ == 0 means the dictionary path was abandoned before
  // materializing any indices (the chunk read was not convertible at the
  // read's start), so there is no alphabet and nothing to resolve — the loop
  // below is a no-op and this only re-types the value buffer to StringView for
  // the flat continuation.
  NIMBLE_CHECK(numValues_ == 0 || !dictionaryState_.alphabet.empty());

  // Resolve dictionary indices to flat StringViews. The underlying string
  // data lives in the encoding's string buffers, which the reader already
  // holds via stringBuffers_ (set by readDictionaryIndices). No string
  // byte copying is needed. Null positions contain uninitialized indices
  // and are skipped — they get StringView() (empty) by the allocate default.
  const auto* indices = reinterpret_cast<const int32_t*>(rawValues_);
  auto flatValues =
      AlignedBuffer::allocate<StringView>(endReadRow, pool_, StringView());
  auto* rawFlatValues = flatValues->asMutable<StringView>();
  // The dict indices in rawValues_ are at output positions [0, numValues_).
  // resultNulls() is the single source of truth for the output-indexed null
  // bitmap: it selects nullsInReadRange_ for dense reads and resultNulls_ for
  // sparse, and returns an empty buffer when there are no nulls. Use it
  // directly rather than re-deriving the selection here.
  const auto& resultNullsBuffer = resultNulls();
  const auto* nulls =
      resultNullsBuffer ? resultNullsBuffer->as<uint64_t>() : nullptr;

  for (vector_size_t i = 0; i < numValues_; ++i) {
    if (nulls != nullptr && velox::bits::isBitNull(nulls, i)) {
      continue;
    }
    const auto idx = indices[i];
    NIMBLE_DCHECK_LT(idx, dictionaryState_.alphabet.size());
    const auto& sv = dictionaryState_.alphabet[idx];
    rawFlatValues[i] = StringView(sv.data(), sv.size());
  }

  values_ = std::move(flatValues);
  rawValues_ = values_->asMutable<char>();

  // prepareRead<int32_t> set valueSize_=4 for dict indices. After
  // converting to StringView values, update valueSize_ so that the
  // flat encoding fallback's addNull/addValue writes at correct byte offsets.
  valueSize_ = sizeof(StringView);
  clearDictionaryState();
  crossChunkRead_ = false;
}

void StringColumnReader::getValues(const RowSet& rows, VectorPtr* result) {
  if (!hasDictionaryState()) {
    rawStringBuffer_ = nullptr;
    rawStringSize_ = 0;
    rawStringUsed_ = 0;
    getFlatValues<StringView, StringView>(rows, result, fileType_->type());
    return;
  }

  compactScalarValues<int32_t, int32_t>(rows, /*isFinal=*/false);
  ensureAlphabetVector();

  *result = std::make_shared<DictionaryVector<StringView>>(
      pool_,
      resultNulls(),
      numValues_,
      dictionaryState_.alphabetVector,
      values_);
  numValues_ = 0;
  // If this batch crossed chunk boundaries, clear the merged alphabet
  // to free memory from previous chunks that are no longer needed for
  // the next read range. Single-chunk batches keep the cached alphabet
  // for reuse. The DictionaryVector holds a shared_ptr to
  // alphabetVector, so the data survives after clearing.
  if (crossChunkRead_) {
    clearDictionaryState();
    crossChunkRead_ = false;
  }
}

bool StringColumnReader::estimateMaterializedSize(
    size_t& byteSize,
    size_t& rowCount) const {
  auto decoderRowCount = decoder_.estimateRowCount();
  if (!decoderRowCount.has_value()) {
    return false;
  }
  auto decoderDataSize = decoder_.estimateStringDataSize();
  if (!decoderDataSize.has_value()) {
    return false;
  }
  rowCount = *decoderRowCount;
  auto rowSize = *decoderDataSize / rowCount;
  rowSize += (rowSize > velox::StringView::kInlineSize) ? 16 : 4;
  byteSize = rowSize * rowCount;
  return true;
}

} // namespace facebook::nimble
