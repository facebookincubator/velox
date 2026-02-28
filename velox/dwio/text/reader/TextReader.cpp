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

#include "velox/dwio/text/reader/TextReader.h"

#include <fast_float/fast_float.h>
#include <string>

#include "velox/common/base/BitUtil.h"
#include "velox/common/encode/Base64.h"
#include "velox/dwio/common/exception/Exceptions.h"
#include "velox/type/Filter.h"
#include "velox/type/fbhive/HiveTypeParser.h"

namespace facebook::velox::text {
namespace {

using common::CompressionKind;

using dwio::common::EOFError;
using dwio::common::RowReader;

template <typename Filter, typename T>
inline bool testFilter(const velox::common::Filter* filter, T value) {
  if constexpr (std::is_same_v<Filter, velox::common::AlwaysTrue>) {
    return true;
  } else {
    return velox::common::applyFilter(
        const_cast<Filter&>(static_cast<const Filter&>(*filter)), value);
  }
}

template <typename Filter>
inline bool testFilterNull(const velox::common::Filter* filter) {
  if constexpr (std::is_same_v<Filter, velox::common::AlwaysTrue>) {
    return true;
  } else {
    return filter->testNull();
  }
}

static constexpr std::string_view kTextfileCompressionExtensionGzip{".gz"};
static constexpr std::string_view kTextfileCompressionExtensionDeflate{
    ".deflate"};
static constexpr std::string_view kTextfileCompressionExtensionZst{".zst"};
static constexpr std::string_view kTextfileCompressionExtensionLz4{".lz4"};
static constexpr std::string_view kTextfileCompressionExtensionLzo{".lzo"};
static constexpr std::string_view kTextfileCompressionExtensionSnappy{
    ".snappy"};

constexpr const int32_t kDecompressionBufferFactor = 3;

void resizeVector(
    BaseVector* FOLLY_NULLABLE data,
    const vector_size_t insertionIdx) {
  if (data == nullptr) {
    return;
  }

  auto dataSize = data->size();
  if (dataSize == 0) {
    data->resize(10);
  } else if (dataSize <= insertionIdx) {
    if (data->type()->kind() == TypeKind::ARRAY) {
      auto oldSize = dataSize;
      auto newSize = dataSize * 2;
      data->resize(newSize);

      auto arrayVector = data->asUnchecked<ArrayVector>();
      auto rawOffsets = arrayVector->offsets()->asMutable<vector_size_t>();
      auto rawSizes = arrayVector->sizes()->asMutable<vector_size_t>();

      auto lastOffset = oldSize > 0 ? rawOffsets[oldSize - 1] : 0;
      auto lastSize = oldSize > 0 ? rawSizes[oldSize - 1] : 0;
      auto newOffset = oldSize > 0 ? lastOffset + lastSize : 0;

      for (auto i = oldSize; i < newSize; ++i) {
        rawSizes[i] = 0;
        rawOffsets[i] = newOffset;
      }
    } else if (data->type()->kind() == TypeKind::MAP) {
      auto oldSize = dataSize;
      auto newSize = dataSize * 2;
      data->resize(newSize);

      auto mapVector = data->asUnchecked<MapVector>();
      auto rawOffsets = mapVector->offsets()->asMutable<vector_size_t>();
      auto rawSizes = mapVector->sizes()->asMutable<vector_size_t>();

      auto lastOffset = oldSize > 0 ? rawOffsets[oldSize - 1] : 0;
      auto lastSize = oldSize > 0 ? rawSizes[oldSize - 1] : 0;
      auto newOffset = oldSize > 0 ? lastOffset + lastSize : 0;

      for (auto i = oldSize; i < newSize; ++i) {
        rawSizes[i] = 0;
        rawOffsets[i] = newOffset;
      }
    } else {
      data->resize(dataSize * 2);
    }
  }
}

void setCompressionSettings(
    const std::string& filename,
    CompressionKind& kind,
    dwio::common::compression::CompressionOptions& compressionOptions) {
  if (filename.ends_with(kTextfileCompressionExtensionLz4) ||
      filename.ends_with(kTextfileCompressionExtensionLzo) ||
      filename.ends_with(kTextfileCompressionExtensionSnappy)) {
    VELOX_FAIL("Unsupported compression extension for file: {}", filename);
  }
  if (filename.ends_with(kTextfileCompressionExtensionGzip)) {
    kind = CompressionKind::CompressionKind_GZIP;
    compressionOptions.format.zlib.windowBits =
        15; // 2^15-byte deflate window size
  } else if (filename.ends_with(kTextfileCompressionExtensionDeflate)) {
    kind = CompressionKind::CompressionKind_ZLIB;
    compressionOptions.format.zlib.windowBits =
        -15; // raw deflate, 2^15-byte window size
  } else if (filename.ends_with(kTextfileCompressionExtensionZst)) {
    kind = CompressionKind::CompressionKind_ZSTD;
  } else {
    kind = CompressionKind::CompressionKind_NONE;
  }
}

} // namespace

FileContents::FileContents(
    MemoryPool& pool,
    const std::shared_ptr<const RowType>& t)
    : schema{t},
      input{nullptr},
      pool{pool},
      fileLength{0},
      compression{CompressionKind::CompressionKind_NONE},
      compressionOptions{} {}

TextRowReader::TextRowReader(
    std::shared_ptr<FileContents> fileContents,
    const RowReaderOptions& opts)
    : RowReader(),
      contents_{fileContents},
      scanSpec_{opts.scanSpec()},
      options_{opts},
      currentRow_{0},
      pos_{opts.offset()},
      atEOL_{false},
      atEOF_{false},
      atSOL_{false},
      atPhysicalEOF_{false},
      depth_{0},
      unreadIdx_{0},
      limit_{opts.limit()},
      fileLength_{getStreamLength()},
      varBinBuf_{
          std::make_shared<dwio::common::DataBuffer<char>>(contents_->pool)} {
  initializeColumnReaders();
  // Seek to first line at or after the specified region.
  if (contents_->compression == CompressionKind::CompressionKind_NONE) {
    // TODO: Inconsistent row skipping behavior (kept for Presto compatibility)
    // Issue: When reading from byte offset > 0, we skip rows inclusively at the
    // start position, but when reading from byte 0, no rows are skipped. This
    // creates inconsistent behavior where a row at the boundary may be skipped
    // when it should be included.
    //
    // Example: If pos_ = 10 is the first byte of row 2, that entire row gets
    // skipped, even though it should be read.
    //
    // Proposed fix: streamPosition_ = (pos_ == 0) ? 0 : --pos_;
    // This would skip rows exclusively of pos_, ensuring consistent behavior.
    const auto streamPosition_ = pos_;

    contents_->inputStream = contents_->input->read(
        streamPosition_,
        contents_->fileLength - streamPosition_,
        dwio::common::LogType::STREAM);

    if (pos_ != 0) {
      unreadData_.clear();
      skipLine();
    }
    if (opts.skipRows() > 0) {
      seekToRow(opts.skipRows());
    }
  } else {
    // compressed text files, the first split reads the whole file, rest read 0
    if (pos_ != 0) {
      atEOF_ = true;
    }
    limit_ = std::numeric_limits<uint64_t>::max();

    contents_->inputStream = contents_->input->loadCompleteFile();
    auto name = contents_->inputStream->getName();
    contents_->decompressedInputStream = createDecompressor(
        contents_->compression,
        std::move(contents_->inputStream),
        // An estimated value used as the output buffer size for the zlib
        // decompressor, and as the fallback value of the decompressed length
        // for other decompressors.
        kDecompressionBufferFactor * contents_->fileLength,
        contents_->pool,
        contents_->compressionOptions,
        fmt::format("Text Reader: Stream {}", name),
        nullptr,
        true,
        contents_->fileLength);

    if (opts.skipRows() > 0) {
      seekToRow(opts.skipRows());
    }
  }
}

#define TEXT_DISPATCH_FILTER(readerFunc, filterPtr)                           \
  do {                                                                        \
    const auto _fKind = (filterPtr) ? (filterPtr)->kind()                     \
                                    : velox::common::FilterKind::kAlwaysTrue; \
    switch (_fKind) {                                                         \
      case velox::common::FilterKind::kAlwaysFalse:                           \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::AlwaysFalse>;           \
        break;                                                                \
      case velox::common::FilterKind::kAlwaysTrue:                            \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::AlwaysTrue>;            \
        break;                                                                \
      case velox::common::FilterKind::kIsNull:                                \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::IsNull>;                \
        break;                                                                \
      case velox::common::FilterKind::kIsNotNull:                             \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::IsNotNull>;             \
        break;                                                                \
      case velox::common::FilterKind::kBoolValue:                             \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::BoolValue>;             \
        break;                                                                \
      case velox::common::FilterKind::kBigintRange:                           \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::BigintRange>;           \
        break;                                                                \
      case velox::common::FilterKind::kNegatedBigintRange:                    \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::NegatedBigintRange>;    \
        break;                                                                \
      case velox::common::FilterKind::kBigintValuesUsingHashTable:            \
        fileColumns_[i].reader = &TextRowReader::readerFunc<                  \
            velox::common::BigintValuesUsingHashTable>;                       \
        break;                                                                \
      case velox::common::FilterKind::kBigintValuesUsingBitmask:              \
        fileColumns_[i].reader = &TextRowReader::readerFunc<                  \
            velox::common::BigintValuesUsingBitmask>;                         \
        break;                                                                \
      case velox::common::FilterKind::kNegatedBigintValuesUsingHashTable:     \
        fileColumns_[i].reader = &TextRowReader::readerFunc<                  \
            velox::common::NegatedBigintValuesUsingHashTable>;                \
        break;                                                                \
      case velox::common::FilterKind::kNegatedBigintValuesUsingBitmask:       \
        fileColumns_[i].reader = &TextRowReader::readerFunc<                  \
            velox::common::NegatedBigintValuesUsingBitmask>;                  \
        break;                                                                \
      case velox::common::FilterKind::kBigintValuesUsingBloomFilter:          \
        fileColumns_[i].reader = &TextRowReader::readerFunc<                  \
            velox::common::BigintValuesUsingBloomFilter>;                     \
        break;                                                                \
      case velox::common::FilterKind::kDoubleRange:                           \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::DoubleRange>;           \
        break;                                                                \
      case velox::common::FilterKind::kFloatRange:                            \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::FloatRange>;            \
        break;                                                                \
      case velox::common::FilterKind::kBytesRange:                            \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::BytesRange>;            \
        break;                                                                \
      case velox::common::FilterKind::kNegatedBytesRange:                     \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::NegatedBytesRange>;     \
        break;                                                                \
      case velox::common::FilterKind::kBytesValues:                           \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::BytesValues>;           \
        break;                                                                \
      case velox::common::FilterKind::kNegatedBytesValues:                    \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::NegatedBytesValues>;    \
        break;                                                                \
      case velox::common::FilterKind::kBigintMultiRange:                      \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::BigintMultiRange>;      \
        break;                                                                \
      case velox::common::FilterKind::kMultiRange:                            \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::MultiRange>;            \
        break;                                                                \
      case velox::common::FilterKind::kHugeintRange:                          \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::HugeintRange>;          \
        break;                                                                \
      case velox::common::FilterKind::kTimestampRange:                        \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::TimestampRange>;        \
        break;                                                                \
      case velox::common::FilterKind::kHugeintValuesUsingHashTable:           \
        fileColumns_[i].reader = &TextRowReader::readerFunc<                  \
            velox::common::HugeintValuesUsingHashTable>;                      \
        break;                                                                \
      default:                                                                \
        fileColumns_[i].reader =                                              \
            &TextRowReader::readerFunc<velox::common::Filter>;                \
        break;                                                                \
    }                                                                         \
  } while (0)

void TextRowReader::initializeColumnReaders() {
  const auto& fileType = getFileType();
  const size_t fileColumnCount = fileType.size();
  fileColumns_.resize(fileColumnCount);
  const auto& scanSpecs = scanSpec_->children();
  for (const auto& scanSpec : scanSpecs) {
    if (scanSpec->channel() == ScanSpec::kNoChannel ||
        !scanSpec->projectOut()) {
      continue;
    }

    const auto& fileName = scanSpec->fieldName();
    auto fileTypeIdx = fileType.getChildIdx(fileName);
    fileColumns_[fileTypeIdx].resultVectorIdx = scanSpec->channel();
  }

  const auto& types = fileType.children();
  const auto& names = fileType.names();
  for (vector_size_t i = 0; i < fileColumnCount; ++i) {
    const velox::common::Filter* filter = nullptr;
    auto* childSpec = scanSpec_->childByName(names[i]);
    if (childSpec) {
      filter = childSpec->filter();
    }

    const auto& type = *types[i];
    auto kind = type.kind();
    switch (kind) {
      case TypeKind::INTEGER:
        if (type.isDate()) {
          TEXT_DISPATCH_FILTER(readDate, filter);
        } else {
          TEXT_DISPATCH_FILTER(readInteger, filter);
        }
        break;
      case TypeKind::BIGINT:
        if (type.isShortDecimal()) {
          TEXT_DISPATCH_FILTER(readBigIntDecimal, filter);
        } else {
          TEXT_DISPATCH_FILTER(readBigInt, filter);
        }
        break;
      case TypeKind::HUGEINT:
        if (type.isLongDecimal()) {
          TEXT_DISPATCH_FILTER(readHugeIntDecimal, filter);
        } else {
          TEXT_DISPATCH_FILTER(readHugeInt, filter);
        }
        break;
      case TypeKind::SMALLINT:
        TEXT_DISPATCH_FILTER(readSmallInt, filter);
        break;
      case TypeKind::TINYINT:
        TEXT_DISPATCH_FILTER(readTinyInt, filter);
        break;
      case TypeKind::BOOLEAN:
        TEXT_DISPATCH_FILTER(readBoolean, filter);
        break;
      case TypeKind::VARCHAR:
        TEXT_DISPATCH_FILTER(readVarChar, filter);
        break;
      case TypeKind::VARBINARY:
        TEXT_DISPATCH_FILTER(readVarBinary, filter);
        break;
      case TypeKind::REAL:
        TEXT_DISPATCH_FILTER(readReal, filter);
        break;
      case TypeKind::DOUBLE:
        TEXT_DISPATCH_FILTER(readDouble, filter);
        break;
      case TypeKind::TIMESTAMP:
        TEXT_DISPATCH_FILTER(readTimestamp, filter);
        break;
      case TypeKind::ARRAY:
        TEXT_DISPATCH_FILTER(readArray, filter);
        break;
      case TypeKind::MAP:
        TEXT_DISPATCH_FILTER(readMap, filter);
        break;
      case TypeKind::ROW:
        TEXT_DISPATCH_FILTER(readRow, filter);
        break;
      default:
        VELOX_NYI("Unsupported type in column reader (kind code {})", kind);
    }
    fileColumns_[i].filter = filter;
  }
}

namespace {
void processMutation(RowVectorPtr& rowVecPtr, const Mutation* mutation) {
  if (!mutation) {
    return;
  }
  const auto acceptedRows = rowVecPtr->size();
  std::vector<uint64_t> passed(bits::nwords(acceptedRows), -1);
  if (mutation->deletedRows) {
    bits::andWithNegatedBits(
        passed.data(), mutation->deletedRows, 0, acceptedRows);
  }
  if (mutation->randomSkip) {
    bits::forEachSetBit(passed.data(), 0, acceptedRows, [&](auto i) {
      if (!mutation->randomSkip->testOne()) {
        bits::clearBit(passed.data(), i);
      }
    });
  }
  auto numPassed = bits::countBits(passed.data(), 0, acceptedRows);
  if (numPassed == 0) {
    rowVecPtr->resize(0);
  } else if (numPassed < acceptedRows) {
    auto indices = allocateIndices(numPassed, rowVecPtr->pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    vector_size_t j = 0;
    bits::forEachSetBit(
        passed.data(), 0, acceptedRows, [&](auto i) { rawIndices[j++] = i; });
    for (auto& child : rowVecPtr->children()) {
      if (!child) {
        continue;
      }
      child->disableMemo();
      child = BaseVector::wrapInDictionary(
          nullptr, indices, numPassed, std::move(child));
    }
    rowVecPtr->resize(numPassed);
  }
}
} // namespace

uint64_t TextRowReader::next(
    uint64_t rows,
    VectorPtr& result,
    const Mutation* mutation) {
  if (atEOF_) {
    return 0;
  }

  const auto startRow = currentRow_;
  RowVectorPtr rowVecPtr = std::dynamic_pointer_cast<RowVector>(result);
  rowVecPtr->resize(static_cast<vector_size_t>(rows));
  auto& children = rowVecPtr->children();

  const auto& fileType = getFileType();
  const auto& fileTypes = fileType.children();
  const auto& fileNames = fileType.names();
  const size_t fileColumnCount = fileType.size();

  vector_size_t acceptedRows = 0;
  const auto initialPos = pos_;
  while (!atEOF_ && acceptedRows < rows) {
    resetLine();
    rowHasError_ = false;
    bool skipRows = false;
    for (size_t i = 0; i < fileColumnCount; i++) {
      const auto& col = fileColumns_[i];
      DelimType delim = DelimTypeNone;
      if (skipRows) {
        bool isNull = false;
        getString(*this, isNull, delim);
        continue;
      }

      BaseVector* childVector;
      if (col.resultVectorIdx == kNotProjected) {
        if (!col.filter) {
          bool isNull = false;
          getString(*this, isNull, delim);
          continue;
        }
        // else ->
        // column not projected but has a filter we just
        // parse it and test the filter.
        childVector = nullptr;
      } else {
        VELOX_DCHECK_LT(col.resultVectorIdx, children.size());
        childVector = children[col.resultVectorIdx].get();
      }

      const auto& type = *fileTypes[i];
      // columnReader returns true -> filterOk, else filterFailed
      skipRows = !(this->*col.reader)(
          type, childVector, acceptedRows, delim, col.filter);
      if (rowHasError_ && contents_->onRowReject) {
        RejectedRow err{currentRow_, fileNames[i], type, errorValue_};
        contents_->onRowReject(err);
        skipRows = true;
      }
      ownedString_.clear();
    }

    if (atEOF_ && getLength() == std::numeric_limits<uint64_t>::max()) {
      // if it's a streaming this is a redundant row which means EOF
      break;
    }

    (void)skipLine();
    ++currentRow_;

    if (skipRows) {
      // we reject that error so we don't increment the size
      // (incrementing size means that we append null on error)
    } else {
      ++acceptedRows;
    }

    bool eof = false;
    if (contents_->compression == CompressionKind::CompressionKind_NONE) {
      eof = pos_ >= getLength();
    } else if (atPhysicalEOF_) {
      eof = pos_ >= contents_->decompressedInputStream->ByteCount();
    }

    if (eof) {
      setEOF();
    }

    // handle empty file
    if (initialPos == pos_ && atEOF_) {
      currentRow_ = startRow;
      acceptedRows = 0;
    }
  }

  // Resize the row vector to the actual number of rows read.
  // Handled here for both cases: pos_ > fileLength_ and pos_ > limit_
  rowVecPtr->resize(acceptedRows);
  processMutation(rowVecPtr, mutation);
  result = std::move(rowVecPtr);
  VELOX_DCHECK_GE(currentRow_, startRow);
  return currentRow_ - startRow;
}

uint64_t TextRowReader::seekToRow(uint64_t rowNumber) {
  VELOX_CHECK_GT(
      rowNumber, currentRow_, "Text file cannot seek to earlier row");

  while (currentRow_ < rowNumber && !skipLine()) {
    currentRow_++;
    resetLine();
  }

  return currentRow_;
}

uint64_t TextRowReader::getLength() {
  if (fileLength_ == std::numeric_limits<uint64_t>::max()) {
    fileLength_ = getStreamLength();
  }
  return fileLength_;
}

uint64_t TextRowReader::getStreamLength() const {
  return contents_->input->getInputStream()->getLength();
}

void TextRowReader::setEOF() {
  atEOF_ = true;
  atEOL_ = true;
}

/// TODO: Update maximum depth after fixing issue with deeply nested complex
/// types
void TextRowReader::incrementDepth() {
  if (depth_ > 4) {
    dwio::common::parse_error("Schema nesting too deep");
  }
  depth_++;
}

void TextRowReader::decrementDepth(DelimType& delim) {
  if (depth_ == 0) {
    dwio::common::logic_error("Attempt to decrement nesting depth of 0");
  }
  depth_--;
  auto d = depth_ + DelimTypeEOR;
  if (delim > d) {
    setNone(delim);
  }
}

void TextRowReader::setEOE(DelimType& delim) {
  // Set delim if it is currently None or a more deeply
  // delimiter, to simply the code where aggregates
  // parse nested aggregates.
  auto d = depth_ + DelimTypeEOE;
  if (isNone(delim) || d < delim) {
    delim = d;
  }
}

void TextRowReader::resetEOE(DelimType& delim) {
  // Reset delim it is EOE or above.
  auto d = depth_ + DelimTypeEOE;
  if (delim >= d) {
    setNone(delim);
  }
}

void TextRowReader::setEOR(DelimType& delim) {
  // Set delim if it is currently None or a more
  // deeply nested delimiter.
  auto d = depth_ + DelimTypeEOR;
  if (isNone(delim) || delim > d) {
    delim = d;
  }
}

bool TextRowReader::isEOR(DelimType delim) {
  // Return true if delim is the EOR for the current depth
  // or a less deeply nested depth.
  return (delim != DelimTypeNone && delim <= (depth_ + DelimTypeEOR));
}

bool TextRowReader::isOuterEOR(DelimType delim) {
  // Return true if delim is the EOR for the enclosing object.
  // For example, when parsing ARRAY elements, which leave delim
  // set to the EOR for their depth on return, isOuterEOR will
  // return true if we have reached the ARRAY EOR delimiter at
  // the end of the latest element.
  return (delim != DelimTypeNone && delim < (depth_ + DelimTypeEOR));
}

bool TextRowReader::isEOEorEOR(DelimType delim) {
  return (!isNone(delim) && delim <= (depth_ + DelimTypeEOE));
}

void TextRowReader::setNone(DelimType& delim) {
  delim = DelimTypeNone;
}

bool TextRowReader::isNone(DelimType delim) {
  return (delim == DelimTypeNone);
}

std::string_view
TextRowReader::getString(TextRowReader& th, bool& isNull, DelimType& delim) {
  if (th.atEOL_) {
    delim = DelimTypeEOR; // top-level EOR
  }

  if (th.isEOEorEOR(delim)) {
    isNull = true;
    return {};
  }

  bool wasEscaped = false;
  th.ownedString_.clear();

  // Processing has to be done character by characater instad of chunk by chunk.
  // This is to avoid edge case handling if escape character(s) are cut off at
  // the end of the chunk.
  while (true) {
    auto v = th.getByteOptimized(delim);
    if (!th.isNone(delim)) {
      break;
    }

    if (th.contents_->serDeOptions.isEscaped &&
        v == th.contents_->serDeOptions.escapeChar) {
      wasEscaped = true;
      th.ownedString_.push_back(static_cast<char>(v));
      v = th.getByteUncheckedOptimized(delim);
      if (!th.isNone(delim)) {
        break;
      }
    }
    th.ownedString_.push_back(static_cast<char>(v));
  }

  if (th.ownedStringView() == th.contents_->serDeOptions.nullString) {
    isNull = true;
    return {};
  }

  if (wasEscaped) {
    // We need to copy the data byte by byte only if there is at least one
    // escaped byte.
    uint64_t j = 0;
    for (uint64_t i = 0; i < th.ownedString_.size(); i++) {
      if (th.ownedString_[i] == th.contents_->serDeOptions.escapeChar &&
          i < th.ownedString_.size() - 1) {
        // Check if it's '\r' or '\n'.
        i++;
        if (th.ownedString_[i] == 'r') {
          th.ownedString_[j++] = '\r';
        } else if (th.ownedString_[i] == 'n') {
          th.ownedString_[j++] = '\n';
        } else {
          // Keep the next byte.
          th.ownedString_[j++] = th.ownedString_[i];
        }
      } else {
        th.ownedString_[j++] = th.ownedString_[i];
      }
    }
    th.ownedString_.resize(j);
  }

  return th.ownedStringView();
}

template <typename T, typename Filter>
bool TextRowReader::setValueFromString(
    std::string_view str,
    BaseVector* data,
    vector_size_t insertionRow,
    std::function<std::optional<T>(std::string_view)> convert,
    const velox::common::Filter* filter) {
  if (atEOF_ && atSOL_) {
    return true;
  }

  auto result = str.empty() ? std::nullopt : convert(str);

  if (data == nullptr) {
    // No output vector — still evaluate filter for non-projected columns.
    if (result) {
      return testFilter<Filter>(filter, *result);
    }
    return testFilterNull<Filter>(filter);
  }

  auto flatVector = data->asUnchecked<FlatVector<T>>();
  if (result) {
    if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
      if (!testFilter<Filter>(filter, *result)) {
        return false;
      }
    }
    flatVector->set(insertionRow, *result);
  } else {
    if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
      if (!filter->testNull()) {
        return false;
      }
    }
    flatVector->setNull(insertionRow, true);
  }
  return true;
}

uint8_t TextRowReader::getByteOptimized(DelimType& delim) {
  setNone(delim);
  auto v = getByteUncheckedOptimized(delim);
  if (isNone(delim)) {
    if (v == '\r') {
      v = getByteUncheckedOptimized<true>(
          delim); // always returns '\n' in this case
    }
    delim = getDelimType(v);
  }
  return v;
}

DelimType TextRowReader::getDelimType(uint8_t v) {
  DelimType delim = DelimTypeNone;

  if (v == '\n') {
    atEOL_ = true;
    delim = DelimTypeEOR; // top level EOR

    /// TODO: Logically should be >=, kept as it is to align with presto reader.
    if (pos_ > limit_) {
      atEOF_ = true;
      delim = DelimTypeEOR;
    }
  } else if (v == contents_->serDeOptions.separators.at(depth_)) {
    setEOE(delim);
  } else {
    setNone(delim);
    uint64_t i = depth_;
    while (i > 0) {
      i--;
      if (v == contents_->serDeOptions.separators.at(i)) {
        delim = i + DelimTypeEOR; // level-based EOR
        break;
      }
    }
  }
  return delim;
}

template <bool skipLF>
char TextRowReader::getByteUncheckedOptimized(DelimType& delim) {
  if (atEOL_) {
    if (!skipLF) {
      delim = DelimTypeEOR; // top level EOR
    }
    return '\n';
  }

  try {
    char v;
    if (contents_->compression != CompressionKind::CompressionKind_NONE &&
        preLoadedUnreadData_.empty()) {
      int length = 0;
      const void* buffer = nullptr;
      atPhysicalEOF_ =
          !contents_->decompressedInputStream->Next(&buffer, &length);
      if (!atPhysicalEOF_) {
        preLoadedUnreadData_ =
            std::string_view(reinterpret_cast<const char*>(buffer), length);
      }
    }

    if (unreadData_.empty() || unreadIdx_ >= unreadData_.size()) {
      bool updated = false;
      if (contents_->compression != CompressionKind::CompressionKind_NONE) {
        unreadData_.assign(
            preLoadedUnreadData_.data(), preLoadedUnreadData_.size());
        preLoadedUnreadData_ = {};
        updated = !unreadData_.empty();
      } else {
        int length = 0;
        const void* buffer = nullptr;
        if (contents_->inputStream->Next(&buffer, &length) && length > 0) {
          VELOX_CHECK_NOT_NULL(buffer);
          unreadData_.assign(reinterpret_cast<const char*>(buffer), length);
          updated = true;
        }
      }

      if (!updated) {
        setEOF();
        delim = DelimTypeEOR;
        return '\0';
      }
      unreadIdx_ = 0;
    }

    v = unreadData_[unreadIdx_++];
    pos_++;

    // only when previous char == '\r'
    if (skipLF) {
      if (v != '\n') {
        pos_--;
        return '\n';
      }
    } else {
      atSOL_ = false;
    }
    return v;
  } catch (EOFError&) {
  } catch (std::runtime_error& e) {
    if (std::string(e.what()).find("Short read of") != 0 && !skipLF) {
      throw;
    }
  }
  if (!skipLF) {
    setEOF();
    delim = DelimTypeEOR;
  }
  return '\n';
}

bool TextRowReader::getEOR(DelimType& delim, bool& isNull) {
  if (isEOR(delim)) {
    isNull = true;
    return true;
  }
  if (atEOL_) {
    delim = DelimTypeEOR; // top-level EOR
    isNull = true;
    return true;
  }
  bool wasAtSOL = atSOL_;
  setNone(delim);
  ownedString_.clear();
  const auto& ns = contents_->serDeOptions.nullString;
  uint8_t v = 0;
  while (true) {
    v = getByteUncheckedOptimized(delim);
    if (isNone(delim)) {
      if (v == '\r') {
        // always returns '\n' in this case
        v = getByteUncheckedOptimized<true>(delim);
      }
      delim = getDelimType(v);
    }

    if (isEOR(delim) || atEOL_) {
      if (ownedStringView() == ns) {
        isNull = true;
      } else if (!ownedString_.empty()) {
        break;
      }
      setEOR(delim);
      return true;
    }
    if (ownedString_.size() >= ns.size() ||
        static_cast<char>(v) != ns[ownedString_.size()]) {
      break;
    }
    ownedString_.push_back(static_cast<char>(v));
  }

  unreadData_.insert(0, 1, static_cast<char>(v));
  pos_--;
  if (!ownedString_.empty()) {
    unreadData_.insert(0, ownedStringView());
    pos_ -= ownedString_.size();
  }
  atEOL_ = false;
  atSOL_ = wasAtSOL;
  setNone(delim);
  return false;
}

bool TextRowReader::skipLine() {
  DelimType delim = DelimTypeNone;
  while (!atEOL_) {
    (void)getByteOptimized(delim);
  }
  /// TODO: Logically should be >=, kept as it is to align with presto reader
  if (pos_ > limit_) {
    setEOF();
    delim = DelimTypeEOR;
  }
  return atEOF_;
}

void TextRowReader::resetLine() {
  if (!atEOF_) {
    atEOL_ = false;
    VELOX_CHECK_EQ(depth_, 0);
  }
  atSOL_ = true;
}

template <typename T>
T TextRowReader::getNumeric(TextRowReader& th, bool& isNull, DelimType& delim) {
  const auto str = getString(th, isNull, delim);

  if (str.empty()) {
    isNull = true;
  }
  if (isNull) {
    return 0;
  }

  const char* ptr = str.data();
  const char* end = str.data() + str.size();

  T v = 0;
  fast_float::parse_options options{
      fast_float::chars_format::general |
      fast_float::chars_format::skip_white_space};
  auto [parseEnd, ec] = fast_float::from_chars_advanced(ptr, end, v, options);
  if (ec != std::errc{} || parseEnd != end) {
    isNull = true;
    th.rowHasError_ = true;
    th.errorValue_ = str;
    return {};
  }

  return v;
}

bool TextRowReader::getBoolean(
    TextRowReader& th,
    bool& isNull,
    DelimType& delim) {
  const auto str = getString(th, isNull, delim);
  if (str.empty()) {
    isNull = true;
  }
  if (isNull) {
    return false;
  }

  switch (str.size()) {
    case 1:
      if (str[0] == '1' || str[0] == 't' || str[0] == 'T') {
        return true;
      }
      if (str[0] == '0' || str[0] == 'f' || str[0] == 'F') {
        return false;
      }
      break;
    case 2:
      if ((static_cast<unsigned char>(str[0]) | 0x20U) == 'o' &&
          (static_cast<unsigned char>(str[1]) | 0x20U) == 'n') {
        return true;
      }
      if ((static_cast<unsigned char>(str[0]) | 0x20U) == 'n' &&
          (static_cast<unsigned char>(str[1]) | 0x20U) == 'o') {
        return false;
      }
      break;
    case 3:
      if ((static_cast<unsigned char>(str[0]) | 0x20U) == 'y' &&
          (static_cast<unsigned char>(str[1]) | 0x20U) == 'e' &&
          (static_cast<unsigned char>(str[2]) | 0x20U) == 's') {
        return true;
      }
      if ((static_cast<unsigned char>(str[0]) | 0x20U) == 'o' &&
          (static_cast<unsigned char>(str[1]) | 0x20U) == 'f' &&
          (static_cast<unsigned char>(str[2]) | 0x20U) == 'f') {
        return false;
      }
      break;
    case 4:
      if ((static_cast<unsigned char>(str[0]) | 0x20U) == 't' &&
          (static_cast<unsigned char>(str[1]) | 0x20U) == 'r' &&
          (static_cast<unsigned char>(str[2]) | 0x20U) == 'u' &&
          (static_cast<unsigned char>(str[3]) | 0x20U) == 'e') {
        return true;
      }
      break;
    case 5:
      if ((static_cast<unsigned char>(str[0]) | 0x20U) == 'f' &&
          (static_cast<unsigned char>(str[1]) | 0x20U) == 'a' &&
          (static_cast<unsigned char>(str[2]) | 0x20U) == 'l' &&
          (static_cast<unsigned char>(str[3]) | 0x20U) == 's' &&
          (static_cast<unsigned char>(str[4]) | 0x20U) == 'e') {
        return false;
      }
      break;
    default:
      break;
  }

  isNull = true;
  th.rowHasError_ = true;
  th.errorValue_ = str;
  return false;
}

void TextRowReader::readElement(
    const std::shared_ptr<const Type>& t,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim) {
  // readElement is used for nested type elements (arrays, maps, rows)
  // where no filter applies, so we use AlwaysTrue with nullptr filter.
  using NoFilter = velox::common::AlwaysTrue;
  switch (t->kind()) {
    case TypeKind::INTEGER:
      if (t->isDate()) {
        readDate<NoFilter>(*t, data, insertionRow, delim, nullptr);
      } else {
        readInteger<NoFilter>(*t, data, insertionRow, delim, nullptr);
      }
      break;

    case TypeKind::BIGINT:
      if (t->isShortDecimal()) {
        readBigIntDecimal<NoFilter>(*t, data, insertionRow, delim, nullptr);
      } else {
        readBigInt<NoFilter>(*t, data, insertionRow, delim, nullptr);
      }
      break;

    case TypeKind::HUGEINT:
      if (t->isLongDecimal()) {
        readHugeIntDecimal<NoFilter>(*t, data, insertionRow, delim, nullptr);
      } else {
        readHugeInt<NoFilter>(*t, data, insertionRow, delim, nullptr);
      }
      break;

    case TypeKind::SMALLINT:
      readSmallInt<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::VARBINARY:
      readVarBinary<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::VARCHAR:
      readVarChar<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::BOOLEAN:
      readBoolean<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::TINYINT:
      readTinyInt<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::ARRAY:
      readArray<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::ROW:
      readRow<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::MAP:
      readMap<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::REAL:
      readReal<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::DOUBLE:
      readDouble<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    case TypeKind::TIMESTAMP:
      readTimestamp<NoFilter>(*t, data, insertionRow, delim, nullptr);
      break;

    default:
      VELOX_NYI("readElement unhandled type (kind code {})", t->kind());
  }
}

template <typename T, typename Filter, typename F>
bool TextRowReader::putValue(
    const F& f,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  T v;
  if (isEOR(delim)) {
    isNull = true;
    v = 0;
  } else {
    v = f(*this, isNull, delim);
  }

  if (atEOF_ && atSOL_) {
    return true;
  }

  if (data == nullptr) {
    // No output vector — still evaluate filter for non-projected columns.
    if (isNull) {
      return testFilterNull<Filter>(filter);
    }
    return testFilter<Filter>(filter, v);
  }

  auto flatVector = data->asUnchecked<FlatVector<T>>();
  if (isNull) {
    if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
      if (!filter->testNull()) {
        return false;
      }
    }
    flatVector->setNull(insertionRow, true);
    return true;
  }

  if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
    if (!testFilter<Filter>(filter, v)) {
      return false;
    }
  }

  flatVector->set(insertionRow, v);
  return true;
}

const RowType& TextRowReader::getFileType() const {
  return *contents_->schema;
}

// Specialized column readers implementation

template <typename Filter>
bool TextRowReader::readInteger(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  return putValue<int32_t, Filter>(
      getNumeric<int32_t>, data, insertionRow, delim, filter);
}

template <typename Filter>
bool TextRowReader::readDate(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto str = getString(*this, isNull, delim);
  return setValueFromString<int32_t, Filter>(
      str,
      data,
      insertionRow,
      [](std::string_view s) -> std::optional<int32_t> {
        return DATE()->toDays(s);
      },
      filter);
}

template <typename Filter>
bool TextRowReader::readBigInt(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  return putValue<int64_t, Filter>(
      getNumeric<int64_t>, data, insertionRow, delim, filter);
}

template <typename Filter>
bool TextRowReader::readBigIntDecimal(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto str = getString(*this, isNull, delim);
  auto decimalParams = getDecimalPrecisionScale(type);
  const auto precision = decimalParams.first;
  const auto scale = decimalParams.second;
  return setValueFromString<int64_t, Filter>(
      str,
      data,
      insertionRow,
      [precision, scale](std::string_view s) -> std::optional<int64_t> {
        int64_t v = 0;
        const auto status = DecimalUtil::castFromString(
            StringView(s.data(), static_cast<int32_t>(s.size())),
            precision,
            scale,
            v);
        return status.ok() ? std::optional<int64_t>(v) : std::nullopt;
      },
      filter);
}

template <typename Filter>
bool TextRowReader::readSmallInt(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  return putValue<int16_t, Filter>(
      getNumeric<int16_t>, data, insertionRow, delim, filter);
}

template <typename Filter>
bool TextRowReader::readTinyInt(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  return putValue<int8_t, Filter>(
      getNumeric<int8_t>, data, insertionRow, delim, filter);
}

template <typename Filter>
bool TextRowReader::readBoolean(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  return putValue<bool, Filter>(getBoolean, data, insertionRow, delim, filter);
}

template <typename Filter>
bool TextRowReader::readVarChar(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto str = getString(*this, isNull, delim);

  if (atEOF_ && atSOL_) {
    return true;
  }

  if (data == nullptr) {
    // No output vector — still evaluate filter for non-projected columns.
    if (isNull) {
      return testFilterNull<Filter>(filter);
    }
    return testFilter<Filter>(filter, str);
  }

  if (isNull) {
    if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
      if (!filter->testNull()) {
        return false;
      }
    }
    const auto& flatVector = data->asUnchecked<FlatVector<StringView>>();
    flatVector->setNull(insertionRow, true);
    return true;
  }

  if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
    if (!testFilter<Filter>(filter, str)) {
      return false;
    }
  }

  const auto& flatVector = data->asUnchecked<FlatVector<StringView>>();
  flatVector->set(
      insertionRow, StringView(str.data(), static_cast<int32_t>(str.size())));

  return true;
}

template <typename Filter>
bool TextRowReader::readVarBinary(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto str = getString(*this, isNull, delim);

  if (atEOF_ && atSOL_) {
    return true;
  }

  if (data == nullptr) {
    // No output vector — still evaluate filter for non-projected columns.
    if (isNull) {
      return testFilterNull<Filter>(filter);
    }
    return testFilter<Filter>(filter, str);
  }

  if (isNull) {
    if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
      if (!filter->testNull()) {
        return false;
      }
    }
    const auto& flatVector = data->asUnchecked<FlatVector<StringView>>();
    flatVector->setNull(insertionRow, true);
    return true;
  }

  if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
    if (!testFilter<Filter>(filter, str)) {
      return false;
    }
  }

  const auto& flatVector = data->asUnchecked<FlatVector<StringView>>();

  size_t len = str.size();
  const auto blen = encoding::Base64::calculateDecodedSize(str.data(), len);
  varBinBuf_->resize(blen.value_or(0));

  Status status = encoding::Base64::decode(
      str.data(), str.size(), varBinBuf_->data(), blen.value_or(0));

  if (status.code() == StatusCode::kOK) {
    flatVector->set(
        insertionRow,
        StringView(varBinBuf_->data(), static_cast<int32_t>(blen.value())));
  } else {
    varBinBuf_->resize(str.size());
    VELOX_CHECK_NOT_NULL(str.data());
    memcpy(varBinBuf_->data(), str.data(), str.size());
    flatVector->set(
        insertionRow,
        StringView(varBinBuf_->data(), static_cast<int32_t>(str.size())));
  }

  return true;
}

template <typename Filter>
bool TextRowReader::readReal(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  return putValue<float, Filter>(
      getNumeric<float>, data, insertionRow, delim, filter);
}

template <typename Filter>
bool TextRowReader::readDouble(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  return putValue<double, Filter>(
      getNumeric<double>, data, insertionRow, delim, filter);
}

template <typename Filter>
bool TextRowReader::readTimestamp(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto str = getString(*this, isNull, delim);

  if (atEOF_ && atSOL_) {
    return true;
  }

  if (data == nullptr) {
    // No output vector — still evaluate filter for non-projected columns.
    if (str.empty()) {
      return testFilterNull<Filter>(filter);
    }
    auto ts = util::Converter<TypeKind::TIMESTAMP>::tryCast(str).thenOrThrow(
        folly::identity,
        [&](const Status& status) { VELOX_USER_FAIL(status.message()); });
    auto value = Timestamp{ts.getSeconds(), ts.getNanos()};
    return testFilter<Filter>(filter, value);
  }

  auto flatVector = data->asUnchecked<FlatVector<Timestamp>>();

  if (str.empty()) {
    if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
      if (!filter->testNull()) {
        return false;
      }
    }
    flatVector->setNull(insertionRow, true);
  } else {
    auto ts = util::Converter<TypeKind::TIMESTAMP>::tryCast(str).thenOrThrow(
        folly::identity,
        [&](const Status& status) { VELOX_USER_FAIL(status.message()); });
    auto value = Timestamp{ts.getSeconds(), ts.getNanos()};
    if constexpr (!std::is_same_v<Filter, velox::common::AlwaysTrue>) {
      if (!testFilter<Filter>(filter, value)) {
        return false;
      }
    }
    flatVector->set(insertionRow, value);
  }

  return true;
}

template <typename Filter>
bool TextRowReader::readHugeInt(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto str = getString(*this, isNull, delim);
  return setValueFromString<int128_t, Filter>(
      str,
      data,
      insertionRow,
      [](std::string_view s) -> std::optional<int128_t> {
        return HugeInt::parse(std::string{s});
      },
      filter);
}

template <typename Filter>
bool TextRowReader::readHugeIntDecimal(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto str = getString(*this, isNull, delim);
  auto decimalParams = getDecimalPrecisionScale(type);
  const auto precision = decimalParams.first;
  const auto scale = decimalParams.second;
  return setValueFromString<int128_t, Filter>(
      str,
      data,
      insertionRow,
      [precision, scale](std::string_view s) -> std::optional<int128_t> {
        int128_t v = 0;
        const auto status = DecimalUtil::castFromString(
            StringView(s.data(), static_cast<int32_t>(s.size())),
            precision,
            scale,
            v);
        return status.ok() ? std::optional<int128_t>(v) : std::nullopt;
      },
      filter);
}

template <typename Filter>
bool TextRowReader::readArray(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto& ct = type.childAt(0);
  const auto& arrayVector = data ? data->asUnchecked<ArrayVector>() : nullptr;

  incrementDepth();
  (void)getEOR(delim, isNull);

  if (arrayVector != nullptr) {
    auto rawSizes = arrayVector->sizes()->asMutable<vector_size_t>();
    auto rawOffsets = arrayVector->offsets()->asMutable<vector_size_t>();

    rawOffsets[insertionRow] = insertionRow > 0
        ? rawOffsets[insertionRow - 1] + rawSizes[insertionRow - 1]
        : 0;
    const int startElementIdx = rawOffsets[insertionRow];

    vector_size_t elementCount = 0;
    if (isNull) {
      arrayVector->setNull(insertionRow, isNull);
      rawSizes[insertionRow] = 0;
    } else {
      while (!isOuterEOR(delim)) {
        setNone(delim);
        auto elementsVector = arrayVector->elements().get();
        resizeVector(elementsVector, startElementIdx + elementCount);

        readElement(ct, elementsVector, startElementIdx + elementCount, delim);

        rawSizes[insertionRow] = ++elementCount;

        if (atEOF_ && atSOL_) {
          decrementDepth(delim);
          return true;
        }
      }
    }

  } else {
    while (!isOuterEOR(delim)) {
      setNone(delim);
      readElement(ct, nullptr, 0, delim);
    }
  }
  decrementDepth(delim);
  return true;
}

template <typename Filter>
bool TextRowReader::readMap(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto& mapt = type.asMap();
  const auto& key = mapt.keyType();
  const auto& value = mapt.valueType();
  const auto& mapVector = data ? data->asUnchecked<MapVector>() : nullptr;
  incrementDepth();
  (void)getEOR(delim, isNull);

  if (mapVector != nullptr) {
    auto rawOffsets = mapVector->offsets()->asMutable<vector_size_t>();
    auto rawSizes = mapVector->sizes()->asMutable<vector_size_t>();

    rawOffsets[insertionRow] = insertionRow > 0
        ? rawOffsets[insertionRow - 1] + rawSizes[insertionRow - 1]
        : 0;
    const int startElementIdx = rawOffsets[insertionRow];

    vector_size_t elementCount = 0;
    if (isNull) {
      mapVector->setNull(insertionRow, isNull);
      rawSizes[insertionRow] = 0;
    } else {
      while (!isOuterEOR(delim)) {
        setNone(delim);
        incrementDepth();

        auto keysVector = mapVector->mapKeys().get();
        resizeVector(keysVector, startElementIdx + elementCount);

        readElement(key, keysVector, startElementIdx + elementCount, delim);

        if (atEOF_ && atSOL_) {
          rawSizes[insertionRow] = elementCount;
          rawOffsets[insertionRow + 1] = startElementIdx + elementCount;
          decrementDepth(delim);
          decrementDepth(delim);
          return true;
        }
        resetEOE(delim);

        auto valsVector = mapVector->mapValues().get();
        resizeVector(valsVector, startElementIdx + elementCount);

        readElement(value, valsVector, startElementIdx + elementCount, delim);

        rawSizes[insertionRow] = ++elementCount;

        decrementDepth(delim);
      }
    }

  } else {
    while (!isOuterEOR(delim)) {
      setNone(delim);
      incrementDepth();
      readElement(key, nullptr, 0, delim);
      resetEOE(delim);
      readElement(value, nullptr, 0, delim);
      decrementDepth(delim);
    }
  }
  decrementDepth(delim);
  return true;
}

template <typename Filter>
bool TextRowReader::readRow(
    const Type& type,
    BaseVector* FOLLY_NULLABLE data,
    vector_size_t insertionRow,
    DelimType& delim,
    const velox::common::Filter* filter) {
  bool isNull = false;
  const auto& childCount = type.size();
  const auto& rowVector = data ? data->asUnchecked<RowVector>() : nullptr;
  incrementDepth();

  if (rowVector != nullptr) {
    if (isNull) {
      rowVector->setNull(insertionRow, isNull);
    } else {
      for (uint64_t j = 0; j < childCount; j++) {
        if (!isOuterEOR(delim)) {
          setNone(delim);
        }

        BaseVector* childVector = nullptr;
        if (j < type.size()) {
          childVector = rowVector->childAt(j).get();
        }
        resizeVector(childVector, insertionRow);
        const auto& childType = type.childAt(j);
        readElement(childType, childVector, insertionRow, delim);

        if (atEOF_ && atSOL_) {
          decrementDepth(delim);
          return true;
        }
      }
    }
  } else {
    for (uint64_t j = 0; j < childCount; j++) {
      if (!isOuterEOR(delim)) {
        setNone(delim);
      }
      const auto& childType = type.childAt(j);
      readElement(childType, nullptr, 0, delim);
    }
  }

  decrementDepth(delim);
  setEOE(delim);
  return true;
}

TextReader::TextReader(
    ReaderOptions options,
    std::unique_ptr<BufferedInput> input)
    : options_{std::move(options)} {
  auto schema = options_.fileSchema();
  VELOX_USER_CHECK_NOT_NULL(schema, "File schema for TEXT must be set.");

  contents_ = std::make_shared<FileContents>(options_.memoryPool(), schema);

  VELOX_USER_CHECK(
      contents_->schema->isRow(), "File schema must be a ROW type");

  contents_->input = std::move(input);

  // Find the size of the file using the option or filesystem.
  contents_->fileLength = std::min(
      options_.tailLocation(),
      static_cast<uint64_t>(contents_->input->getInputStream()->getLength()));

  /**
   * We are now allowing delimiters/separators and escape characters to be the
   * same. This could be error prone because we are checking for delimiters
   * before escape characters.
   *
   * Example:
   * delim = ','; escapeChar = ','
   * dataToParse = "1,,2"
   * Schema = ROW(ARRAY(VARCHAR()))
   *
   * Scenario 1: Check delimiter before escape (current implementation)
   * Output: ["1", NULL, "2"]
   *
   * Scenario 2: Check escape before delim
   * Output: ["1,2"]
   *
   * TODO: This is not a bug but would be good to be able to handle this
   * ambiguity
   */

  contents_->serDeOptions = options_.serDeOptions();
  contents_->onRowReject = options_.onRowReject();
  VELOX_CHECK(
      contents_->serDeOptions.nullString != "\r",
      "\'\\r\' is not allowed to be nullString");
  VELOX_CHECK(
      contents_->serDeOptions.nullString != "\n",
      "\'\\n\' is not allowed to be nullString");

  setCompressionSettings(
      contents_->input->getName(),
      contents_->compression,
      contents_->compressionOptions);
}

std::optional<uint64_t> TextReader::numberOfRows() const {
  return std::nullopt;
}

std::unique_ptr<ColumnStatistics> TextReader::columnStatistics(
    uint32_t /*index*/) const {
  return nullptr;
}

const std::shared_ptr<const RowType>& TextReader::rowType() const {
  return contents_->schema;
}

const std::shared_ptr<const TypeWithId>& TextReader::typeWithId() const {
  if (!typeWithId_) {
    typeWithId_ = TypeWithId::create(rowType());
  }
  return typeWithId_;
}

std::unique_ptr<RowReader> TextReader::createRowReader(
    const RowReaderOptions& opts) const {
  return std::make_unique<TextRowReader>(contents_, opts);
}

} // namespace facebook::velox::text
