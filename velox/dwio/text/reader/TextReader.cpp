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
#include <cctype>
#include <fast_float/fast_float.h>
#include <glog/logging.h>
#include "velox/common/encode/Base64.h"
#include "velox/dwio/common/exception/Exceptions.h"
#include "velox/type/Conversions.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/fbhive/HiveTypeParser.h"

namespace facebook::velox::text {

using common::CompressionKind;
using dwio::common::RowReader;
using dwio::common::verify;
using folly::AsciiCaseInsensitive;
using folly::StringPiece;

constexpr const char* kTextfileCompressionExtensionGzip = ".gz";
constexpr const char* kTextfileCompressionExtensionDeflate = ".deflate";
constexpr const char* kTextfileCompressionExtensionZst = ".zst";
constexpr const char* kTextfileCompressionExtensionLz4 = ".lz4";
constexpr const char* kTextfileCompressionExtensionLzo = ".lzo";
constexpr const char* kTextfileCompressionExtensionSnappy = ".snappy";

namespace {
constexpr const int32_t kDecompressionBufferFactor = 3;

bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
      str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void setCompressionSettings(
    const std::string& filename,
    CompressionKind& kind,
    dwio::common::compression::CompressionOptions& compressionOptions) {
  if (endsWith(filename, kTextfileCompressionExtensionLz4) ||
      endsWith(filename, kTextfileCompressionExtensionLzo) ||
      endsWith(filename, kTextfileCompressionExtensionSnappy)) {
    VELOX_FAIL("Unsupported compression extension for file: {}", filename);
  }
  if (endsWith(filename, kTextfileCompressionExtensionGzip)) {
    kind = CompressionKind::CompressionKind_GZIP;
    compressionOptions.format.zlib.windowBits = 15;
  } else if (endsWith(filename, kTextfileCompressionExtensionDeflate)) {
    kind = CompressionKind::CompressionKind_ZLIB;
    compressionOptions.format.zlib.windowBits = -15;
  } else if (endsWith(filename, kTextfileCompressionExtensionZst)) {
    kind = CompressionKind::CompressionKind_ZSTD;
  } else {
    kind = CompressionKind::CompressionKind_NONE;
  }
}

// Parse boolean using folly's case-insensitive comparison
std::optional<bool> parseBoolean(std::string_view value) {
  StringPiece sp(value.data(), value.size());
  if (sp.equals("TRUE", AsciiCaseInsensitive()) ||
      sp.equals("true", AsciiCaseInsensitive())) {
    return true;
  }
  if (sp.equals("FALSE", AsciiCaseInsensitive()) ||
      sp.equals("false", AsciiCaseInsensitive())) {
    return false;
  }
  return std::nullopt;
}

template <typename T>
std::optional<T> parseFloating(std::string_view value) {
  if (value.empty()) {
    return std::nullopt;
  }

  T parsed{};
  auto* begin = value.data();
  auto* end = value.data() + value.size();
  fast_float::parse_options options{
      fast_float::chars_format::general |
      fast_float::chars_format::skip_white_space};
  auto [parseEnd, error] =
      fast_float::from_chars_advanced(begin, end, parsed, options);
  if (error != std::errc{}) {
    return std::nullopt;
  }
  while (parseEnd != end &&
         std::isspace(static_cast<unsigned char>(*parseEnd))) {
    ++parseEnd;
  }
  if (parseEnd != end) {
    return std::nullopt;
  }
  return parsed;
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
      compressionOptions{},
      needsEscape{} {
  needsEscape.fill(false);
  needsEscape.at(0) = true;
}

TextRowReader::TextRowReader(
    std::shared_ptr<FileContents> fileContents,
    const RowReaderOptions& opts)
    : RowReader(),
      contents_{fileContents},
      schemaWithId_{TypeWithId::create(fileContents->schema)},
      scanSpec_{opts.scanSpec()},
      selectedSchema_{nullptr},
      options_{opts},
      columnSelector_{
          ColumnSelector::apply(opts.selector(), contents_->schema)},
      pos_{opts.offset()},
      limit_{opts.limit()},
      varBinBuf_{
          std::make_shared<dwio::common::DataBuffer<char>>(contents_->pool)} {
  if (contents_->compression == CompressionKind::CompressionKind_NONE) {
    // Create stream with 1MB block size for better I/O efficiency
    contents_->inputStream =
        std::make_unique<dwio::common::SeekableFileInputStream>(
            contents_->input->getInputStream(),
            pos_,
            contents_->fileLength - pos_,
            contents_->pool,
            dwio::common::LogType::STREAM,
            kTextBlockSize);

    if (pos_ != 0) {
      skipPartialLine_ = true;
    }
    rowsToSkip_ = opts.skipRows();
  } else {
    if (pos_ != 0) {
      atEOF_ = true;
    }
    limit_ = std::numeric_limits<uint64_t>::max();

    contents_->inputStream = contents_->input->loadCompleteFile();
    auto name = contents_->inputStream->getName();
    contents_->decompressedInputStream = createDecompressor(
        contents_->compression,
        std::move(contents_->inputStream),
        kDecompressionBufferFactor * contents_->fileLength,
        contents_->pool,
        contents_->compressionOptions,
        fmt::format("Text Reader: Stream {}", name),
        nullptr,
        true,
        contents_->fileLength);

    rowsToSkip_ = opts.skipRows();
  }

  // Cache frequently-accessed values for hot path
  fieldDelim_ = contents_->serDeOptions.separators.at(0);
  nullString_ = contents_->serDeOptions.nullString;
  isEscaped_ = contents_->serDeOptions.isEscaped;

  // Initialize precompiled setters for all selected columns
  initializeColumnSetters();
}

template <typename T>
static std::optional<T> parseIntegerWithDecimal(std::string_view str) {
  if (str.empty()) {
    return std::nullopt;
  }

  // Test if first char is acceptable for integer format
  char c = str[0];
  if (c != '-' && !std::isdigit(static_cast<unsigned char>(c))) {
    return std::nullopt;
  }

  int64_t v = 0;
  unsigned long long scanPos = 0;
  errno = 0;

  // Need null-terminated string for sscanf
  std::string tmp(str);
  auto scanCount = sscanf(tmp.c_str(), "%" SCNd64 "%lln", &v, &scanPos);
  if (scanCount != 1 || errno == ERANGE) {
    return std::nullopt;
  }

  // Check remaining chars: only allow decimal point followed by digits
  if (scanPos < str.size()) {
    for (size_t i = scanPos; i < str.size(); i++) {
      if (i == scanPos && str[i] == '.') {
        continue;
      }
      if (str[i] >= '0' && str[i] <= '9') {
        continue;
      }
      // Invalid char (includes 'e' for scientific notation)
      return std::nullopt;
    }
  }

  // Overflow check for smaller integer types
  if constexpr (!std::is_same_v<T, int64_t>) {
    if (static_cast<int64_t>(static_cast<T>(v)) != v) {
      return std::nullopt;
    }
  }

  return static_cast<T>(v);
}

void TextRowReader::initializeColumnSetters() {
  const auto& fileSchema = contents_->schema;
  const size_t numColumns = fileSchema->size();

  // Get requested schema for type coercion
  auto projectSelectedType = options_.projectSelectedType();
  auto reqSchema =
      projectSelectedType ? getSelectedType() : TypeWithId::create(getType());

  columnSetters_.resize(numColumns);
  size_t outputCol = 0;

  for (size_t col = 0; col < numColumns; ++col) {
    auto colNode = schemaWithId_->childAt(col);
    if (columnSelector_.shouldReadNode(colNode->id())) {
      auto fileType = fileSchema->childAt(col);
      auto reqType = projectSelectedType
          ? reqSchema->type()->asRow().childAt(outputCol)
          : reqSchema->type()->asRow().childAt(col);

      // When projectSelectedType is true, output is compacted (0, 1, 2, ...)
      // When false, output matches file column position (col)
      size_t targetCol = projectSelectedType ? outputCol : col;
      columnSetters_[col] =
          ColumnSetter{targetCol, makeSetter(fileType, reqType)};
      ++outputCol;
    }
    // else: columnSetters_[col] remains nullopt
  }
}

template <typename ParseType, typename StoreType>
static TextRowReader::SetterFunction makeNumericSetter(
    std::function<std::optional<ParseType>(std::string_view)> parser) {
  return [parser](BaseVector* data, vector_size_t row, std::string_view value) {
    if (value.empty()) {
      data->setNull(row, true);
      return;
    }
    auto result = parser(value);
    auto* vec = data->as<FlatVector<StoreType>>();
    if (result.has_value()) {
      vec->set(row, static_cast<StoreType>(result.value()));
    } else {
      vec->setNull(row, true);
    }
  };
}

TextRowReader::SetterFunction TextRowReader::makeSetter(
    const TypePtr& fileType,
    const TypePtr& reqType,
    int depth) {
  // Handle same-type cases first (most common, no coercion)
  if (fileType->kind() == reqType->kind() && !reqType->isDate() &&
      !reqType->isShortDecimal() && !reqType->isLongDecimal()) {
    switch (fileType->kind()) {
      case TypeKind::BOOLEAN:
        return [](BaseVector* data, vector_size_t row, std::string_view value) {
          if (value.empty()) {
            data->setNull(row, true);
            return;
          }
          auto result = parseBoolean(value);
          auto* vec = data->as<FlatVector<bool>>();
          if (result.has_value())
            vec->set(row, result.value());
          else
            vec->setNull(row, true);
        };

      case TypeKind::TINYINT:
        return makeNumericSetter<int8_t, int8_t>(
            parseIntegerWithDecimal<int8_t>);
      case TypeKind::SMALLINT:
        return makeNumericSetter<int16_t, int16_t>(
            parseIntegerWithDecimal<int16_t>);
      case TypeKind::INTEGER:
        return makeNumericSetter<int32_t, int32_t>(
            parseIntegerWithDecimal<int32_t>);
      case TypeKind::BIGINT:
        return makeNumericSetter<int64_t, int64_t>(
            parseIntegerWithDecimal<int64_t>);

      case TypeKind::REAL:
        return [](BaseVector* data, vector_size_t row, std::string_view value) {
          if (value.empty()) {
            data->setNull(row, true);
            return;
          }
          auto result = parseFloating<float>(value);
          auto* vec = data->as<FlatVector<float>>();
          if (result.has_value()) {
            vec->set(row, result.value());
          } else {
            vec->setNull(row, true);
          }
        };

      case TypeKind::DOUBLE:
        return [](BaseVector* data, vector_size_t row, std::string_view value) {
          if (value.empty()) {
            data->setNull(row, true);
            return;
          }
          auto result = parseFloating<double>(value);
          auto* vec = data->as<FlatVector<double>>();
          if (result.has_value()) {
            vec->set(row, result.value());
          } else {
            vec->setNull(row, true);
          }
        };

      case TypeKind::VARCHAR:
        if (isEscaped_) {
          return
              [this](
                  BaseVector* data, vector_size_t row, std::string_view value) {
                auto* vec = data->as<FlatVector<StringView>>();
                std::string unescaped = unescapeValue(value);
                vec->set(row, StringView(unescaped));
              };
        } else {
          return
              [](BaseVector* data, vector_size_t row, std::string_view value) {
                auto* vec = data->as<FlatVector<StringView>>();
                vec->set(row, StringView(value.data(), value.size()));
              };
        }

      case TypeKind::VARBINARY:
        return [this](
                   BaseVector* data,
                   vector_size_t row,
                   std::string_view value) {
          if (value.empty()) {
            data->setNull(row, true);
            return;
          }
          auto* vec = data->as<FlatVector<StringView>>();
          size_t len = value.size();
          auto blen = encoding::Base64::calculateDecodedSize(value.data(), len);
          if (blen.hasValue()) {
            varBinBuf_->resize(blen.value());
            Status status = encoding::Base64::decode(
                value.data(), value.size(), varBinBuf_->data(), blen.value());
            if (status.ok()) {
              vec->set(row, StringView(varBinBuf_->data(), blen.value()));
              return;
            }
          }
          vec->set(row, StringView(value.data(), value.size()));
        };

      case TypeKind::TIMESTAMP:
        return [](BaseVector* data, vector_size_t row, std::string_view value) {
          if (value.empty()) {
            data->setNull(row, true);
            return;
          }
          auto* vec = data->as<FlatVector<Timestamp>>();
          std::string temp(value);
          auto result = util::Converter<TypeKind::TIMESTAMP>::tryCast(temp);
          if (result.hasValue())
            vec->set(row, result.value());
          else
            vec->setNull(row, true);
        };

      default:
        break; // Fall through to complex types / coercion
    }
  }

  // INTEGER -> DATE
  if (fileType->kind() == TypeKind::INTEGER && reqType->isDate()) {
    return [](BaseVector* data, vector_size_t row, std::string_view value) {
      if (value.empty()) {
        data->setNull(row, true);
        return;
      }
      auto* vec = data->as<FlatVector<int32_t>>();
      auto result = util::fromDateString(
          value.data(), value.size(), util::ParseMode::kPrestoCast);
      if (result.hasError()) {
        result = util::fromDateString(
            value.data(), value.size(), util::ParseMode::kSparkCast);
      }
      if (!result.hasError())
        vec->set(row, result.value());
      else
        vec->setNull(row, true);
    };
  }

  // Integer widening coercions
  if (fileType->kind() == TypeKind::TINYINT) {
    if (reqType->kind() == TypeKind::SMALLINT)
      return makeNumericSetter<int8_t, int16_t>(
          parseIntegerWithDecimal<int8_t>);
    if (reqType->kind() == TypeKind::INTEGER)
      return makeNumericSetter<int8_t, int32_t>(
          parseIntegerWithDecimal<int8_t>);
    if (reqType->kind() == TypeKind::BIGINT)
      return makeNumericSetter<int8_t, int64_t>(
          parseIntegerWithDecimal<int8_t>);
  }
  if (fileType->kind() == TypeKind::SMALLINT) {
    if (reqType->kind() == TypeKind::INTEGER)
      return makeNumericSetter<int16_t, int32_t>(
          parseIntegerWithDecimal<int16_t>);
    if (reqType->kind() == TypeKind::BIGINT)
      return makeNumericSetter<int16_t, int64_t>(
          parseIntegerWithDecimal<int16_t>);
  }
  if (fileType->kind() == TypeKind::INTEGER &&
      reqType->kind() == TypeKind::BIGINT) {
    return makeNumericSetter<int32_t, int64_t>(
        parseIntegerWithDecimal<int32_t>);
  }

  // REAL -> DOUBLE
  if (fileType->kind() == TypeKind::REAL &&
      reqType->kind() == TypeKind::DOUBLE) {
    return [](BaseVector* data, vector_size_t row, std::string_view value) {
      if (value.empty()) {
        data->setNull(row, true);
        return;
      }
      auto result = parseFloating<float>(value);
      auto* vec = data->as<FlatVector<double>>();
      if (result.has_value()) {
        vec->set(row, static_cast<double>(result.value()));
      } else {
        vec->setNull(row, true);
      }
    };
  }

  // BOOLEAN -> integer coercions
  if (fileType->kind() == TypeKind::BOOLEAN) {
    auto boolToInt = [](std::string_view value) -> std::optional<int64_t> {
      auto result = parseBoolean(value);
      return result.has_value() ? std::optional<int64_t>(result.value() ? 1 : 0)
                                : std::nullopt;
    };
    if (reqType->kind() == TypeKind::TINYINT)
      return makeNumericSetter<int64_t, int8_t>(boolToInt);
    if (reqType->kind() == TypeKind::SMALLINT)
      return makeNumericSetter<int64_t, int16_t>(boolToInt);
    if (reqType->kind() == TypeKind::INTEGER)
      return makeNumericSetter<int64_t, int32_t>(boolToInt);
    if (reqType->kind() == TypeKind::BIGINT)
      return makeNumericSetter<int64_t, int64_t>(boolToInt);
  }

  // BIGINT -> SHORT_DECIMAL
  if (fileType->kind() == TypeKind::BIGINT && reqType->isShortDecimal()) {
    auto decimalParams = getDecimalPrecisionScale(*reqType);
    return [decimalParams](
               BaseVector* data, vector_size_t row, std::string_view value) {
      if (value.empty()) {
        data->setNull(row, true);
        return;
      }
      auto* vec = data->as<FlatVector<int64_t>>();
      int64_t v = 0;
      auto status = DecimalUtil::castFromString(
          StringView(value.data(), static_cast<int32_t>(value.size())),
          decimalParams.first,
          decimalParams.second,
          v);
      if (status.ok())
        vec->set(row, v);
      else
        vec->setNull(row, true);
    };
  }

  // HUGEINT / LONG_DECIMAL
  if (fileType->kind() == TypeKind::HUGEINT) {
    if (reqType->isLongDecimal()) {
      auto decimalParams = getDecimalPrecisionScale(*reqType);
      return [decimalParams](
                 BaseVector* data, vector_size_t row, std::string_view value) {
        if (value.empty()) {
          data->setNull(row, true);
          return;
        }
        auto* vec = data->as<FlatVector<int128_t>>();
        int128_t v = 0;
        auto status = DecimalUtil::castFromString(
            StringView(value.data(), static_cast<int32_t>(value.size())),
            decimalParams.first,
            decimalParams.second,
            v);
        if (status.ok())
          vec->set(row, v);
        else
          vec->setNull(row, true);
      };
    } else {
      return [](BaseVector* data, vector_size_t row, std::string_view value) {
        if (value.empty()) {
          data->setNull(row, true);
          return;
        }
        auto* vec = data->as<FlatVector<int128_t>>();
        std::string temp(value);
        try {
          vec->set(row, HugeInt::parse(temp));
        } catch (...) {
          vec->setNull(row, true);
        }
      };
    }
  }

  // Complex types: ARRAY, MAP, ROW - create recursive setters with depth-based
  // delimiters
  if (fileType->kind() == TypeKind::ARRAY) {
    char elemDelim = contents_->serDeOptions.separators.at(depth);
    auto elementSetter =
        makeSetter(fileType->childAt(0), reqType->childAt(0), depth + 1);
    return [this, elemDelim, elementSetter](
               BaseVector* data, vector_size_t row, std::string_view value) {
      writeArrayWithSetter(
          data->as<ArrayVector>(), row, value, elemDelim, elementSetter);
    };
  }

  if (fileType->kind() == TypeKind::MAP) {
    char pairDelim = contents_->serDeOptions.separators.at(depth);
    char kvDelim = contents_->serDeOptions.separators.at(depth + 1);
    auto keySetter =
        makeSetter(fileType->childAt(0), reqType->childAt(0), depth + 2);
    auto valueSetter =
        makeSetter(fileType->childAt(1), reqType->childAt(1), depth + 2);
    return [this, pairDelim, kvDelim, keySetter, valueSetter](
               BaseVector* data, vector_size_t row, std::string_view value) {
      writeMapWithSetters(
          data->as<MapVector>(),
          row,
          value,
          pairDelim,
          kvDelim,
          keySetter,
          valueSetter);
    };
  }

  if (fileType->kind() == TypeKind::ROW) {
    char fieldDelim = contents_->serDeOptions.separators.at(depth);
    const auto& fileRowType = fileType->asRow();
    const auto& reqRowType = reqType->asRow();
    std::vector<SetterFunction> childSetters;
    for (size_t i = 0; i < fileRowType.size(); ++i) {
      childSetters.push_back(
          makeSetter(fileRowType.childAt(i), reqRowType.childAt(i), depth + 1));
    }
    return [this, fieldDelim, childSetters = std::move(childSetters)](
               BaseVector* data, vector_size_t row, std::string_view value) {
      writeRowWithSetters(
          data->as<RowVector>(), row, value, fieldDelim, childSetters);
    };
  }

  VELOX_FAIL(
      "Unsupported type coercion: {} -> {}",
      fileType->toString(),
      reqType->toString());
}

bool TextRowReader::loadBuffer() {
  // Get next chunk from stream - ZERO COPY, just get pointer
  const void* data = nullptr;
  int length = 0;
  bool hasData = false;

  if (contents_->compression != CompressionKind::CompressionKind_NONE) {
    hasData = contents_->decompressedInputStream->Next(&data, &length);
    atPhysicalEOF_ = !hasData;
  } else {
    hasData = contents_->inputStream->Next(&data, &length);
  }

  if (!hasData || length <= 0) {
    return false; // EOF - no more data available
  }

  // Zero-copy: just point to stream's buffer
  streamData_ = reinterpret_cast<const char*>(data);
  streamSize_ = length;
  streamPos_ = 0;
  return true;
}

std::pair<size_t, std::string_view> TextRowReader::findLine() {
  // Search for '\n' in current stream buffer
  if (streamPos_ >= static_cast<size_t>(streamSize_)) {
    return {std::string::npos, {}};
  }

  const char* remaining = streamData_ + streamPos_;
  size_t remainingSize = streamSize_ - streamPos_;
  const char* found =
      reinterpret_cast<const char*>(memchr(remaining, '\n', remainingSize));

  if (!found) {
    return {std::string::npos, {}};
  }

  size_t lineEnd = found - streamData_;
  size_t lineLen = lineEnd - streamPos_;

  // If we have leftover from previous chunk, combine it
  if (!leftover_.empty()) {
    leftover_.append(remaining, lineLen);
    return {lineEnd, std::string_view(leftover_)};
  }

  // Zero-copy: return view directly into stream buffer
  return {lineEnd, std::string_view(remaining, lineLen)};
}

void TextRowReader::skipToNextLine() {
  while (true) {
    auto [lineEnd, line] = findLine();
    if (lineEnd != std::string::npos) {
      size_t bytesSkipped = lineEnd - streamPos_ + 1;
      streamPos_ = lineEnd + 1;
      pos_ += bytesSkipped;
      leftover_.clear();
      return;
    }

    // No '\n' found - discard and load more (we're skipping, don't need
    // leftover)
    pos_ += streamSize_ - streamPos_;
    leftover_.clear();

    if (!loadBuffer()) {
      atEOF_ = true;
      return;
    }
  }
}

size_t TextRowReader::findDelimiter(
    std::string_view str,
    char delim,
    size_t start) const {
  if (!contents_->serDeOptions.isEscaped) {
    return str.find(delim, start);
  }

  const char escapeChar = contents_->serDeOptions.escapeChar;
  size_t pos = start;

  while (pos < str.size()) {
    size_t found = str.find(delim, pos);
    if (found == std::string_view::npos || found == 0) {
      return found;
    }

    size_t escapeCount = 0;
    size_t checkPos = found;
    while (checkPos > start && str[checkPos - 1] == escapeChar) {
      escapeCount++;
      checkPos--;
    }

    if (escapeCount % 2 == 1) {
      pos = found + 1;
      continue;
    }
    return found;
  }
  return std::string_view::npos;
}

bool TextRowReader::isNullValue(std::string_view value) const {
  return value == contents_->serDeOptions.nullString;
}

std::string TextRowReader::unescapeValue(std::string_view value) const {
  if (!contents_->serDeOptions.isEscaped) {
    return std::string(value);
  }

  std::string result;
  result.reserve(value.size());
  const char escapeChar = contents_->serDeOptions.escapeChar;

  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == escapeChar && i + 1 < value.size()) {
      char next = value[i + 1];
      switch (next) {
        case 'r':
          result.push_back('\r');
          break;
        case 'n':
          result.push_back('\n');
          break;
        default:
          result.push_back(next);
          break;
      }
      ++i;
    } else {
      result.push_back(value[i]);
    }
  }
  return result;
}

void TextRowReader::processLine(
    RowVector* result,
    vector_size_t row,
    std::string_view line) {
  const size_t numColumns = columnSetters_.size();

  size_t fieldStart = 0;

  for (size_t col = 0; col < numColumns; ++col) {
    size_t fieldEnd = findDelimiter(line, fieldDelim_, fieldStart);
    bool isLast = (fieldEnd == std::string_view::npos);

    // Check if this column has a precompiled setter (i.e., is selected)
    if (const auto& setterOpt = columnSetters_[col]) {
      std::string_view fieldValue = isLast
          ? line.substr(fieldStart)
          : line.substr(fieldStart, fieldEnd - fieldStart);

      // Check for null
      if (fieldValue == nullString_) {
        result->childAt(setterOpt->outputIndex)->setNull(row, true);
      } else {
        // Use precompiled setter - no type dispatch!
        setterOpt->setter(
            result->childAt(setterOpt->outputIndex).get(), row, fieldValue);
      }
    }

    if (isLast) {
      // Handle missing columns (set remaining selected columns to null)
      for (size_t i = col + 1; i < numColumns; ++i) {
        if (const auto& s = columnSetters_[i]) {
          result->childAt(s->outputIndex)->setNull(row, true);
        }
      }
      break;
    }
    fieldStart = fieldEnd + 1;
  }
}

void TextRowReader::writeArrayWithSetter(
    ArrayVector* arrayVec,
    vector_size_t row,
    std::string_view value,
    char elemDelim,
    const SetterFunction& elementSetter) {
  // Empty value represents an empty array.
  if (value.empty()) {
    auto* rawOffsets = arrayVec->offsets()->asMutable<vector_size_t>();
    auto* rawSizes = arrayVec->sizes()->asMutable<vector_size_t>();
    vector_size_t startOffset =
        row > 0 ? rawOffsets[row - 1] + rawSizes[row - 1] : 0;
    rawOffsets[row] = startOffset;
    rawSizes[row] = 0;
    arrayVec->setNull(row, false);
    return;
  }

  // Parse elements
  std::vector<std::string_view> elements;
  size_t pos = 0;
  while (pos <= value.size()) {
    size_t end = findDelimiter(value, elemDelim, pos);
    bool isLast = (end == std::string_view::npos);
    elements.push_back(
        isLast ? value.substr(pos) : value.substr(pos, end - pos));
    if (isLast)
      break;
    pos = end + 1;
  }

  // Set up offsets
  auto* rawOffsets = arrayVec->offsets()->asMutable<vector_size_t>();
  auto* rawSizes = arrayVec->sizes()->asMutable<vector_size_t>();

  vector_size_t startOffset =
      row > 0 ? rawOffsets[row - 1] + rawSizes[row - 1] : 0;
  rawOffsets[row] = startOffset;
  rawSizes[row] = static_cast<vector_size_t>(elements.size());

  // Resize elements vector
  auto* elementsVec = arrayVec->elements().get();
  if (elementsVec->size() <
      startOffset + static_cast<vector_size_t>(elements.size())) {
    elementsVec->resize(startOffset + elements.size());
  }

  // Use precompiled setter for each element
  for (size_t i = 0; i < elements.size(); ++i) {
    const auto& elem = elements[i];
    if (elem == nullString_) {
      elementsVec->setNull(startOffset + i, true);
    } else {
      elementSetter(elementsVec, startOffset + i, elem);
    }
  }
}

void TextRowReader::writeMapWithSetters(
    MapVector* mapVec,
    vector_size_t row,
    std::string_view value,
    char pairDelim,
    char kvDelim,
    const SetterFunction& keySetter,
    const SetterFunction& valueSetter) {
  // Empty value represents an empty map.
  if (value.empty()) {
    auto* rawOffsets = mapVec->offsets()->asMutable<vector_size_t>();
    auto* rawSizes = mapVec->sizes()->asMutable<vector_size_t>();
    vector_size_t startOffset =
        row > 0 ? rawOffsets[row - 1] + rawSizes[row - 1] : 0;
    rawOffsets[row] = startOffset;
    rawSizes[row] = 0;
    mapVec->setNull(row, false);
    return;
  }

  // Parse key-value pairs
  std::vector<std::pair<std::string_view, std::string_view>> pairs;
  size_t pos = 0;
  while (pos <= value.size()) {
    size_t pairEnd = findDelimiter(value, pairDelim, pos);
    bool isLast = (pairEnd == std::string_view::npos);
    std::string_view pair =
        isLast ? value.substr(pos) : value.substr(pos, pairEnd - pos);

    size_t kvSep = findDelimiter(pair, kvDelim, 0);
    if (kvSep != std::string_view::npos) {
      pairs.emplace_back(pair.substr(0, kvSep), pair.substr(kvSep + 1));
    } else {
      pairs.emplace_back(pair, std::string_view{});
    }

    if (isLast)
      break;
    pos = pairEnd + 1;
  }

  // Set up offsets
  auto* rawOffsets = mapVec->offsets()->asMutable<vector_size_t>();
  auto* rawSizes = mapVec->sizes()->asMutable<vector_size_t>();

  vector_size_t startOffset =
      row > 0 ? rawOffsets[row - 1] + rawSizes[row - 1] : 0;
  rawOffsets[row] = startOffset;
  rawSizes[row] = static_cast<vector_size_t>(pairs.size());

  // Resize key/value vectors
  auto* keysVec = mapVec->mapKeys().get();
  auto* valuesVec = mapVec->mapValues().get();
  if (keysVec->size() <
      startOffset + static_cast<vector_size_t>(pairs.size())) {
    keysVec->resize(startOffset + pairs.size());
    valuesVec->resize(startOffset + pairs.size());
  }

  // Use precompiled setters
  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto& [k, v] = pairs[i];
    if (k == nullString_) {
      keysVec->setNull(startOffset + i, true);
    } else {
      keySetter(keysVec, startOffset + i, k);
    }
    if (v == nullString_ || v.empty()) {
      valuesVec->setNull(startOffset + i, true);
    } else {
      valueSetter(valuesVec, startOffset + i, v);
    }
  }
}

void TextRowReader::writeRowWithSetters(
    RowVector* rowVec,
    vector_size_t row,
    std::string_view value,
    char fieldDelim,
    const std::vector<SetterFunction>& fieldSetters) {
  const size_t numFields = fieldSetters.size();

  size_t fieldStart = 0;
  for (size_t i = 0; i < numFields; ++i) {
    size_t fieldEnd = findDelimiter(value, fieldDelim, fieldStart);
    bool isLast = (fieldEnd == std::string_view::npos);

    std::string_view fieldValue = isLast
        ? value.substr(fieldStart)
        : value.substr(fieldStart, fieldEnd - fieldStart);

    auto* childVec = rowVec->childAt(i).get();
    if (childVec) {
      if (fieldValue == nullString_) {
        childVec->setNull(row, true);
      } else {
        fieldSetters[i](childVec, row, fieldValue);
      }
    }

    if (isLast) {
      // Remaining fields are null
      for (size_t j = i + 1; j < numFields; ++j) {
        auto* remainingVec = rowVec->childAt(j).get();
        if (remainingVec) {
          remainingVec->setNull(row, true);
        }
      }
      break;
    }
    fieldStart = fieldEnd + 1;
  }
}

// TODO: Performance optimizations to consider:
// 1. Reuse result vector directly instead of creating intermediate rowVecPtr
//    - Use ScanSpec::channel() for direct output mapping
//    - Avoid projectColumns() overhead when no filtering needed
// 3. Handle constant columns (partition columns) via constantColumnVectors_
// 4. Consider inline filtering for filter pushdown without intermediate vector
//    - Parse all fields, apply filter, only write passing rows
// Note: Current implementation uses intermediate vector + projectColumns()
// to support filter pushdown on non-projected columns.
uint64_t TextRowReader::next(
    uint64_t rows,
    VectorPtr& result,
    const Mutation* mutation) {
  if (atEOF_) {
    return 0;
  }

  auto& t = schemaWithId_;
  verify(
      t->type()->isRow(),
      "Top-level TypeKind of schema is not Row for file %s",
      getStreamNameData());

  auto projectSelectedType = options_.projectSelectedType();
  auto reqT =
      (projectSelectedType ? getSelectedType() : TypeWithId::create(getType()));
  verify(
      reqT->type()->isRow(),
      "Top-level TypeKind of schema is not Row for file %s",
      getStreamNameData());

  auto rowVecPtr = BaseVector::create<RowVector>(
      reqT->type(), static_cast<vector_size_t>(rows), &contents_->pool);

  // Handle deferred skip from constructor
  if (skipPartialLine_) {
    skipToNextLine();
    skipPartialLine_ = false;
  }

  while (rowsToSkip_ > 0 && !atEOF_) {
    skipToNextLine();
    --rowsToSkip_;
  }

  vector_size_t rowsRead = 0;

  while (rowsRead < static_cast<vector_size_t>(rows)) {
    auto [lineEnd, line] = findLine();

    if (lineEnd == std::string::npos) {
      // No '\n' found - save partial line to leftover and load more
      size_t remaining = streamSize_ - streamPos_;
      if (remaining > 0) {
        leftover_.append(streamData_ + streamPos_, remaining);
      }
      pos_ += remaining;

      if (!loadBuffer()) {
        // EOF - process remaining data as last line
        if (!leftover_.empty()) {
          std::string_view lastLine(leftover_);
          if (!lastLine.empty() && lastLine.back() == '\r') {
            lastLine.remove_suffix(1);
          }
          processLine(rowVecPtr.get(), rowsRead, lastLine);
          ++rowsRead;
          ++currentRow_;
          leftover_.clear();
        }
        atEOF_ = true;
        break;
      }
      // Loop back to search for '\n' in the newly loaded data.
      // This handles very long lines that span multiple I/O chunks.
      continue;
    }

    // Found '\n' - process the line
    // Handle \r\n line endings
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }

    size_t bytesConsumed = lineEnd - streamPos_ + 1;
    streamPos_ = lineEnd + 1;
    pos_ += bytesConsumed;
    leftover_.clear(); // Clear leftover after successful line

    processLine(rowVecPtr.get(), rowsRead, line);
    ++rowsRead;
    ++currentRow_;

    if (contents_->compression == CompressionKind::CompressionKind_NONE &&
        pos_ > limit_) {
      atEOF_ = true;
      break;
    }
  }

  rowVecPtr->resize(rowsRead);
  result = projectColumns(rowVecPtr, *scanSpec_, mutation);

  return rowsRead;
}

int64_t TextRowReader::nextRowNumber() {
  return atEOF_ ? -1 : static_cast<int64_t>(currentRow_) + 1;
}

int64_t TextRowReader::nextReadSize(uint64_t size) {
  // Text files don't have row count metadata like Parquet/DWRF,
  // so we just return the requested size (or -1 if at EOF).
  return atEOF_ ? -1 : static_cast<int64_t>(size);
}

void TextRowReader::updateRuntimeStats(
    dwio::common::RuntimeStatistics& /*stats*/) const {
  // No-op for non-selective reader.
}

void TextRowReader::resetFilterCaches() {
  // No-op for non-selective reader.
}

std::optional<size_t> TextRowReader::estimatedRowSize() const {
  return std::nullopt;
}

const ColumnSelector& TextRowReader::getColumnSelector() const {
  return columnSelector_;
}

std::shared_ptr<const TypeWithId> TextRowReader::getSelectedType() const {
  if (!selectedSchema_) {
    selectedSchema_ = columnSelector_.buildSelected();
  }
  return selectedSchema_;
}

uint64_t TextRowReader::getRowNumber() const {
  return currentRow_;
}

uint64_t TextRowReader::seekToRow(uint64_t rowNumber) {
  VELOX_CHECK_GT(
      rowNumber, currentRow_, "Text file cannot seek to earlier row");
  while (currentRow_ < rowNumber && !atEOF_) {
    skipToNextLine();
    ++currentRow_;
  }
  return currentRow_;
}

const RowReaderOptions& TextRowReader::getDefaultOpts() {
  static RowReaderOptions defaultOpts;
  return defaultOpts;
}

const std::shared_ptr<const RowType>& TextRowReader::getType() const {
  return contents_->schema;
}

bool TextRowReader::isSelectedField(
    const std::shared_ptr<const TypeWithId>& type) {
  return columnSelector_.shouldReadNode(type->id());
}

const char* TextRowReader::getStreamNameData() const {
  return contents_->input->getName().data();
}

uint64_t TextRowReader::getStreamLength() const {
  return contents_->input->getInputStream()->getLength();
}

TextReader::TextReader(
    const ReaderOptions& options,
    std::unique_ptr<BufferedInput> input)
    : options_{options} {
  auto schema = options_.fileSchema();
  VELOX_USER_CHECK_NOT_NULL(schema, "File schema for TEXT must be set.");

  if (!schema) {
    // Create dummy for testing.
    internalSchema_ = std::dynamic_pointer_cast<const RowType>(
        type::fbhive::HiveTypeParser().parse("struct<col0:string>"));
    DWIO_ENSURE_NOT_NULL(internalSchema_.get());
    schema = internalSchema_;
  }
  schemaWithId_ = TypeWithId::create(schema);
  contents_ = std::make_shared<FileContents>(options_.memoryPool(), schema);

  if (!contents_->schema->isRow()) {
    throw std::invalid_argument("file schema must be a ROW type");
  }

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

  // Set the SerDe options.
  contents_->serDeOptions = options_.serDeOptions();
  if (contents_->serDeOptions.isEscaped) {
    for (auto delim : contents_->serDeOptions.separators) {
      contents_->needsEscape.at(delim) = true;
    }
    contents_->needsEscape.at(contents_->serDeOptions.escapeChar) = true;
  }

  // Validate SerDe options.
  VELOX_CHECK(
      contents_->serDeOptions.nullString.compare("\r") != 0,
      "\'\\r\' is not allowed to be nullString");
  VELOX_CHECK(
      contents_->serDeOptions.nullString.compare("\n") != 0,
      "\'\\n\n is not allowed to be nullString");

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

CompressionKind TextReader::getCompression() const {
  return contents_->compression;
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

uint64_t TextReader::getFileLength() const {
  return contents_->fileLength;
}

} // namespace facebook::velox::text
