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

#include <utility>

#include "velox/common/encode/Base64.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/text/common/Common.h"
#include "velox/dwio/text/reader/TextReader.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::text {
ReaderBase::ReaderBase(
    dwio::common::ReaderOptions options,
    std::unique_ptr<dwio::common::BufferedInput> input)
    : options_{std::move(options)},
      input_{std::move(input)},
      schema_{options_.fileSchema()},
      memoryPool_(&options_.memoryPool()) {
  if (options_.scanSpec()) {
    typeWithId_ = std::shared_ptr<dwio::common::TypeWithId>(
        dwio::common::TypeWithId::create(schema_, *options_.scanSpec()));
  } else {
    typeWithId_ = std::shared_ptr<dwio::common::TypeWithId>(
        dwio::common::TypeWithId::create(schema_));
  }
}

std::unique_ptr<dwio::common::SeekableInputStream> ReaderBase::loadBlock(
    common::Region region) const {
  auto stream = input_->enqueue(region);
  input_->load(dwio::common::LogType::BLOCK);
  return stream;
}

TextRowReader::TextRowReader(
    const std::shared_ptr<ReaderBase>& reader,
    const dwio::common::RowReaderOptions& options)
    : readerBase_(reader),
      fileSchema_{readerBase_->schema()},
      fieldDelim_{readerBase_->serdeOptions().separators[0]},
      collectionDelim_{readerBase_->serdeOptions().separators[1]},
      mapKeyDelim_{readerBase_->serdeOptions().separators[2]},
      row_{0},
      skipRows_{options.skipRows()},
      fileLength_{readerBase_->fileLength()},
      dataOffset_{options.offset()},
      dataEndOffset_{std::min(dataOffset_ + options.length(), fileLength_)},
      blockEndOffset_{dataOffset_},
      bufferPtr_{nullptr},
      bufferSize_{0},
      bufferOffset_{0} {
  skippedPartialStartLine_ = dataOffset_ == 0 ? true : false;

  auto& scanSpec = options.scanSpec();
  auto& childSpecs = scanSpec->children();
  for (const auto& childSpec : childSpecs) {
    if (!childSpec->projectOut()) {
      continue;
    }
    if (childSpec->isConstant()) {
      constantColumnVectors_[childSpec->channel()] = childSpec->constantValue();
    } else if (childSpec->readFromFile()) {
      auto fileIndex = fileSchema_->getChildIdx(childSpec->fieldName());
      auto channel = childSpec->channel();
      fileIndexToSetters_[fileIndex] = {
          channel, makeSetter(fileSchema_->childAt(fileIndex))};
    }
  }
}

uint64_t TextRowReader::next(
    uint64_t size,
    VectorPtr& result,
    const dwio::common::Mutation* /*mutation*/) {
  if (dataOffset_ > dataEndOffset_ || dataOffset_ >= fileLength_) {
    // If we already passed the split boundary, we should not read any more
    // rows.
    return 0;
  }

  auto rowResult = result->as<RowVector>();
  rowResult->ensureWritable(SelectivityVector(size, true));

  int32_t row = 0;
  bool pastSplitBoundary = false;

  while (row < size) {
    // Load new block if needed
    if (dataOffset_ == blockEndOffset_) {
      size_t readSize = 0;
      if (dataOffset_ < dataEndOffset_) {
        // Read normal block size or remaining bytes in split
        readSize = std::min(kBlockSize, dataEndOffset_ - dataOffset_);
      } else {
        // Past split boundary: read extra data to finish last row
        readSize = std::min(kEstimatedRowSize, fileLength_ - dataOffset_);
      }

      stream_ = readerBase_->loadBlock({dataOffset_, readSize});
      blockEndOffset_ = dataOffset_ + readSize;

      // Reset buffer state
      bufferPtr_ = nullptr;
      bufferSize_ = 0;
      bufferOffset_ = 0;
    }

    // Read buffer if fully consumed
    if (bufferOffset_ >= bufferSize_) {
      if (!stream_->Next(
              reinterpret_cast<const void**>(&bufferPtr_), &bufferSize_)) {
        break;
      }
      bufferOffset_ = 0;
    }

    // Skip first partial row if not yet skipped
    if (!skippedPartialStartLine_) {
      std::string_view remainingStr(
          bufferPtr_ + bufferOffset_, bufferSize_ - bufferOffset_);
      if (remainingStr.empty()) {
        // If the buffer is empty, we can't skip anything
        continue;
      }
      // If the first row is partial, skip it
      auto end = remainingStr.find(TextFileTraits::kNewLine);
      if (end != std::string::npos) {
        // Move offset past the newline
        auto skipLen = end + 1;
        bufferOffset_ += skipLen;
        dataOffset_ += skipLen;
        skippedPartialStartLine_ = true;
      } else {
        // No newline found, skip entire buffer
        dataOffset_ += remainingStr.size();
        bufferOffset_ = bufferSize_;
        continue;
      }
    }

    // Parse lines from current buffer
    std::string_view remainingStr(
        bufferPtr_ + bufferOffset_, bufferSize_ - bufferOffset_);
    while (!remainingStr.empty() && row < size) {
      auto end = remainingStr.find(TextFileTraits::kNewLine);
      if (end == std::string::npos) {
        leftover_.append(remainingStr);
        bufferOffset_ = bufferSize_;
        dataOffset_ += remainingStr.size();
        break;
      }

      if (!leftover_.empty()) {
        leftover_.append(remainingStr, 0, end);
        if (skipRows_ > 0) {
          --skipRows_;
        } else {
          processLine(rowResult, row, leftover_);
          ++row;
        }
        leftover_.clear();
      } else {
        if (skipRows_ > 0) {
          --skipRows_;
        } else {
          processLine(rowResult, row, remainingStr.substr(0, end));
          ++row;
        }
      }

      // Advance buffer and data offsets past this line (+1 for newline)
      bufferOffset_ += end + 1;
      dataOffset_ += end + 1;

      remainingStr.remove_prefix(end + 1);

      if (dataOffset_ > dataEndOffset_) {
        pastSplitBoundary = true;
        break;
      }
    }

    if (pastSplitBoundary) {
      // Stop reading new rows once the first full row past boundary is
      // processed
      break;
    }
  }

  result->resize(row);
  for (const auto& [channel, valueVector] : constantColumnVectors_) {
    rowResult->childAt(channel) =
        BaseVector::wrapInConstant(row, 0, valueVector);
  }
  row_ += row;
  return row;
}

int64_t TextRowReader::nextReadSize(uint64_t size) {
  if (dataOffset_ > dataEndOffset_) {
    return kAtEnd;
  } else {
    return 0;
  }
}

void TextRowReader::processLine(
    RowVector* result,
    int32_t row,
    std::string_view line) {
  std::size_t columnIndex = 0;
  std::size_t start = 0;

  while (start <= line.size()) {
    VELOX_CHECK_LT(
        columnIndex, fileSchema_->size(), "Too many columns in line");

    std::size_t end = line.find(fieldDelim_, start);
    bool isLast = (end == std::string::npos);
    std::string_view token =
        isLast ? line.substr(start) : line.substr(start, end - start);
    auto it = fileIndexToSetters_.find(columnIndex);
    if (it != fileIndexToSetters_.end()) {
      auto channel = it->second.first;
      auto setter = it->second.second;
      setter(result->childAt(channel), row, token);
    }

    columnIndex++;
    if (isLast) {
      break;
    }
    start = end + 1;
  }
}

TextRowReader::SetterFunction TextRowReader::makeSetter(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return makePrimitiveSetter<TypeKind::BOOLEAN>();
    case TypeKind::TINYINT:
      return makePrimitiveSetter<TypeKind::TINYINT>();
    case TypeKind::SMALLINT:
      return makePrimitiveSetter<TypeKind::SMALLINT>();
    case TypeKind::INTEGER:
      if (type->isDate()) {
        return [](VectorPtr& vec, vector_size_t row, std::string_view val) {
          auto flat = vec->as<FlatVector<int32_t>>();
          if (val.empty() || TextFileTraits::kNullData == val) {
            flat->setNull(row, true);
          } else {
            flat->set(row, castFromDateString(val));
          }
        };
      } else {
        return makePrimitiveSetter<TypeKind::INTEGER>();
      }
    case TypeKind::BIGINT:
      return makePrimitiveSetter<TypeKind::BIGINT>();
    case TypeKind::REAL:
      return makePrimitiveSetter<TypeKind::REAL>();
    case TypeKind::DOUBLE:
      return makePrimitiveSetter<TypeKind::DOUBLE>();
    case TypeKind::TIMESTAMP:
      return makePrimitiveSetter<TypeKind::TIMESTAMP>();
    case TypeKind::VARCHAR:
      return [](VectorPtr& vec, vector_size_t row, std::string_view val) {
        if (TextFileTraits::kNullData == val) {
          vec->as<FlatVector<StringView>>()->setNull(row, true);
        } else {
          vec->as<FlatVector<StringView>>()->set(
              row, StringView(val.data(), val.size()));
        }
      };
    case TypeKind::VARBINARY:
      return [](VectorPtr& vec, vector_size_t row, std::string_view val) {
        if (TextFileTraits::kNullData == val) {
          vec->as<FlatVector<StringView>>()->setNull(row, true);
        } else {
          auto decodedValue =
              encoding::Base64::decode({val.data(), val.size()});
          vec->as<FlatVector<StringView>>()->set(row, StringView(decodedValue));
        }
      };
    case TypeKind::ARRAY: {
      auto arrayType = std::dynamic_pointer_cast<const ArrayType>(type);
      auto elementSetter = makeSetter(arrayType->elementType());
      return [this, arrayType, elementSetter = std::move(elementSetter)](
                 VectorPtr& vector, vector_size_t row, std::string_view value) {
        writeArrayValue(elementSetter, vector, row, value);
      };
    }
    case TypeKind::ROW: {
      auto rowType = std::dynamic_pointer_cast<const RowType>(type);
      std::vector<SetterFunction> childSetters;
      for (auto& child : rowType->children()) {
        childSetters.push_back(makeSetter(child));
      }
      return [this, childSetters = std::move(childSetters)](
                 VectorPtr& vector, vector_size_t row, std::string_view value) {
        writeRowValue(childSetters, vector, row, value);
      };
    }
    case TypeKind::MAP: {
      auto mapType = std::dynamic_pointer_cast<const MapType>(type);
      auto keySetter = makeSetter(mapType->keyType());
      auto valueSetter = makeSetter(mapType->valueType());

      return [this,
              keySetter = std::move(keySetter),
              valueSetter = std::move(valueSetter),
              mapType](
                 VectorPtr& vector, vector_size_t row, std::string_view value) {
        writeMapValue(keySetter, valueSetter, vector, row, value);
      };
    }
    default:
      VELOX_UNSUPPORTED("Unsupported type: {}", type->toString());
  }
}

template <TypeKind Kind>
TextRowReader::SetterFunction TextRowReader::makePrimitiveSetter() {
  return [this](VectorPtr& vec, vector_size_t row, std::string_view val) {
    using TCpp = typename TypeTraits<Kind>::NativeType;
    auto flat = vec->as<FlatVector<TCpp>>();
    if (val.empty() || TextFileTraits::kNullData == val) {
      flat->setNull(row, true);
    } else {
      flat->set(row, castFromString<Kind>(val));
    }
  };
}

void TextRowReader::writeArrayValue(
    const SetterFunction& setter,
    VectorPtr& columnVector,
    int32_t row,
    std::string_view value) const {
  auto arrayVector = columnVector->as<ArrayVector>();
  auto elementsVector = arrayVector->elements();

  std::vector<std::string_view> elements;
  std::size_t start = 0;
  while (start <= value.size()) {
    std::size_t end = value.find(collectionDelim_, start);
    bool isLast = (end == std::string::npos);
    std::string_view token =
        isLast ? value.substr(start) : value.substr(start, end - start);
    elements.push_back(token);
    if (isLast) {
      break;
    }
    start = end + 1;
  }

  const vector_size_t offset = elementsVector->size();
  const vector_size_t length = elements.size();
  elementsVector->resize(offset + length);
  for (size_t i = 0; i < length; ++i) {
    setter(elementsVector, offset + i, elements[i]);
  }

  arrayVector->setOffsetAndSize(row, offset, length);
}

void TextRowReader::writeRowValue(
    const std::vector<SetterFunction>& childSetters,
    VectorPtr& vector,
    vector_size_t row,
    std::string_view value) const {
  auto rowVector = vector->as<RowVector>();
  auto& children = rowVector->children();

  std::size_t columnIndex = 0;
  std::size_t start = 0;

  auto childrenSize = children.size();
  auto valueSize = value.size();
  while (columnIndex < childrenSize && start <= valueSize) {
    std::size_t end = value.find(collectionDelim_, start);
    bool isLast = (end == std::string::npos);
    std::string_view token =
        isLast ? value.substr(start) : value.substr(start, end - start);
    if (token == TextFileTraits::kNullData) {
      children[columnIndex]->setNull(row, true);
    } else {
      childSetters[columnIndex](children[columnIndex], row, token);
    }
    columnIndex++;
    if (isLast) {
      break;
    }
    start = end + 1;
  }

  for (auto i = columnIndex; i < childrenSize; ++i) {
    children[i]->setNull(row, true);
  }
}

void TextRowReader::writeMapValue(
    const SetterFunction& keySetter,
    const SetterFunction& valueSetter,
    VectorPtr& vector,
    vector_size_t row,
    std::string_view value) const {
  auto* mapVector = vector->as<MapVector>();
  if (value.empty()) {
    mapVector->setNull(row, true);
    return;
  }

  auto& keyVector = mapVector->mapKeys();
  auto& valueVector = mapVector->mapValues();
  std::vector<std::pair<std::string_view, std::string_view>> entries;

  std::size_t start = 0;
  while (start <= value.size()) {
    std::size_t entryEnd = value.find(collectionDelim_, start);
    bool isLast = (entryEnd == std::string::npos);
    if (isLast) {
      entryEnd = value.size();
    }

    std::string_view key;
    std::string_view val;
    std::size_t kvSep = value.find(mapKeyDelim_, start);
    if (kvSep == std::string::npos || kvSep >= entryEnd) {
      // No key-value delimiter found, treat entire token as key, value=null
      key = value.substr(start, entryEnd - start);
      val = TextFileTraits::kNullData;
    } else {
      key = value.substr(start, kvSep - start);
      val = value.substr(kvSep + 1, entryEnd - kvSep - 1);
    }

    // Skip entry if key is null representation ("\N")
    if (key != TextFileTraits::kNullData) {
      entries.emplace_back(key, val);
    }

    if (isLast) {
      break;
    }
    start = entryEnd + 1;
  }

  const vector_size_t offset = keyVector->size();
  const vector_size_t length = entries.size();

  keyVector->resize(offset + length);
  valueVector->resize(offset + length);

  for (size_t i = 0; i < length; ++i) {
    keySetter(keyVector, offset + i, entries[i].first);
    valueSetter(valueVector, offset + i, entries[i].second);
  }

  mapVector->setOffsetAndSize(row, offset, length);
}

template <TypeKind KIND>
typename TypeTraits<KIND>::NativeType TextRowReader::castFromString(
    const std::string_view& value) {
  auto result = util::Converter<KIND>::tryCast(folly::StringPiece(value));
  if (result.hasError()) {
    VELOX_FAIL("TextRowReader: '{}'", result.error().message());
  }
  VELOX_CHECK(!result.hasError());
  return result.value();
}

int32_t TextRowReader::castFromDateString(const std::string_view& value) {
  auto result = util::fromDateString(
      value.data(), value.size(), util::ParseMode::kPrestoCast);
  if (result.hasError()) {
    VELOX_FAIL("TextRowReader: '{}'", result.error().message());
  }
  VELOX_CHECK(!result.hasError());
  return result.value();
}

TextReader::TextReader(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options)
    : readerBase_{std::make_shared<ReaderBase>(options, std::move(input))} {}
} // namespace facebook::velox::text
