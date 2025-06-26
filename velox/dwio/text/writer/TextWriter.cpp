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

#include "velox/dwio/text/writer/TextWriter.h"
#include "velox/common/encode/Base64.h"

#include <unordered_map>
#include <utility>

namespace facebook::velox::text {

using dwio::common::SerDeOptions;

template <typename T>
std::optional<std::string> toTextStr(T val) {
  return std::optional(std::to_string(val));
}

template <>
std::optional<std::string> toTextStr<bool>(bool val) {
  return val ? std::optional("true") : std::optional("false");
}

template <>
std::optional<std::string> toTextStr<float>(float val) {
  if (std::isnan(val)) {
    return std::optional("NaN");
  } else if (std::isinf(val)) {
    return std::optional("Infinity");
  } else {
    return {std::to_string(val)};
  }
}

template <>
std::optional<std::string> toTextStr<double>(double val) {
  if (std::isnan(val)) {
    return std::optional("NaN");
  } else if (std::isinf(val)) {
    return std::optional("Infinity");
  } else {
    return {std::to_string(val)};
  }
}

template <>
std::optional<std::string> toTextStr<Timestamp>(Timestamp val) {
  TimestampToStringOptions options;
  options.dateTimeSeparator = ' ';
  options.precision = TimestampPrecision::kMilliseconds;
  return {val.toString(options)};
}

TextWriter::TextWriter(
    RowTypePtr schema,
    std::unique_ptr<dwio::common::FileSink> sink,
    const std::shared_ptr<text::WriterOptions>& options,
    SerDeOptions serDeOptions)
    : schema_(std::move(schema)),
      bufferedWriterSink_(std::make_unique<BufferedWriterSink>(
          std::move(sink),
          options->memoryPool->addLeafChild(fmt::format(
              "{}.text_writer_node.{}",
              options->memoryPool->name(),
              folly::to<std::string>(folly::Random::rand64()))),
          options->defaultFlushCount)),
      depth_(0),
      serDeOptions_(std::move(serDeOptions)),
      decodedVectorMap_{} {}

void TextWriter::setDelimiters(const std::vector<char>& delimiters) {
  VELOX_CHECK_EQ(delimiters.size(), 8, "8 delimiters expected");
  for (int i = 0; i < delimiters.size(); ++i) {
    serDeOptions_.separators[i] = delimiters[i];
  }
}

DecodedVectorInfo TextWriter::getDecodedVectorInfo(
    const BaseVector* vector) const {
  auto it = decodedVectorMap_.find(vector);
  VELOX_CHECK(
      it != decodedVectorMap_.end(), "DecodedVector not found for vector");
  return it->second;
}

char TextWriter::getDelimiterForDepth(uint8_t depth) const {
  VELOX_CHECK_LT(
      depth,
      serDeOptions_.separators.size(),
      "Depth {} exceeds maximum supported depth",
      depth);
  return serDeOptions_.separators[depth];
}

char TextWriter::getCurrentCollectionDelim(const BaseVector* vector) const {
  const auto& info = getDecodedVectorInfo(vector);
  return getDelimiterForDepth(info.depth);
}

void TextWriter::createDecodedVector(uint8_t depth, const BaseVector* vecPtr) {
  /// TODO: Match this with Reader's depth support for compatibility.
  VELOX_CHECK_LE(depth, 4, "Text writer does not support more than 4 levels.");

  VELOX_CHECK(
      decodedVectorMap_.find(vecPtr) == decodedVectorMap_.end(),
      "This vector has already been processed");

  auto type = vecPtr->type();
  SelectivityVector rows(vecPtr->size());
  auto decodedVector = std::make_shared<DecodedVector>(*vecPtr, rows);
  decodedVectorMap_.insert(
      std::make_pair(vecPtr, DecodedVectorInfo{depth, decodedVector, type}));

  if (!type->isPrimitiveType()) {
    switch (type->kind()) {
      case TypeKind::ARRAY: {
        const auto arrVecPtr = decodedVector->base()->as<ArrayVector>();
        createDecodedVector(depth + 1, arrVecPtr->elements().get());
        break;
      }
      case TypeKind::MAP: {
        const auto mapVecPtr = decodedVector->base()->as<MapVector>();
        createDecodedVector(depth + 1, mapVecPtr->mapKeys().get());
        createDecodedVector(depth + 1, mapVecPtr->mapValues().get());
        break;
      }
      case TypeKind::ROW: {
        const auto rowVecPtr = decodedVector->base()->as<RowVector>();
        const auto numColumns = rowVecPtr->childrenSize();
        for (column_index_t column = 0; column < numColumns; ++column) {
          createDecodedVector(depth + 1, rowVecPtr->childAt(column).get());
        }
        break;
      }
      default:
        VELOX_NYI(
            "Text writer does not support type {}", vecPtr->type()->toString());
    }
  }
}

void TextWriter::writeCellValueRecursively(
    const BaseVector* decodedColumnVectorPtr,
    vector_size_t rowToWrite) {
  const auto& decodedVectorInfo = getDecodedVectorInfo(decodedColumnVectorPtr);
  const auto& decodedVector = decodedVectorInfo.decVecPtr;

  if (decodedVector->isNullAt(rowToWrite)) {
    return;
  }

  const auto type = decodedColumnVectorPtr->type();
  VELOX_CHECK(!type->isPrimitiveType(), "Complex type expected");

  switch (type->kind()) {
    case TypeKind::ARRAY: {
      // ARRAY vector members
      const auto& arrVecPtr = decodedVector->base()->as<ArrayVector>();
      const auto& indices = decodedVector->indices();
      const auto& size = arrVecPtr->sizeAt(indices[rowToWrite]);
      const auto& offset = arrVecPtr->offsetAt(indices[rowToWrite]);

      // Get decoded vector for T in ARRAY<T>.
      const auto& decodedChildVectorInfo =
          getDecodedVectorInfo(arrVecPtr->elements().get());
      const auto& decodedChildVector = decodedChildVectorInfo.decVecPtr;

      for (vector_size_t i = 0; i < size; ++i) {
        if (i != 0) {
          bufferedWriterSink_->write(
              getCurrentCollectionDelim(decodedColumnVectorPtr));
        }

        const auto& valueType = decodedChildVectorInfo.type;
        if (valueType->isPrimitiveType()) {
          writePrimitiveCellValue(decodedChildVector, valueType, offset + i);
        } else {
          writeCellValueRecursively(decodedChildVector->base(), offset + i);
        }
      }
      break;
    }
    case TypeKind::MAP: {
      // MAP vector members
      const auto& mapVecPtr = decodedVector->base()->as<MapVector>();
      const auto& indices = decodedVector->indices();
      const auto& size = mapVecPtr->sizeAt(indices[rowToWrite]);
      const auto& offset = mapVecPtr->offsetAt(indices[rowToWrite]);

      // Get decoded vectors and decoded vectors info for K and V in MAP<K, V>.
      const auto& decodedKeyVectorInfo =
          getDecodedVectorInfo(mapVecPtr->mapKeys().get());
      const auto& decodedValueVectorInfo =
          getDecodedVectorInfo(mapVecPtr->mapValues().get());
      const auto& decodedKeyVector = decodedKeyVectorInfo.decVecPtr;
      const auto& decodedValueVector = decodedValueVectorInfo.decVecPtr;

      for (vector_size_t i = 0; i < size; ++i) {
        if (i != 0) {
          bufferedWriterSink_->write(
              getCurrentCollectionDelim(decodedColumnVectorPtr));
        }

        const auto& keyType = decodedKeyVectorInfo.type;
        VELOX_CHECK(
            keyType->isPrimitiveType(),
            "Map key is expected to be primitive type. Got {} instead",
            keyType->toString());

        writePrimitiveCellValue(decodedKeyVector, keyType, offset + i);
        const auto& mapInfo = getDecodedVectorInfo(decodedColumnVectorPtr);
        bufferedWriterSink_->write(getDelimiterForDepth(mapInfo.depth + 1));

        const auto& valueType = decodedValueVectorInfo.type;
        if (valueType->isPrimitiveType()) {
          writePrimitiveCellValue(decodedValueVector, valueType, offset + i);
        } else {
          writeCellValueRecursively(decodedValueVector->base(), offset + i);
        }
      }
      break;
    }
    case TypeKind::ROW: {
      const auto rowVecPtr =
          decodedVectorInfo.decVecPtr->base()->as<RowVector>();
      const auto numColumns = rowVecPtr->childrenSize();

      // Outermost layer, write newline here.
      if (decodedVectorInfo.depth == 0) {
        const auto& rows = decodedColumnVectorPtr->size();
        for (vector_size_t row = 0; row < rows; ++row) {
          for (column_index_t col = 0; col < numColumns; ++col) {
            if (col != 0) {
              bufferedWriterSink_->write(serDeOptions_.separators[0]);
            }

            const auto& decodedChildVectorInfo =
                getDecodedVectorInfo(rowVecPtr->childAt(col).get());
            const auto& decodedChildVector = decodedChildVectorInfo.decVecPtr;

            const auto& childType = decodedChildVectorInfo.type;
            if (childType->isPrimitiveType()) {
              writePrimitiveCellValue(decodedChildVector, childType, row);
            } else {
              writeCellValueRecursively(decodedChildVector->base(), row);
            }
          }

          bufferedWriterSink_->write(serDeOptions_.newLine);
        }
      } else {
        // Nested ROW case
        for (column_index_t col = 0; col < numColumns; ++col) {
          if (col != 0) {
            bufferedWriterSink_->write(serDeOptions_.separators[0]);
          }

          const auto& decodedChildVectorInfo =
              getDecodedVectorInfo(rowVecPtr->childAt(col).get());
          const auto& childDecodedVector = decodedChildVectorInfo.decVecPtr;
          const auto& childType = decodedChildVectorInfo.type;

          if (childType->isPrimitiveType()) {
            writePrimitiveCellValue(childDecodedVector, childType, rowToWrite);
          } else {
            writeCellValueRecursively(childDecodedVector->base(), rowToWrite);
          }
        }
      }
      break;
    }
    default:
      VELOX_NYI("Text writer does not support type {}", type->toString());
  }
}

void TextWriter::writePrimitiveCellValue(
    const std::shared_ptr<DecodedVector>& decodedColumnVector,
    const TypePtr& type,
    vector_size_t row) {
  std::optional<std::string> dataStr;
  std::optional<StringView> dataSV;

  if (decodedColumnVector->isNullAt(row)) {
    bufferedWriterSink_->write(
        serDeOptions_.nullString.data(), serDeOptions_.nullString.length());
    return;
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      dataStr =
          toTextStr(folly::to<bool>(decodedColumnVector->valueAt<bool>(row)));
      break;
    case TypeKind::TINYINT:
      dataStr = toTextStr(decodedColumnVector->valueAt<int8_t>(row));
      break;
    case TypeKind::SMALLINT:
      dataStr = toTextStr(decodedColumnVector->valueAt<int16_t>(row));
      break;
    case TypeKind::INTEGER:
      dataStr = toTextStr(decodedColumnVector->valueAt<int32_t>(row));
      break;
    case TypeKind::BIGINT:
      dataStr = toTextStr(decodedColumnVector->valueAt<int64_t>(row));
      break;
    case TypeKind::REAL:
      dataStr = toTextStr(decodedColumnVector->valueAt<float>(row));
      break;
    case TypeKind::DOUBLE:
      dataStr = toTextStr(decodedColumnVector->valueAt<double>(row));
      break;
    case TypeKind::TIMESTAMP:
      dataStr = toTextStr(decodedColumnVector->valueAt<Timestamp>(row));
      break;
    case TypeKind::VARCHAR:
      dataSV = std::optional(decodedColumnVector->valueAt<StringView>(row));
      break;
    case TypeKind::VARBINARY: {
      auto data = decodedColumnVector->valueAt<StringView>(row);
      dataStr =
          std::optional(encoding::Base64::encode(data.data(), data.size()));
      break;
    }
    case TypeKind::UNKNOWN:
      [[fallthrough]];
    default:
      VELOX_NYI("{} is not supported yet in TextWriter", type->kind());
  }

  if (dataStr.has_value()) {
    VELOX_CHECK(!dataSV.has_value());
    bufferedWriterSink_->write(
        dataStr.value().data(), dataStr.value().length());
    return;
  }

  VELOX_CHECK(dataSV.has_value());
  bufferedWriterSink_->write(dataSV.value().data(), dataSV.value().size());
}

void TextWriter::write(const VectorPtr& data) {
  VELOX_CHECK_EQ(
      data->encoding(),
      VectorEncoding::Simple::ROW,
      "Text writer expects row vector input");

  /**
  TODO: verify if we could allow downcasting in the future. Current Java
  implementation (coordinator layer) allows upcasting (i.e. from int32_t to
  int64_t) but fails the query for downcasts. We need to check if for
  upcastings, does coordinator have another layer of casting (changes schema
  type before reaching worker nodes) or do we have to handle it in writer
  */
  VELOX_CHECK(
      data->type()->equivalent(*schema_),
      "The file schema type should be equal with the input row vector type.");

  createDecodedVector(0, data.get());
  writeCellValueRecursively(data.get(), 0);
}

void TextWriter::flush() {
  bufferedWriterSink_->flush();
}

void TextWriter::close() {
  bufferedWriterSink_->close();
}

void TextWriter::abort() {
  bufferedWriterSink_->abort();
}

std::unique_ptr<dwio::common::Writer> TextWriterFactory::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const std::shared_ptr<dwio::common::WriterOptions>& options) {
  auto textOptions = std::dynamic_pointer_cast<text::WriterOptions>(options);
  VELOX_CHECK_NOT_NULL(
      textOptions, "Text writer factory expected a Text WriterOptions object.");
  return std::make_unique<TextWriter>(
      asRowType(options->schema), std::move(sink), textOptions);
}

std::unique_ptr<dwio::common::WriterOptions>
TextWriterFactory::createWriterOptions() {
  return std::make_unique<text::WriterOptions>();
}

} // namespace facebook::velox::text
