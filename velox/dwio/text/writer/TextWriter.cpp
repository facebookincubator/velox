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

#include <utility>
#include "velox/common/base/Pointers.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::text {

static std::string encodeColumnCellValue(
    const std::shared_ptr<DecodedVector>& decodedColumnVector,
    const TypePtr& type,
    vector_size_t row) {
  if (decodedColumnVector->isNullAt(row)) {
    return TextWriter::nullData;
  }

  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return std::to_string(decodedColumnVector->valueAt<bool>(row));
    case TypeKind::TINYINT:
      return std::to_string(decodedColumnVector->valueAt<int8_t>(row));
    case TypeKind::SMALLINT:
      return std::to_string(decodedColumnVector->valueAt<int16_t>(row));
    case TypeKind::INTEGER:
      return std::to_string(decodedColumnVector->valueAt<int32_t>(row));
    case TypeKind::BIGINT:
      return std::to_string(decodedColumnVector->valueAt<int64_t>(row));
    case TypeKind::REAL:
      return std::to_string(decodedColumnVector->valueAt<float>(row));
    case TypeKind::DOUBLE:
      return std::to_string(decodedColumnVector->valueAt<double>(row));
    case TypeKind::VARCHAR:
      return decodedColumnVector->valueAt<StringView>(row).getString();
    case TypeKind::TIMESTAMP:
      return decodedColumnVector->valueAt<Timestamp>(row).toString();
    case TypeKind::VARBINARY:
      return decodedColumnVector->valueAt<StringView>(row).getString();
    case TypeKind::ARRAY:
      [[fallthrough]];
    case TypeKind::MAP:
      [[fallthrough]];
    case TypeKind::ROW:
      [[fallthrough]];
    case TypeKind::UNKNOWN:
      [[fallthrough]];
    default:
      VELOX_NYI("{} is not supported yet in TextWriter", type->kind());
  }
}

TextWriter::TextWriter(
    RowTypePtr schema,
    std::unique_ptr<dwio::common::FileSink> sink,
    const std::shared_ptr<text::WriterOptions>& options)
    : schema_(std::move(schema)) {
  bufferedWriterSink_ = std::make_unique<BufferedWriterSink>(
      std::move(sink),
      options->memoryPool->addLeafChild(fmt::format(
          "{}.text_writer_node.{}",
          options->memoryPool->name(),
          folly::to<std::string>(folly::Random::rand64()))),
      options->defaultFlushCount);
}

void TextWriter::write(const VectorPtr& data) {
  VELOX_CHECK_EQ(
      data->encoding(),
      VectorEncoding::Simple::ROW,
      "Text writer expects row vector input");
  VELOX_CHECK(
      data->type()->equivalent(*schema_),
      "The file schema type should be equal with the input row vector type.");
  const RowVector* dataRowVector = data->as<RowVector>();

  std::vector<std::shared_ptr<DecodedVector>> decodedColumnVectors;
  const auto numColumns = dataRowVector->childrenSize();
  for (size_t column = 0; column < numColumns; ++column) {
    auto decodedColumnVector = std::make_shared<DecodedVector>(DecodedVector(
        *dataRowVector->childAt(column),
        SelectivityVector(dataRowVector->size())));
    decodedColumnVectors.push_back(decodedColumnVector);
  }

  for (vector_size_t row = 0; row < data->size(); ++row) {
    for (size_t column = 0; column < numColumns; ++column) {
      if (column != 0) {
        bufferedWriterSink_->write(SOH);
      }
      auto columnData = encodeColumnCellValue(
          decodedColumnVectors.at(column), schema_->childAt(column), row);
      bufferedWriterSink_->write(columnData.c_str(), columnData.length());
    }
    bufferedWriterSink_->write(newLine);
  }
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

const std::string TextWriter::nullData = "\\N";

} // namespace facebook::velox::text
