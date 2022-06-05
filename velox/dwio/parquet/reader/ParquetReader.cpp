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

#include "ParquetReader.h"
#include <thrift/protocol/TCompactProtocol.h>
#include "ReaderUtil.h"
#include "velox/dwio/common/MetricsLog.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/parquet/thrift/ThriftTransport.h"

namespace facebook::velox::parquet {

using namespace velox::dwio::common;

////////////////////////////////////////////////////////////////////////////////

ReaderBase::ReaderBase(
    std::unique_ptr<InputStream> stream,
    const ReaderOptions& options)
    : pool_(options.getMemoryPool()),
      options_(options),
      stream_{std::move(stream)} {
  input_ = std::make_shared<BufferedInput>(*stream_, pool_);

  fileLength_ = stream_->getLength();
  DWIO_ENSURE(fileLength_ > 0, "Parquet file is empty");
  DWIO_ENSURE(fileLength_ >= 12, "Parquet file is too small");

  loadFileMetaData();
  initializeSchema();
}

memory::MemoryPool& ReaderBase::getMemoryPool() const {
  return pool_;
}

const InputStream& ReaderBase::getStream() const {
  return *stream_;
}

const thrift::FileMetaData& ReaderBase::getFileMetaData() const {
  return *fileMetaData_;
}

BufferedInput& ReaderBase::getBufferedInput() const {
  return *input_;
}

const uint64_t ReaderBase::getFileLength() const {
  return fileLength_;
}

void ReaderBase::loadFileMetaData() {
  bool preloadFile_ = fileLength_ <= FILE_PRELOAD_THRESHOLD;
  uint64_t readSize =
      preloadFile_ ? fileLength_ : std::min(fileLength_, DIRECTORY_SIZE_GUESS);

  input_->enqueue({fileLength_ - readSize, readSize});
  input_->load(preloadFile_ ? LogType::FILE : LogType::FOOTER);

  int readBytes = 4;
  const void* buf;

  readInput(*input_, readSize - 4, &buf, 4, LogType::FOOTER);
  DWIO_ENSURE(
      strncmp(static_cast<const char*>(buf), "PAR1", 4) == 0,
      "No magic bytes found at end of the Parquet file");

  readInput(*input_, readSize - 8, &buf, 4, LogType::FOOTER);
  uint32_t footerLen = *(static_cast<const uint32_t*>(buf));
  DWIO_ENSURE(
      footerLen > 0 && footerLen + 12 < fileLength_,
      "Footer length error in the Parquet file");
  //  printf("footerLen %d\n", footerLen);

  readInput(
      *input_, readSize - (footerLen + 8), &buf, footerLen, LogType::FOOTER);

  auto thriftTransport =
      std::make_shared<thrift::ThriftBufferedTransport>(buf, footerLen);
  auto thriftProtocol =
      std::make_unique<apache::thrift::protocol::TCompactProtocolT<
          thrift::ThriftBufferedTransport>>(thriftTransport);
  fileMetaData_ = std::make_unique<thrift::FileMetaData>();
  fileMetaData_->read(thriftProtocol.get());
}

void ReaderBase::initializeSchema() {
  if (fileMetaData_->__isset.encryption_algorithm) {
    VELOX_UNSUPPORTED("Encrypted Parquet files are not supported");
  }

  DWIO_ENSURE(
      fileMetaData_->schema.size() > 1,
      "Invalid Parquet schema: Need at least one non-root column in the file");
  DWIO_ENSURE(
      fileMetaData_->schema[0].repetition_type ==
          thrift::FieldRepetitionType::REQUIRED,
      "Invalid Parquet schema: root element must be REQUIRED");
  DWIO_ENSURE(
      fileMetaData_->schema[0].num_children > 0,
      "Invalid Parquet schema: root element must have at least 1 child");

  std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>> children;
  children.reserve(fileMetaData_->schema[0].num_children);

  uint32_t maxDefine = 0;
  uint32_t maxRepeat = 0;
  uint32_t schemaIdx = 0;
  uint32_t columnIdx = 0;
  uint32_t maxSchemaElementIdx = fileMetaData_->schema.size() - 1;
  schemaWithId_ = getParquetColumnInfo(
      maxSchemaElementIdx, maxRepeat, maxDefine, schemaIdx, columnIdx);
  schema_ = createRowType(schemaWithId_->getChildren());
}

std::shared_ptr<const ParquetTypeWithId> ReaderBase::getParquetColumnInfo(
    uint32_t maxSchemaElementIdx,
    uint32_t maxRepeat,
    uint32_t maxDefine,
    uint32_t& schemaIdx,
    uint32_t& columnIdx) const {
  DWIO_ENSURE(fileMetaData_ != nullptr);
  DWIO_ENSURE(schemaIdx < fileMetaData_->schema.size());

  auto& schema = fileMetaData_->schema;
  uint32_t curSchemaIdx = schemaIdx;
  auto& schemaElement = schema[curSchemaIdx];
  //  schemaIdx++;

  if (schemaElement.__isset.repetition_type) {
    if (schemaElement.repetition_type !=
        thrift::FieldRepetitionType::REQUIRED) {
      maxDefine++;
    }
    if (schemaElement.repetition_type ==
        thrift::FieldRepetitionType::REPEATED) {
      maxRepeat++;
    }
  }

  if (!schemaElement.__isset.type) { // inner node
    DWIO_ENSURE(
        schemaElement.__isset.num_children && schemaElement.num_children > 0,
        "Node has no children but should");

    std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>> children;

    for (int32_t i = 0; i < schemaElement.num_children; i++) {
      auto child = getParquetColumnInfo(
          maxSchemaElementIdx, maxRepeat, maxDefine, ++schemaIdx, columnIdx);
      children.push_back(child);
    }
    DWIO_ENSURE(!children.empty());

    if (schemaElement.__isset.converted_type) {
      switch (schemaElement.converted_type) {
        case thrift::ConvertedType::LIST:
        case thrift::ConvertedType::MAP:
          DWIO_ENSURE(children.size() == 1);
          return std::make_shared<const ParquetTypeWithId>(
              children[0]->type,
              std::move(children[0]->getChildren()),
              curSchemaIdx, // TODO: there are holes in the ids
              maxSchemaElementIdx,
              -1, // columnIdx,
              schemaElement.name,
              maxRepeat,
              maxDefine);
        case thrift::ConvertedType::MAP_KEY_VALUE: // child of MAP
          DWIO_ENSURE(
              schemaElement.repetition_type ==
              thrift::FieldRepetitionType::REPEATED);
          DWIO_ENSURE(children.size() == 2);
          return std::make_shared<const ParquetTypeWithId>(
              TypeFactory<TypeKind::MAP>::create(
                  children[0]->type, children[1]->type),
              std::move(children),
              curSchemaIdx, // TODO: there are holes in the ids
              maxSchemaElementIdx,
              -1, // columnIdx,
              schemaElement.name,
              maxRepeat,
              maxDefine);
        default:
          VELOX_UNSUPPORTED(
              "Unsupported SchemaElement type: {}",
              schemaElement.converted_type);
      }
    } else {
      if (schemaElement.repetition_type ==
          thrift::FieldRepetitionType::REPEATED) {
        // child of LIST: "bag"
        DWIO_ENSURE(children.size() == 1);
        return std::make_shared<ParquetTypeWithId>(
            TypeFactory<TypeKind::ARRAY>::create(children[0]->type),
            std::move(children),
            curSchemaIdx,
            maxSchemaElementIdx,
            -1, // columnIdx,
            schemaElement.name,
            maxRepeat,
            maxDefine);
      } else {
        // Row type
        return std::make_shared<const ParquetTypeWithId>(
            createRowType(children),
            std::move(children),
            curSchemaIdx,
            maxSchemaElementIdx,
            -1, // columnIdx,
            schemaElement.name,
            maxRepeat,
            maxDefine);
      }
    }
  } else { // leaf node
    const auto veloxType = convertType(schemaElement);
    std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>>
        children{};
    std::shared_ptr<const ParquetTypeWithId> leafTypePtr =
        std::make_shared<const ParquetTypeWithId>(
            veloxType,
            std::move(children),
            curSchemaIdx,
            maxSchemaElementIdx,
            columnIdx++,
            schemaElement.name,
            maxRepeat,
            maxDefine);

    if (schemaElement.repetition_type ==
        thrift::FieldRepetitionType::REPEATED) {
      // Array
      children.reserve(1);
      children.push_back(leafTypePtr);
      return std::make_shared<const ParquetTypeWithId>(
          TypeFactory<TypeKind::ARRAY>::create(veloxType),
          std::move(children),
          curSchemaIdx,
          maxSchemaElementIdx,
          columnIdx++,
          schemaElement.name,
          maxRepeat,
          maxDefine);
    }

    return leafTypePtr;
  }
}

TypePtr ReaderBase::convertType(
    const thrift::SchemaElement& schemaElement) const {
  DWIO_ENSURE(schemaElement.__isset.type && schemaElement.num_children == 0);
  DWIO_ENSURE(
      schemaElement.type != thrift::Type::FIXED_LEN_BYTE_ARRAY ||
          schemaElement.__isset.type_length,
      "FIXED_LEN_BYTE_ARRAY requires length to be set");

  if (schemaElement.__isset.converted_type) {
    switch (schemaElement.converted_type) {
      case thrift::ConvertedType::INT_8:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "INT8 converted type can only be set for value of thrift::Type::INT32");
        return TINYINT();

      case thrift::ConvertedType::INT_16:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "INT16 converted type can only be set for value of thrift::Type::INT32");
        return SMALLINT();

      case thrift::ConvertedType::INT_32:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "INT32 converted type can only be set for value of thrift::Type::INT32");
        return INTEGER();

      case thrift::ConvertedType::INT_64:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "INT64 converted type can only be set for value of thrift::Type::INT32");
        return BIGINT();

      case thrift::ConvertedType::UINT_8:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "UINT_8 converted type can only be set for value of thrift::Type::INT32");
        return TINYINT();

      case thrift::ConvertedType::UINT_16:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "UINT_16 converted type can only be set for value of thrift::Type::INT32");
        return SMALLINT();

      case thrift::ConvertedType::UINT_32:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "UINT_32 converted type can only be set for value of thrift::Type::INT32");
        return INTEGER();

      case thrift::ConvertedType::UINT_64:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT64,
            "UINT_64 converted type can only be set for value of thrift::Type::INT64");
        return TINYINT();

      case thrift::ConvertedType::DATE:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "DATE converted type can only be set for value of thrift::Type::INT32");
        return DATE();

      case thrift::ConvertedType::TIMESTAMP_MICROS:
      case thrift::ConvertedType::TIMESTAMP_MILLIS:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT64,
            "TIMESTAMP_MICROS or TIMESTAMP_MILLIS converted type can only be set for value of thrift::Type::INT64");
        return TIMESTAMP();

      case thrift::ConvertedType::DECIMAL:
        DWIO_ENSURE(
            !schemaElement.__isset.precision || !schemaElement.__isset.scale,
            "DECIMAL requires a length and scale specifier!");
        VELOX_UNSUPPORTED("Decimal type is not supported yet");

      case thrift::ConvertedType::UTF8:
        switch (schemaElement.type) {
          case thrift::Type::BYTE_ARRAY:
          case thrift::Type::FIXED_LEN_BYTE_ARRAY:
            return VARCHAR();
          default:
            DWIO_RAISE(
                "UTF8 converted type can only be set for thrift::Type::(FIXED_LEN_)BYTE_ARRAY");
        }
      case thrift::ConvertedType::MAP:
      case thrift::ConvertedType::MAP_KEY_VALUE:
      case thrift::ConvertedType::LIST:
      case thrift::ConvertedType::ENUM:
      case thrift::ConvertedType::TIME_MILLIS:
      case thrift::ConvertedType::TIME_MICROS:
      case thrift::ConvertedType::JSON:
      case thrift::ConvertedType::BSON:
      case thrift::ConvertedType::INTERVAL:
      default:
        DWIO_RAISE(
            "Unsupported Parquet SchemaElement converted type: ",
            schemaElement.converted_type);
    }
  } else {
    switch (schemaElement.type) {
      case thrift::Type::type::BOOLEAN:
        return BOOLEAN();
      case thrift::Type::type::INT32:
        return INTEGER();
      case thrift::Type::type::INT64:
        return BIGINT();
      case thrift::Type::type::INT96:
        return DOUBLE(); // TODO: Lose precision
      case thrift::Type::type::FLOAT:
        return REAL();
      case thrift::Type::type::DOUBLE:
        return DOUBLE();
      case thrift::Type::type::BYTE_ARRAY:
      case thrift::Type::type::FIXED_LEN_BYTE_ARRAY:
        if (binaryAsString) {
          return VARCHAR();
        } else {
          return VARBINARY();
        }

      default:
        DWIO_RAISE("Unknown Parquet SchemaElement type: ", schemaElement.type);
    }
  }
}

const uint64_t ReaderBase::getFileNumRows() const {
  return fileMetaData_->num_rows;
}

const std::shared_ptr<const RowType>& ReaderBase::getSchema() const {
  return schema_;
}

const std::shared_ptr<const TypeWithId>& ReaderBase::getSchemaWithId() {
  return schemaWithId_;
}

std::shared_ptr<const RowType> ReaderBase::createRowType(
    std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>>
        children) {
  std::vector<std::string> childNames;
  std::vector<TypePtr> childTypes;
  for (auto& child : children) {
    childNames.push_back(
        std::static_pointer_cast<const ParquetTypeWithId>(child)->name_);
    childTypes.push_back(child->type);
  }
  return TypeFactory<TypeKind::ROW>::create(
      std::move(childNames), std::move(childTypes));
}

////////////////////////////////////////////////////////////////////////////////

ParquetReader::ParquetReader(
    std::unique_ptr<InputStream> stream,
    const ReaderOptions& options)
    : readerBase_(std::make_shared<ReaderBase>(std::move(stream), options)) {}

} // namespace facebook::velox::parquet