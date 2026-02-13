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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/Writer.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/extension_type.h"
#include "arrow/ipc/writer.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "arrow/util/base64.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/parallel.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/ArrowSchema.h"
#include "velox/dwio/parquet/writer/arrow/ColumnWriter.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/PathInternal.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"

using arrow::Array;
using arrow::BinaryArray;
using arrow::BooleanArray;
using arrow::ChunkedArray;
using arrow::DataType;
using arrow::DictionaryArray;
using arrow::ExtensionArray;
using arrow::ExtensionType;
using arrow::Field;
using arrow::FixedSizeBinaryArray;
using arrow::ListArray;
using arrow::MemoryPool;
using arrow::NumericArray;
using arrow::PrimitiveArray;
using arrow::RecordBatch;
using arrow::ResizableBuffer;
using arrow::Result;
using arrow::Status;
using arrow::Table;
using arrow::TimeUnit;

using arrow::internal::checked_cast;

namespace facebook::velox::parquet::arrow::arrow {

using schema::GroupNode;

namespace {

int calculateLeafCount(const DataType* type) {
  if (type->id() == ::arrow::Type::EXTENSION) {
    type = checked_cast<const ExtensionType&>(*type).storage_type().get();
  }
  // Note numFields() can be 0 for an empty struct type.
  if (!::arrow::is_nested(type->id())) {
    // Primitive type.
    return 1;
  }

  int numLeaves = 0;
  for (const auto& field : type->fields()) {
    numLeaves += calculateLeafCount(field->type().get());
  }
  return numLeaves;
}

// Determines if the |schema_field|'s root ancestor is nullable.
bool hasNullableRoot(
    const SchemaManifest& schemaManifest,
    const SchemaField* schemaField) {
  VELOX_DCHECK_NOT_NULL(schemaField);
  const SchemaField* currentField = schemaField;
  bool nullable = schemaField->field->nullable();
  while (currentField != nullptr) {
    nullable = currentField->field->nullable();
    currentField = schemaManifest.getParent(currentField);
  }
  return nullable;
}

// Manages writing nested parquet columns with support for all nested types.
// Supported by Parquet.
class ArrowColumnWriterV2 {
 public:
  // Constructs a new object (use make() method below to construct from
  // a ChunkedArray).
  // LevelBuilders should contain one MultipathLevelBuilder per chunk of the
  // arrow-column to write.
  ArrowColumnWriterV2(
      std::vector<std::unique_ptr<MultipathLevelBuilder>> levelBuilders,
      int startLeafColumnIndex,
      int leafCount,
      RowGroupWriter* rowGroupWriter)
      : levelBuilders_(std::move(levelBuilders)),
        startLeafColumnIndex_(startLeafColumnIndex),
        leafCount_(leafCount),
        rowGroupWriter_(rowGroupWriter) {}

  // Writes out all leaf Parquet columns to the rowGroupWriter that this
  // object was constructed with.  Each leaf column is written fully before
  // the next column is written (i.e. no buffering is assumed).
  //
  // Columns are written in DFS order.
  Status write(ArrowWriteContext* ctx) {
    for (int leafIdx = 0; leafIdx < leafCount_; leafIdx++) {
      ColumnWriter* columnWriter;
      if (rowGroupWriter_->buffered()) {
        const int columnIndex = startLeafColumnIndex_ + leafIdx;
        PARQUET_CATCH_NOT_OK(
            columnWriter = rowGroupWriter_->column(columnIndex));
      } else {
        PARQUET_CATCH_NOT_OK(columnWriter = rowGroupWriter_->nextColumn());
      }
      for (auto& levelBuilder : levelBuilders_) {
        RETURN_NOT_OK(levelBuilder->write(
            leafIdx, ctx, [&](const MultipathLevelBuilderResult& result) {
              size_t visitedComponentSize =
                  result.postListVisitedElements.size();
              VELOX_DCHECK_GT(visitedComponentSize, 0);
              if (visitedComponentSize != 1) {
                return Status::NotImplemented(
                    "Lists with non-zero length null components are not supported");
              }
              const ElementRange& range = result.postListVisitedElements[0];
              std::shared_ptr<Array> valuesArray =
                  result.leafArray->Slice(range.start, range.size());

              return columnWriter->writeArrow(
                  result.defLevels,
                  result.repLevels,
                  result.defRepLevelCount,
                  *valuesArray,
                  ctx,
                  result.leafIsNullable);
            }));
      }

      if (!rowGroupWriter_->buffered()) {
        PARQUET_CATCH_NOT_OK(columnWriter->close());
      }
    }
    return Status::OK();
  }

  // MultipathLevelBuilder.
  //
  // It is necessary to create a new builder per array because the
  // MultipathlevelBuilder extracts the data necessary for writing each leaf
  // column at construction time (it optimizes based on null count) and with
  // slicing via |offset| ephemeral chunks are created which need to be tracked
  // across each leaf column-write. This decision could potentially be
  // revisited if we wanted to use "buffered" RowGroupWriters (we could
  // construct each builder on demand in that case).
  static ::arrow::Result<std::unique_ptr<ArrowColumnWriterV2>> make(
      const ChunkedArray& data,
      int64_t offset,
      const int64_t size,
      const SchemaManifest& schemaManifest,
      RowGroupWriter* rowGroupWriter,
      int startLeafColumnIndex = -1) {
    int64_t absolutePosition = 0;
    int chunkIndex = 0;
    int64_t chunkOffset = 0;
    if (data.length() == 0) {
      return std::make_unique<ArrowColumnWriterV2>(
          std::vector<std::unique_ptr<MultipathLevelBuilder>>{},
          startLeafColumnIndex,
          calculateLeafCount(data.type().get()),
          rowGroupWriter);
    }
    while (chunkIndex < data.num_chunks() && absolutePosition < offset) {
      const int64_t chunkLength = data.chunk(chunkIndex)->length();
      if (absolutePosition + chunkLength > offset) {
        // Relative offset into the chunk to reach the desired start offset for
        // writing.
        chunkOffset = offset - absolutePosition;
        break;
      } else {
        ++chunkIndex;
        absolutePosition += chunkLength;
      }
    }

    if (absolutePosition >= data.length()) {
      return Status::Invalid(
          "Cannot write data at offset past end of chunked array");
    }

    int64_t valuesWritten = 0;
    std::vector<std::unique_ptr<MultipathLevelBuilder>> builders;
    const int leafCount = calculateLeafCount(data.type().get());
    bool isNullable = false;

    int columnIndex = 0;
    if (rowGroupWriter->buffered()) {
      columnIndex = startLeafColumnIndex;
    } else {
      // The rowGroupWriter hasn't been advanced yet so add 1 to the current
      // which is the one this instance will start writing for.
      columnIndex = rowGroupWriter->currentColumn() + 1;
    }

    for (int leafOffset = 0; leafOffset < leafCount; ++leafOffset) {
      const SchemaField* schemaField = nullptr;
      RETURN_NOT_OK(schemaManifest.getColumnField(
          columnIndex + leafOffset, &schemaField));
      bool nullableRoot = hasNullableRoot(schemaManifest, schemaField);
      if (leafOffset == 0) {
        isNullable = nullableRoot;
      }

// Don't validate common ancestry for all leafs if not in debug.
#ifndef NDEBUG
      break;
#else
      if (isNullable != nullableRoot) {
        return Status::UnknownError(
            "Unexpected mismatched nullability between column index",
            columnIndex + leafOffset,
            " and ",
            columnIndex);
      }
#endif
    }
    while (valuesWritten < size) {
      const Array& chunk = *data.chunk(chunkIndex);
      const int64_t availableValues = chunk.length() - chunkOffset;
      const int64_t chunkWriteSize =
          std::min(size - valuesWritten, availableValues);

      // The chunk offset here will be 0 except for possibly the first chunk
      // because of the advancing logic above.
      std::shared_ptr<Array> arrayToWrite =
          chunk.Slice(chunkOffset, chunkWriteSize);

      if (arrayToWrite->length() > 0) {
        ARROW_ASSIGN_OR_RAISE(
            std::unique_ptr<MultipathLevelBuilder> builder,
            MultipathLevelBuilder::make(*arrayToWrite, isNullable));
        if (leafCount != builder->getLeafCount()) {
          return Status::UnknownError(
              "data type leaf_count != builder_leaf_count",
              leafCount,
              " ",
              builder->getLeafCount());
        }
        builders.emplace_back(std::move(builder));
      }

      if (chunkWriteSize == availableValues) {
        chunkOffset = 0;
        ++chunkIndex;
      }
      valuesWritten += chunkWriteSize;
    }
    return std::make_unique<ArrowColumnWriterV2>(
        std::move(builders), columnIndex, leafCount, rowGroupWriter);
  }

  int leafCount() const {
    return leafCount_;
  }

 private:
  // One builder per column-chunk.
  std::vector<std::unique_ptr<MultipathLevelBuilder>> levelBuilders_;
  int startLeafColumnIndex_;
  int leafCount_;
  RowGroupWriter* rowGroupWriter_;
};

} // namespace

// ----------------------------------------------------------------------
// FileWriter implementation.

class FileWriterImpl : public FileWriter {
 public:
  FileWriterImpl(
      std::shared_ptr<::arrow::Schema> schema,
      MemoryPool* pool,
      std::unique_ptr<ParquetFileWriter> writer,
      std::shared_ptr<ArrowWriterProperties> arrowProperties)
      : schema_(std::move(schema)),
        writer_(std::move(writer)),
        rowGroupWriter_(nullptr),
        columnWriteContext_(pool, arrowProperties.get()),
        arrowProperties_(std::move(arrowProperties)),
        closed_(false) {
    if (arrowProperties_->useThreads()) {
      parallelColumnWriteContexts_.reserve(schema_->num_fields());
      for (int i = 0; i < schema_->num_fields(); ++i) {
        // Explicitly create each ArrowWriteContext object to avoid
        // unintentional call of the copy constructor. Otherwise, the buffers
        // in the type of shared_ptr will be shared among all contexts.
        parallelColumnWriteContexts_.emplace_back(pool, arrowProperties_.get());
      }
    }
  }

  Status init() {
    return SchemaManifest::make(
        writer_->schema(),
        nullptr,
        defaultArrowReaderProperties(),
        &schemaManifest_);
  }

  Status newRowGroup(int64_t chunkSize) override {
    if (rowGroupWriter_ != nullptr) {
      PARQUET_CATCH_NOT_OK(rowGroupWriter_->close());
    }
    PARQUET_CATCH_NOT_OK(rowGroupWriter_ = writer_->appendRowGroup());
    return Status::OK();
  }

  Status close() override {
    if (!closed_) {
      // Make idempotent.
      closed_ = true;
      if (rowGroupWriter_ != nullptr) {
        PARQUET_CATCH_NOT_OK(rowGroupWriter_->close());
      }
      PARQUET_CATCH_NOT_OK(writer_->close());
    }
    return Status::OK();
  }

  Status writeColumnChunk(const Array& data) override {
    // A bit awkward here since cannot instantiate ChunkedArray from const
    // Array&.
    auto chunk = ::arrow::MakeArray(data.data());
    auto chunkedArray = std::make_shared<::arrow::ChunkedArray>(chunk);
    return writeColumnChunk(chunkedArray, 0, data.length());
  }

  Status writeColumnChunk(
      const std::shared_ptr<ChunkedArray>& data,
      int64_t offset,
      int64_t size) override {
    if (arrowProperties_->engineVersion() == ArrowWriterProperties::V2 ||
        arrowProperties_->engineVersion() == ArrowWriterProperties::V1) {
      if (rowGroupWriter_->buffered()) {
        return Status::Invalid(
            "Cannot write column chunk into the buffered row group.");
      }
      ARROW_ASSIGN_OR_RAISE(
          std::unique_ptr<ArrowColumnWriterV2> writer,
          ArrowColumnWriterV2::make(
              *data, offset, size, schemaManifest_, rowGroupWriter_));
      return writer->write(&columnWriteContext_);
    }

    return Status::NotImplemented("Unknown engine version.");
  }

  Status writeColumnChunk(
      const std::shared_ptr<::arrow::ChunkedArray>& data) override {
    return writeColumnChunk(data, 0, data->length());
  }

  std::shared_ptr<::arrow::Schema> schema() const override {
    return schema_;
  }

  Status writeTable(const Table& table, int64_t chunkSize) override {
    RETURN_NOT_OK(table.Validate());

    if (!table.schema()->Equals(*schema_, false)) {
      return Status::Invalid(
          "table schema does not match this writer's. table:'",
          table.schema()->ToString(),
          "' this:'",
          schema_->ToString(),
          "'");
    } else if (chunkSize > this->properties().maxRowGroupLength()) {
      chunkSize = this->properties().maxRowGroupLength();
    }
    // maxRowGroupBytes is applied only after the row group has accumulated
    // data.
    if (rowGroupWriter_ != nullptr && rowGroupWriter_->numRows() > 0) {
      double avgRowSize = rowGroupWriter_->totalBufferedBytes() * 1.0 /
          rowGroupWriter_->numRows();
      chunkSize = std::min(
          chunkSize,
          static_cast<int64_t>(
              this->properties().maxRowGroupBytes() / avgRowSize));
    }
    if (chunkSize <= 0 && table.num_rows() > 0) {
      return Status::Invalid("rows per row_group must be greater than 0");
    }

    auto writeRowGroup = [&](int64_t offset, int64_t size) {
      RETURN_NOT_OK(newRowGroup(size));
      for (int i = 0; i < table.num_columns(); i++) {
        RETURN_NOT_OK(writeColumnChunk(table.column(i), offset, size));
      }
      return Status::OK();
    };

    if (table.num_rows() == 0) {
      // Append a row group with 0 rows.
      RETURN_NOT_OK_ELSE(writeRowGroup(0, 0), PARQUET_IGNORE_NOT_OK(close()));
      return Status::OK();
    }

    for (int chunk = 0; chunk * chunkSize < table.num_rows(); chunk++) {
      int64_t offset = chunk * chunkSize;
      RETURN_NOT_OK_ELSE(
          writeRowGroup(offset, std::min(chunkSize, table.num_rows() - offset)),
          PARQUET_IGNORE_NOT_OK(close()));
    }
    return Status::OK();
  }

  Status newBufferedRowGroup() override {
    if (rowGroupWriter_ != nullptr) {
      PARQUET_CATCH_NOT_OK(rowGroupWriter_->close());
    }
    PARQUET_CATCH_NOT_OK(rowGroupWriter_ = writer_->appendBufferedRowGroup());
    return Status::OK();
  }

  Status writeRecordBatch(const RecordBatch& batch) override {
    if (batch.num_rows() == 0) {
      return Status::OK();
    }

    if (rowGroupWriter_ == nullptr || !rowGroupWriter_->buffered()) {
      RETURN_NOT_OK(newBufferedRowGroup());
    }

    auto writeBatch = [&](int64_t offset, int64_t size) {
      std::vector<std::unique_ptr<ArrowColumnWriterV2>> writers;
      int columnIndexStart = 0;

      for (int i = 0; i < batch.num_columns(); i++) {
        ChunkedArray chunkedArray{batch.column(i)};
        ARROW_ASSIGN_OR_RAISE(
            std::unique_ptr<ArrowColumnWriterV2> writer,
            ArrowColumnWriterV2::make(
                chunkedArray,
                offset,
                size,
                schemaManifest_,
                rowGroupWriter_,
                columnIndexStart));
        columnIndexStart += writer->leafCount();
        if (arrowProperties_->useThreads()) {
          writers.emplace_back(std::move(writer));
        } else {
          RETURN_NOT_OK(writer->write(&columnWriteContext_));
        }
      }

      if (arrowProperties_->useThreads()) {
        VELOX_DCHECK_EQ(parallelColumnWriteContexts_.size(), writers.size());
        RETURN_NOT_OK(
            ::arrow::internal::ParallelFor(
                static_cast<int>(writers.size()),
                [&](int i) {
                  return writers[i]->write(&parallelColumnWriteContexts_[i]);
                },
                arrowProperties_->executor()));
      }

      return Status::OK();
    };

    // Max number of rows allowed in a row group.
    const int64_t maxRowGroupLength = this->properties().maxRowGroupLength();
    // Max number of bytes allowed in a row group.
    const int64_t maxRowGroupBytes = this->properties().maxRowGroupBytes();

    int64_t offset = 0;
    while (offset < batch.num_rows()) {
      int64_t groupRows = rowGroupWriter_->numRows();
      int64_t batchSize =
          std::min(maxRowGroupLength - groupRows, batch.num_rows() - offset);
      if (groupRows > 0) {
        int64_t bufferedBytes = rowGroupWriter_->totalBufferedBytes();
        double avgRowSize = bufferedBytes * 1.0 / groupRows;
        batchSize = std::min(
            batchSize,
            static_cast<int64_t>(
                (maxRowGroupBytes - bufferedBytes) / avgRowSize));
      }
      if (batchSize > 0) {
        RETURN_NOT_OK(writeBatch(offset, batchSize));
        offset += batchSize;
      } else if (offset < batch.num_rows()) {
        // Current row group is full, write remaining rows in a new group.
        RETURN_NOT_OK(newBufferedRowGroup());
      }
    }

    return Status::OK();
  }

  const WriterProperties& properties() const {
    return *writer_->properties();
  }

  ::arrow::MemoryPool* memoryPool() const override {
    return columnWriteContext_.memoryPool;
  }

  const std::shared_ptr<FileMetaData> metadata() const override {
    return writer_->metadata();
  }

 private:
  friend class FileWriter;

  std::shared_ptr<::arrow::Schema> schema_;

  SchemaManifest schemaManifest_;

  std::unique_ptr<ParquetFileWriter> writer_;
  RowGroupWriter* rowGroupWriter_;
  ArrowWriteContext columnWriteContext_;
  std::shared_ptr<ArrowWriterProperties> arrowProperties_;
  bool closed_;

  /// If arrowProperties_->useThreads() is true, the vector size is equal to
  /// schema_->num_fields() to make it thread-safe. Otherwise, the vector is
  /// empty and columnWriteContext_ above is shared by all columns.
  std::vector<ArrowWriteContext> parallelColumnWriteContexts_;
};

FileWriter::~FileWriter() {}

Status FileWriter::make(
    ::arrow::MemoryPool* pool,
    std::unique_ptr<ParquetFileWriter> writer,
    std::shared_ptr<::arrow::Schema> schema,
    std::shared_ptr<ArrowWriterProperties> arrowProperties,
    std::unique_ptr<FileWriter>* out) {
  std::unique_ptr<FileWriterImpl> impl(new FileWriterImpl(
      std::move(schema), pool, std::move(writer), std::move(arrowProperties)));
  RETURN_NOT_OK(impl->init());
  *out = std::move(impl);
  return Status::OK();
}

Status FileWriter::open(
    const ::arrow::Schema& schema,
    ::arrow::MemoryPool* pool,
    std::shared_ptr<::arrow::io::OutputStream> sink,
    std::shared_ptr<WriterProperties> properties,
    std::unique_ptr<FileWriter>* writer) {
  ARROW_ASSIGN_OR_RAISE(
      *writer,
      open(
          std::move(schema),
          pool,
          std::move(sink),
          std::move(properties),
          defaultArrowWriterProperties()));
  return Status::OK();
}

Status getSchemaMetadata(
    const ::arrow::Schema& schema,
    ::arrow::MemoryPool* pool,
    const ArrowWriterProperties& properties,
    std::shared_ptr<const KeyValueMetadata>* out) {
  if (!properties.storeSchema()) {
    *out = nullptr;
    return Status::OK();
  }

  static const std::string kArrowSchemaKey = "ARROW:schema";
  std::shared_ptr<KeyValueMetadata> result;
  if (schema.metadata()) {
    result = schema.metadata()->Copy();
  } else {
    result = ::arrow::key_value_metadata({}, {});
  }

  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<Buffer> serialized,
      ::arrow::ipc::SerializeSchema(schema, pool));

  // The serialized schema is not UTF-8, which is required for Thrift.
  std::string schemaAsString = serialized->ToString();
  std::string schemaBase64 = ::arrow::util::base64_encode(schemaAsString);
  result->Append(kArrowSchemaKey, schemaBase64);
  *out = result;
  return Status::OK();
}

Status FileWriter::open(
    const ::arrow::Schema& schema,
    ::arrow::MemoryPool* pool,
    std::shared_ptr<::arrow::io::OutputStream> sink,
    std::shared_ptr<WriterProperties> properties,
    std::shared_ptr<ArrowWriterProperties> arrowProperties,
    std::unique_ptr<FileWriter>* writer) {
  ARROW_ASSIGN_OR_RAISE(
      *writer,
      open(
          std::move(schema),
          pool,
          std::move(sink),
          std::move(properties),
          arrowProperties));
  return Status::OK();
}

Result<std::unique_ptr<FileWriter>> FileWriter::open(
    const ::arrow::Schema& schema,
    ::arrow::MemoryPool* pool,
    std::shared_ptr<::arrow::io::OutputStream> sink,
    std::shared_ptr<WriterProperties> properties,
    std::shared_ptr<ArrowWriterProperties> arrowProperties) {
  std::shared_ptr<SchemaDescriptor> parquetSchema;
  RETURN_NOT_OK(
      toParquetSchema(&schema, *properties, *arrowProperties, &parquetSchema));

  auto schemaNode =
      std::static_pointer_cast<GroupNode>(parquetSchema->schemaRoot());

  std::shared_ptr<const KeyValueMetadata> metadata;
  RETURN_NOT_OK(getSchemaMetadata(schema, pool, *arrowProperties, &metadata));

  std::unique_ptr<ParquetFileWriter> baseWriter;
  PARQUET_CATCH_NOT_OK(
      baseWriter = ParquetFileWriter::open(
          std::move(sink),
          schemaNode,
          std::move(properties),
          std::move(metadata)));

  std::unique_ptr<FileWriter> writer;
  auto schemaPtr = std::make_shared<::arrow::Schema>(schema);
  RETURN_NOT_OK(make(
      pool,
      std::move(baseWriter),
      std::move(schemaPtr),
      std::move(arrowProperties),
      &writer));

  return writer;
}

Status writeFileMetaData(
    const FileMetaData& fileMetadata,
    ::arrow::io::OutputStream* sink) {
  PARQUET_CATCH_NOT_OK(
      ::facebook::velox::parquet::arrow::writeFileMetaData(fileMetadata, sink));
  return Status::OK();
}

Status writeMetaDataFile(
    const FileMetaData& fileMetadata,
    ::arrow::io::OutputStream* sink) {
  PARQUET_CATCH_NOT_OK(
      ::facebook::velox::parquet::arrow::writeMetaDataFile(fileMetadata, sink));
  return Status::OK();
}

Status writeTable(
    const ::arrow::Table& table,
    ::arrow::MemoryPool* pool,
    std::shared_ptr<::arrow::io::OutputStream> sink,
    int64_t chunkSize,
    std::shared_ptr<WriterProperties> properties,
    std::shared_ptr<ArrowWriterProperties> arrowProperties) {
  std::unique_ptr<FileWriter> writer;
  ARROW_ASSIGN_OR_RAISE(
      writer,
      FileWriter::open(
          *table.schema(),
          pool,
          std::move(sink),
          std::move(properties),
          std::move(arrowProperties)));
  RETURN_NOT_OK(writer->writeTable(table, chunkSize));
  return writer->close();
}

} // namespace facebook::velox::parquet::arrow::arrow
