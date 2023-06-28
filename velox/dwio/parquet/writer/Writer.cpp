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

#include "velox/dwio/parquet/writer/Writer.h"
#include <arrow/c/bridge.h> // @manual
#include <arrow/table.h> // @manual
#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::parquet {

void Writer::flush() {
  if (stagingRows_ > 0) {
    if (!arrowWriter_) {
      stream_ = std::make_shared<DataBufferSink>(
          finalSink_.get(),
          pool_,
          queryCtx_->queryConfig().dataBufferGrowRatio());
      auto arrowProperties =
          ::parquet::ArrowWriterProperties::Builder().build();
      PARQUET_ASSIGN_OR_THROW(
          arrowWriter_,
          ::parquet::arrow::FileWriter::Open(
              *(schema_.get()),
              arrow::default_memory_pool(),
              stream_,
              properties_,
              arrowProperties));
    }

    auto fields = schema_->fields();
    std::vector<std::shared_ptr<arrow::ChunkedArray>> chunks;
    for (int colIdx = 0; colIdx < fields.size(); colIdx++) {
      auto dataType = fields.at(colIdx)->type();
      auto chunk = arrow::ChunkedArray::Make(
                       std::move(stagingChunks_.at(colIdx)), dataType)
                       .ValueOrDie();
      chunks.push_back(chunk);
    }
    auto table = arrow::Table::Make(schema_, std::move(chunks), stagingRows_);
    PARQUET_THROW_NOT_OK(arrowWriter_->WriteTable(*table, maxRowGroupRows_));
    if (queryCtx_->queryConfig().dataBufferGrowRatio() > 1) {
      PARQUET_THROW_NOT_OK(stream_->Flush());
    }
    for (auto& chunk : stagingChunks_) {
      chunk.clear();
    }
    stagingRows_ = 0;
    stagingBytes_ = 0;
  }
}

/**
 * This method would cache input `ColumnarBatch` to make the size of row group
 * big. It would flush when:
 * - the cached numRows bigger than `maxRowGroupRows_`
 * - the cached bytes bigger than `maxRowGroupBytes_`
 *
 * This method assumes each input `ColumnarBatch` have same schema.
 */
void Writer::write(const RowVectorPtr& data) {
  ArrowArray array;
  ArrowSchema schema;
  exportToArrow(data, array, &pool_);
  exportToArrow(data, schema);
  PARQUET_ASSIGN_OR_THROW(
      auto recordBatch, arrow::ImportRecordBatch(&array, &schema));
  if (!schema_) {
    schema_ = recordBatch->schema();
    for (int colIdx = 0; colIdx < schema_->num_fields(); colIdx++) {
      stagingChunks_.push_back(std::vector<std::shared_ptr<arrow::Array>>());
    }
  }

  auto bytes = data->estimateFlatSize();
  auto numRows = data->size();
  if (stagingBytes_ + bytes > maxRowGroupBytes_ ||
      stagingRows_ + numRows > maxRowGroupRows_) {
    flush();
  }

  for (int colIdx = 0; colIdx < recordBatch->num_columns(); colIdx++) {
    auto array = recordBatch->column(colIdx);
    stagingChunks_.at(colIdx).push_back(array);
  }
  stagingRows_ += numRows;
  stagingBytes_ += bytes;
}

void Writer::newRowGroup(int32_t numRows) {
  PARQUET_THROW_NOT_OK(arrowWriter_->NewRowGroup(numRows));
}

void Writer::close() {
  flush();

  if (arrowWriter_) {
    PARQUET_THROW_NOT_OK(arrowWriter_->Close());
    arrowWriter_.reset();
  }

  PARQUET_THROW_NOT_OK(stream_->Close());

  stagingChunks_.clear();
}

} // namespace facebook::velox::parquet
