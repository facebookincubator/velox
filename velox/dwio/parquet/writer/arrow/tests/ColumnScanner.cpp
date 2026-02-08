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

#include "velox/dwio/parquet/writer/arrow/tests/ColumnScanner.h"

#include <cstdint>
#include <memory>

#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"

using arrow::MemoryPool;

namespace facebook::velox::parquet::arrow {

std::shared_ptr<Scanner> Scanner::make(
    std::shared_ptr<ColumnReader> colReader,
    int64_t batchSize,
    MemoryPool* pool) {
  switch (colReader->type()) {
    case Type::kBoolean:
      return std::make_shared<BoolScanner>(
          std::move(colReader), batchSize, pool);
    case Type::kInt32:
      return std::make_shared<Int32Scanner>(
          std::move(colReader), batchSize, pool);
    case Type::kInt64:
      return std::make_shared<Int64Scanner>(
          std::move(colReader), batchSize, pool);
    case Type::kInt96:
      return std::make_shared<Int96Scanner>(
          std::move(colReader), batchSize, pool);
    case Type::kFloat:
      return std::make_shared<FloatScanner>(
          std::move(colReader), batchSize, pool);
    case Type::kDouble:
      return std::make_shared<DoubleScanner>(
          std::move(colReader), batchSize, pool);
    case Type::kByteArray:
      return std::make_shared<ByteArrayScanner>(
          std::move(colReader), batchSize, pool);
    case Type::kFixedLenByteArray:
      return std::make_shared<FixedLenByteArrayScanner>(
          std::move(colReader), batchSize, pool);
    default:
      ParquetException::NYI("type reader not implemented");
  }
  // Unreachable code, but suppress compiler warning.
  return std::shared_ptr<Scanner>(nullptr);
}

int64_t scanAllValues(
    int32_t batchSize,
    int16_t* defLevels,
    int16_t* repLevels,
    uint8_t* values,
    int64_t* valuesBuffered,
    ColumnReader* reader) {
  switch (reader->type()) {
    case parquet::Type::kBoolean:
      return scanAll<BoolReader>(
          batchSize, defLevels, repLevels, values, valuesBuffered, reader);
    case parquet::Type::kInt32:
      return scanAll<Int32Reader>(
          batchSize, defLevels, repLevels, values, valuesBuffered, reader);
    case parquet::Type::kInt64:
      return scanAll<Int64Reader>(
          batchSize, defLevels, repLevels, values, valuesBuffered, reader);
    case parquet::Type::kInt96:
      return scanAll<Int96Reader>(
          batchSize, defLevels, repLevels, values, valuesBuffered, reader);
    case parquet::Type::kFloat:
      return scanAll<FloatReader>(
          batchSize, defLevels, repLevels, values, valuesBuffered, reader);
    case parquet::Type::kDouble:
      return scanAll<DoubleReader>(
          batchSize, defLevels, repLevels, values, valuesBuffered, reader);
    case parquet::Type::kByteArray:
      return scanAll<ByteArrayReader>(
          batchSize, defLevels, repLevels, values, valuesBuffered, reader);
    case parquet::Type::kFixedLenByteArray:
      return scanAll<FixedLenByteArrayReader>(
          batchSize, defLevels, repLevels, values, valuesBuffered, reader);
    default:
      ParquetException::NYI("type reader not implemented");
  }
  // Unreachable code, but suppress compiler warning.
  return 0;
}

} // namespace facebook::velox::parquet::arrow
