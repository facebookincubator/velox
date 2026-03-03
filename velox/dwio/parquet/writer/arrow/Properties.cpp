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

// Adapted from Apache Arrow.

#include <sstream>
#include <utility>

#include "velox/dwio/parquet/writer/arrow/Properties.h"

#include "arrow/io/buffered.h"
#include "arrow/io/memory.h"
#include "arrow/util/thread_pool.h"

namespace facebook::velox::parquet::arrow {

ReaderProperties defaultReaderProperties() {
  static ReaderProperties defaultReaderProperties;
  return defaultReaderProperties;
}

std::shared_ptr<ArrowInputStream> ReaderProperties::getStream(
    std::shared_ptr<ArrowInputFile> source,
    int64_t start,
    int64_t numBytes) {
  if (bufferedStreamEnabled_) {
    // ARROW-6180 / PARQUET-1636 Create isolated reader that references segment.
    // Of source.
    PARQUET_ASSIGN_OR_THROW(
        std::shared_ptr<::arrow::io::InputStream> safeStream,
        ::arrow::io::RandomAccessFile::GetStream(source, start, numBytes));
    PARQUET_ASSIGN_OR_THROW(
        auto stream,
        ::arrow::io::BufferedInputStream::Create(
            bufferSize_, pool_, safeStream, numBytes));
    return std::move(stream);
  } else {
    PARQUET_ASSIGN_OR_THROW(auto data, source->ReadAt(start, numBytes));

    if (data->size() != numBytes) {
      std::stringstream ss;
      ss << "Tried reading " << numBytes << " bytes starting at position "
         << start << " from file but only got " << data->size();
      throw ParquetException(ss.str());
    }
    return std::make_shared<::arrow::io::BufferReader>(data);
  }
}

::arrow::internal::Executor* ArrowWriterProperties::executor() const {
  return executor_ != nullptr ? executor_
                              : ::arrow::internal::GetCpuThreadPool();
}

ArrowReaderProperties defaultArrowReaderProperties() {
  static ArrowReaderProperties defaultReaderProps;
  return defaultReaderProps;
}

std::shared_ptr<ArrowWriterProperties> defaultArrowWriterProperties() {
  static std::shared_ptr<ArrowWriterProperties> defaultWriterProperties =
      ArrowWriterProperties::Builder().build();
  return defaultWriterProperties;
}

} // namespace facebook::velox::parquet::arrow
