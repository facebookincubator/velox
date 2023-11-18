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
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>

#include <folly/init/Init.h>

#include "velox/dwio/parquet/reader/ParquetReader.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

static bool ValidatePath(const char* flagname, const std::string& value) {
  if (!value.empty() && value[0] == '/') {
    return true;
  }
  std::cerr << "The file_path must not be empty and should be an absolute path."
            << std::endl;
  return false;
}

DEFINE_uint64(
    rows_to_read,
    std::numeric_limits<uint64_t>::max(),
    "Number of rows to read");
DEFINE_uint64(batch_size, 100, "Max number of rows to read at a time");
DEFINE_string(file_path, "", "Path to parquet file");
DEFINE_validator(file_path, &ValidatePath);

std::shared_ptr<facebook::velox::common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType) {
  auto scanSpec = std::make_shared<facebook::velox::common::ScanSpec>("");
  scanSpec->addAllChildFields(*rowType);
  return scanSpec;
}

// A temporary program that reads from parquet file and prints its meta data and
// content. Usage: velox_dwio_print_parquet {parquet_file_path}
int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();

  facebook::velox::dwio::common::ReaderOptions readerOpts{pool.get()};
  auto reader = std::make_unique<ParquetReader>(
      std::make_unique<BufferedInput>(
          std::make_shared<LocalReadFile>(FLAGS_file_path),
          readerOpts.getMemoryPool()),
      readerOpts);

  std::cout << "number of rows: " << reader->numberOfRows().value()
            << std::endl;

  auto outputType = reader->rowType();
  std::cout << "Velox column types:" << outputType->toString() << std::endl;

  dwio::common::RowReaderOptions rowReaderOpts;
  auto scanSpec = makeScanSpec(outputType);
  rowReaderOpts.setScanSpec(scanSpec);

  auto rowReader = reader->createRowReader(rowReaderOpts);

  VectorPtr result = BaseVector::create(outputType, 0, pool.get());
  uint64_t to_read = FLAGS_rows_to_read;

  while (to_read > 0) {
    int readed = rowReader->next(std::min(FLAGS_batch_size, to_read), result);

    if (readed <= 0) {
      break;
    }

    for (vector_size_t i = 0; i < result->size(); i++) {
      std::cout << result->toString(i) << std::endl;
    }
    VELOX_CHECK_EQ(readed, result->size(), "readed != result->size()");
    to_read = to_read - readed;
  }
  return 0;
}
