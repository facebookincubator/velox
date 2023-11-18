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
#include <folly/init/Init.h>
#include <cstdint>
#include <iostream>
#include "velox/dwio/parquet/reader/ParquetReader.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

std::shared_ptr<facebook::velox::common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType) {
  auto scanSpec = std::make_shared<facebook::velox::common::ScanSpec>("");
  scanSpec->addAllChildFields(*rowType);
  return scanSpec;
}

// A temporary program that reads from parquet file and prints its meta data and
// content. Usage: velox_scan_parquet {parquet_file_path}
int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <parquet file>" << std::endl;
    return 1;
  }
  std::string filePath{argv[1]};

  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();

  facebook::velox::dwio::common::ReaderOptions readerOpts{pool.get()};
  auto reader = new ParquetReader(
      std::make_unique<BufferedInput>(
          std::make_shared<LocalReadFile>(filePath),
          readerOpts.getMemoryPool()),
      readerOpts);

  std::cout << "number of rows: " << reader->numberOfRows().value()
            << std::endl;

  auto outputType = reader->rowType();
  std::cout << "velox type: " << outputType->toString() << std::endl;

  dwio::common::RowReaderOptions rowReaderOpts;
  auto scanSpec = makeScanSpec(outputType);
  rowReaderOpts.setScanSpec(scanSpec);

  auto rowReader = reader->createRowReader(rowReaderOpts);

  VectorPtr result = BaseVector::create(outputType, 0, pool.get());
  while (rowReader->next(500, result)) {
    for (vector_size_t i = 0; i < result->size(); i++) {
      std::cout << result->toString(i) << std::endl;
    }
  }
  return 0;
}
