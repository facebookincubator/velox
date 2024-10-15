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
#include <fstream>
#include <iostream>
#include "velox/dwio/pagefile/reader/PageFileReader.h"
#include "velox/serializers/PrestoSerializer.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::pagefile;


void serialize(
    const RowVectorPtr& rowVector,
    std::ostream* output,
    std::unique_ptr<serializer::presto::PrestoVectorSerde>& serde,
    std::shared_ptr<facebook::velox::memory::MemoryPool> pool) {
  const bool useLosslessTimestamp = true;
  serializer::presto::PrestoVectorSerde::PrestoOptions paramOptions{
      useLosslessTimestamp, CompressionKind::CompressionKind_NONE};
  auto streamInitialSize = output->tellp();

  auto arena = std::make_unique<StreamArena>(pool.get());
  auto rowType = asRowType(rowVector->type());
  auto numRows = rowVector->size();
  auto serializer =
      serde->createIterativeSerializer(rowType, numRows, arena.get(), &paramOptions);

  serializer->append(rowVector);
  auto size = serializer->maxSerializedSize();
  facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
  OStreamOutputStream out(output, &listener);
  serializer->flush(&out);
}


// A temporary program that reads from parquet file and prints its meta data and
// content. Usage: velox_scan_parquet {parquet_file_path}
int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  if (argc != 2 && argc != 3) {
    std::cerr << "Expected 2 arguments: <parquet file> <output file>"
              << std::endl;
    return 1;
  }

  std::string filePath{argv[1]};

  facebook::velox::memory::MemoryManager::initialize({});
  auto pool = facebook::velox::memory::memoryManager()->addLeafPool();

  // facebook::velox::dwio::common::ReaderOptions readerOpts{pool.get()};
  // auto reader = new PagefileReader(
  //     std::make_unique<BufferedInput>(
  //         std::make_shared<LocalReadFile>(filePath),
  //         readerOpts.memoryPool()),
  //     readerOpts);
  //
  // std::cout << "number of rows: " << reader->numberOfRows().value()
  //           << std::endl;
  //
  // auto outputType = reader->rowType();
  // std::cout << "velox type: " << outputType->toString() << std::endl;
  //
  // dwio::common::RowReaderOptions rowReaderOpts;
  // auto scanSpec = makeScanSpec(outputType);
  // rowReaderOpts.setScanSpec(scanSpec);
  //
  // auto rowReader = reader->createRowReader(rowReaderOpts);
  // VectorPtr result = BaseVector::create(outputType, 0, pool.get());
  //
  // if (argc == 2) {
  //   while (rowReader->next(500, result)) {
  //     // print result
  //     for (int i = 0; i < result->size(); i++) {
  //       std::cout << result->toString(i) << std::endl;
  //     }
  //   }
  //   return 0;
  // }
  // std::ofstream ostrm(argv[2], std::ios::binary);
  // auto serde = std::make_unique<serializer::presto::PrestoVectorSerde>();
  //
  // // Hack: convert VectorPtr to RowVectorPtr
  // RowVectorPtr ptr(result->as<RowVector>(), [](RowVector*) {});
  //
  // while (rowReader->next(500, result)) {
  //   serialize(ptr, &ostrm, serde, pool);
  // }
  // return 0;
}
