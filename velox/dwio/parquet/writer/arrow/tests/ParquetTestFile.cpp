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

#include "velox/dwio/parquet/writer/arrow/tests/ParquetTestFile.h"

#include "velox/common/file/LocalFile.h"
#include "velox/dwio/common/BufferedInput.h"

namespace facebook::velox::parquet::arrow::test {

void ParquetTestFile::write(
    const std::shared_ptr<common::testutil::TempFilePath>& path,
    const std::shared_ptr<::arrow::Buffer>& buffer) {
  auto file = std::make_unique<LocalWriteFile>(path->getPath(), false, false);
  file->append(buffer->ToString());
  file->close();
}

ParquetTestFile ParquetTestFile::open(
    const std::shared_ptr<::arrow::Buffer>& buffer,
    const std::string& poolName) {
  ParquetTestFile file;
  file.path_ = common::testutil::TempFilePath::create();
  write(file.path_, buffer);

  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  file.rootPool_ = memory::memoryManager()->addRootPool(poolName);
  file.leafPool_ = file.rootPool_->addLeafChild(poolName);

  dwio::common::ReaderOptions readerOptions{file.leafPool_.get()};
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(file.path_->getPath()),
      readerOptions.memoryPool());
  file.reader_ =
      std::make_unique<ParquetReader>(std::move(input), readerOptions);
  return file;
}

} // namespace facebook::velox::parquet::arrow::test
