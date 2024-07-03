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

#include "velox/dwio/parquet/reader/ParquetReaderCli.h"
#include "velox/dwio/common/tests/utils/DataSetBuilder.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;
using namespace facebook::velox::test;

const uint32_t kBatchCount = 3;
const uint32_t kNumRows = 1000000;
const uint32_t kNumRowsPerGroup = 10000;
const uint32_t kDefaultReadingSize = 1000;

class ParquetReaderCliTest : public ParquetTestBase {
protected:
 void SetUp() override {}

 void createFiles(uint32_t numFiles, std::shared_ptr<const RowType> rowType) {
   rootPool_ = memory::memoryManager()->addRootPool("ParquetReaderCliTest");
   leafPool_ = rootPool_->addLeafChild("ParquetReaderCliTest");
   fileDirectory_ = exec::test::TempDirectoryPath::create();
   auto dataSetBuilder_ = std::make_unique<DataSetBuilder>(*leafPool_, 0);

   for (uint32_t i = 0; i < numFiles; i++) {
     auto fileName = std::to_string(i) + ".parquet";
     auto path = fileDirectory_->getPath() + "/" + fileName;
     auto localWriteFile = std::make_unique<LocalWriteFile>(path, true, false);
     auto sink =
         std::make_unique<WriteFileSink>(std::move(localWriteFile), path);
     facebook::velox::parquet::WriterOptions options;
     options.memoryPool = rootPool_.get();

     auto writer_ = std::make_unique<facebook::velox::parquet::Writer>(
         std::move(sink), options, rowType);

     auto batches =
         dataSetBuilder_->makeDataset(rowType, kBatchCount, kNumRows)
             .withRowGroupSpecificData(kNumRowsPerGroup)
             .build();

     for (auto& batch : *batches) {
       writer_->write(batch);
     }
     writer_->flush();
     writer_->close();
     files_.push_back(std::make_shared<LocalReadFile>(path));
   }
 }

 std::shared_ptr<memory::MemoryPool> rootPool_;
 std::shared_ptr<memory::MemoryPool> leafPool_;
 std::shared_ptr<facebook::velox::exec::test::TempDirectoryPath>
     fileDirectory_;
 std::vector<std::shared_ptr<ReadFile>> files_;
};

TEST_F(ParquetReaderCliTest, readLocalFiles) {
 auto rowType =
     ROW({"bigint", "double", "string"}, {BIGINT(), DOUBLE(), VARCHAR()});
 createFiles(3, rowType);

 std::unique_ptr<FilterGenerator> filterGenerator =
     std::make_unique<FilterGenerator>(rowType, 0);
 auto filterBuilder = [&]() {
   SubfieldFilters filters;
   return filters;
 };
 auto scanSpec = filterGenerator->makeScanSpec(std::move(filterBuilder()));

 for (auto file : files_) {
   auto reader = ParquetReaderCli(
       kDefaultReadingSize, rootPool_, leafPool_, file, rowType, scanSpec);
   auto hasData = true;
   uint32_t totalSize = 0;

   while (hasData) {
     auto result = BaseVector::create(rowType, 0, leafPool_.get());
     auto numRowsRead = reader.read(result);

     hasData = numRowsRead != 0;
     if (hasData) {
       auto rowVector = result->asUnchecked<RowVector>();

       totalSize += rowVector->size();
       EXPECT_EQ(numRowsRead, rowVector->size());
     }
   }

   EXPECT_EQ(totalSize, kBatchCount * kNumRows);
 }
}

TEST_F(ParquetReaderCliTest, readLocalFilesWithFilter) {
 auto rowType =
     ROW({"bigint", "double", "string"}, {BIGINT(), DOUBLE(), VARCHAR()});
 createFiles(3, rowType);

 std::unique_ptr<FilterGenerator> filterGenerator =
     std::make_unique<FilterGenerator>(rowType, 0);
 auto filterBuilder = [&]() {
   SubfieldFilters filters;
   filters[Subfield("bigint")] =
       std::make_unique<facebook::velox::common::BigintRange>(
           std::numeric_limits<int64_t>::min(),
           std::numeric_limits<int64_t>::max(),
           false);
   return filters;
 };
 auto scanSpec = filterGenerator->makeScanSpec(std::move(filterBuilder()));

 for (auto file : files_) {
   auto reader = ParquetReaderCli(
       kDefaultReadingSize, rootPool_, leafPool_, file, rowType, scanSpec);

   auto hasData = true;
   uint32_t totalSize = 0;

   while (hasData) {
     auto result = BaseVector::create(rowType, 0, leafPool_.get());
     auto numRowsRead = reader.read(result);

     hasData = numRowsRead != 0;

     if (hasData) {
       auto rowVector = result->asUnchecked<RowVector>();
       totalSize += rowVector->size();
     }
   }

   EXPECT_NE(totalSize, kBatchCount * kNumRows);
 }
}
