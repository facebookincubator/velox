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

#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"

namespace facebook::velox::parquet {

// This ParquetReaderCli can be used to read Parquet files directly.
class ParquetReaderCli {
public:
 explicit ParquetReaderCli(
     uint32_t readBatchSize,
     std::shared_ptr<memory::MemoryPool> rootPool,
     std::shared_ptr<memory::MemoryPool> leafPool,
     std::shared_ptr<ReadFile> file,
     std::shared_ptr<const RowType> rowType,
     std::shared_ptr<common::ScanSpec> scanSpec)
     : readBatchSize_(readBatchSize),
       file_(file),
       rootPool_(rootPool),
       leafPool_(leafPool),
       rowType_(rowType),
       scanSpec_(scanSpec) {
   prepareParquetReaderCli();
 }

 /**
  * Read at most readBatchSize_ rows from the file.
  * @param result Vector to collect the results.
  * @return number of rows that have been read in this round. If filters exist,
  * the number of rows in the result vector might be smaller because some rows
  * might be discarted after applying the filters. It returns none-zero values
  * if there are remaining data in the file. Otherwise, it returns 0 when the
  * file is exhausted.
  */
 uint64_t read(std::shared_ptr<BaseVector>& result);

 uint64_t read(std::shared_ptr<BaseVector>& result, uint64_t batchSize);

private:
 void prepareParquetReaderCli();

 uint32_t readBatchSize_;
 std::shared_ptr<ReadFile> file_;
 std::shared_ptr<memory::MemoryPool> rootPool_;
 std::shared_ptr<memory::MemoryPool> leafPool_;
 std::shared_ptr<const RowType> rowType_;
 std::shared_ptr<common::ScanSpec> scanSpec_;
 std::unique_ptr<dwio::common::RowReader> rowReader_;
};

} // namespace facebook::velox::parquet