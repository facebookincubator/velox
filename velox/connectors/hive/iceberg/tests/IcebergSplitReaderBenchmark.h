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
#pragma once

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplitReader.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/tests/utils/DataSetBuilder.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

namespace facebook::velox::iceberg::reader::test {

using TempDirectoryPath = common::testutil::TempDirectoryPath;

constexpr uint32_t kNumRowsPerBatch = 20000;
constexpr uint32_t kNumBatches = 50;
constexpr uint32_t kNumRowsPerRowGroup = 10000;
constexpr double kFilterErrorMargin = 0.2;

/// How deleted row positions are stored for a data file.
enum class DeleteFileKind {
  /// Iceberg V2 positional delete files of (file_path, pos) rows.
  kPositional,
  /// One Iceberg V3 deletion vector in a Puffin file.
  kDeletionVector,
};

class IcebergSplitReaderBenchmark {
 public:
  explicit IcebergSplitReaderBenchmark() {
    rootPool_ =
        memory::memoryManager()->addRootPool("IcebergSplitReaderBenchmark");
    leafPool_ = rootPool_->addLeafChild("IcebergSplitReaderBenchmark");
    dataSetBuilder_ =
        std::make_unique<facebook::velox::test::DataSetBuilder>(*leafPool_, 0);
    filesystems::registerLocalFileSystem();
    dwio::common::registerFileSinks();
    dwrf::registerDwrfReaderFactory();
  }

  ~IcebergSplitReaderBenchmark() {}

  void writeToFile(const std::vector<RowVectorPtr>& batches);

  void writeToPositionDeleteFile(
      const std::string& filePath,
      const std::vector<RowVectorPtr>& vectors);

  dwio::common::FilterSpec createFilterSpec(
      const std::string& columnName,
      float startPct,
      float selectPct,
      const TypePtr& type,
      bool isForRowGroupSkip,
      bool allowNulls);

  std::shared_ptr<dwio::common::ScanSpec> createScanSpec(
      const std::vector<RowVectorPtr>& batches,
      RowTypePtr& rowType,
      const std::vector<dwio::common::FilterSpec>& filterSpecs,
      std::vector<uint64_t>& hitRows,
      common::SubfieldFilters& filters);

  int read(
      const RowTypePtr& rowType,
      uint32_t nextSize,
      std::unique_ptr<connector::hive::iceberg::IcebergSplitReader>
          icebergSplitReader);

  /// Reads one column of a data file whose deleted rows are stored as
  /// 'deleteFileKind'. 'numDeleteFiles' splits the same deleted positions
  /// across that many positional delete files; a deletion vector is always one
  /// file.
  void readSingleColumn(
      const std::string& columnName,
      const TypePtr& type,
      float startPct,
      float selectPct,
      float deleteRate,
      uint32_t nextSize,
      DeleteFileKind deleteFileKind,
      int32_t numDeleteFiles);

  std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>>
  createIcebergSplitsWithDeletes(
      int32_t deleteRowsPercentage,
      DeleteFileKind deleteFileKind,
      int32_t numDeleteFiles);

  std::vector<std::string> listFiles(const std::string& dirPath);

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeIcebergSplit(
      const std::string& dataFilePath,
      const std::vector<connector::hive::iceberg::IcebergDeleteFile>&
          deleteFiles = {});

  std::vector<int64_t> makeRandomDeleteRows(int32_t deleteRowsCount);

  std::vector<int64_t> makeSequenceRows(int32_t maxRowNumber);

  std::string writePositionDeleteFile(
      const std::string& dataFilePath,
      const std::vector<int64_t>& deleteRows,
      int32_t fileIndex);

  connector::hive::iceberg::IcebergDeleteFile writeDeletionVectorFile(
      const std::string& dataFilePath,
      const std::vector<int64_t>& deleteRows);

 private:
  const std::string fileName_ = "test.data";
  const std::shared_ptr<TempDirectoryPath> fileFolder_ =
      TempDirectoryPath::create();
  const std::shared_ptr<TempDirectoryPath> deleteFileFolder_ =
      TempDirectoryPath::create();

  std::unique_ptr<facebook::velox::test::DataSetBuilder> dataSetBuilder_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::unique_ptr<dwrf::Writer> writer_;
  dwio::common::RuntimeStats runtimeStats_;

  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::DWRF};
  const std::string kHiveConnectorId = "hive-iceberg";
};

void run(
    uint32_t,
    const std::string& columnName,
    const TypePtr& type,
    float filterRateX100,
    float deleteRateX100,
    uint32_t nextSize);

/// Reads every row of one BIGINT column with 'deleteRateX100' percent of rows
/// deleted, stored as 'deleteFileKind' in 'numDeleteFiles' files.
void runDeletes(
    uint32_t,
    const std::string& columnName,
    const TypePtr& type,
    float deleteRateX100,
    DeleteFileKind deleteFileKind,
    int32_t numDeleteFiles,
    uint32_t nextSize);

} // namespace facebook::velox::iceberg::reader::test
