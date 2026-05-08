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

#include "velox/experimental/cudf/benchmarks/CudfBenchmarkHelpers.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/exec/Values.h"

#include <cudf/io/parquet.hpp>

#include <filesystem>

namespace facebook::velox::cudf_velox {

namespace fs = std::filesystem;

// TODO (dm): Duplicate of TpchQueryBuilder::readFileSchema
TableInfo readTableInfo(
    const std::string& tableName,
    const std::string& dataPath,
    const std::vector<std::string>& standardColumns,
    dwio::common::FileFormat format,
    memory::MemoryPool* pool) {
  TableInfo info;

  const fs::path tablePath{dataPath + "/" + tableName};
  for (auto const& entry : fs::directory_iterator{tablePath}) {
    if (!entry.is_regular_file() || entry.path().filename().c_str()[0] == '.') {
      continue;
    }

    if (info.dataFiles.empty()) {
      dwio::common::ReaderOptions readerOptions{pool, nullptr, nullptr};
      readerOptions.setFileFormat(format);
      auto readFile = filesystems::getFileSystem(entry.path().string(), nullptr)
                          ->openFileForRead(entry.path().string());
      std::shared_ptr<ReadFile> sharedFile;
      sharedFile.reset(readFile.release());
      auto input =
          std::make_unique<dwio::common::BufferedInput>(sharedFile, *pool);
      auto reader = dwio::common::getReaderFactory(format)->createReader(
          std::move(input), readerOptions);
      auto fileType = reader->rowType();
      auto fileNames = fileType->names();

      VELOX_CHECK_GE(fileNames.size(), standardColumns.size());
      for (size_t i = 0; i < standardColumns.size(); ++i) {
        info.fileColumnNames[standardColumns[i]] = fileNames[i];
      }

      auto types = fileType->children();
      types.resize(standardColumns.size());
      auto colNames = standardColumns;
      info.type =
          std::make_shared<RowType>(std::move(colNames), std::move(types));
    }

    info.dataFiles.push_back(entry.path().string());
  }

  std::sort(info.dataFiles.begin(), info.dataFiles.end());
  return info;
}

std::vector<RowVectorPtr> readParquetIntoCudfVectors(
    const std::vector<std::string>& files,
    const RowTypePtr& outputType,
    const std::unordered_map<std::string, std::string>& fileColumnNames,
    memory::MemoryPool* pool,
    int32_t batchSizeBytes) {
  std::vector<std::string> fileColNames;
  for (size_t i = 0; i < outputType->size(); ++i) {
    const auto& stdName = outputType->nameOf(i);
    auto it = fileColumnNames.find(stdName);
    fileColNames.push_back(it != fileColumnNames.end() ? it->second : stdName);
  }

  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto mr = cudf_velox::get_output_mr();

  std::vector<RowVectorPtr> allBatches;

  for (const auto& filePath : files) {
    auto readerOptions = cudf::io::parquet_reader_options::builder(
                             cudf::io::source_info{filePath})
                             .build();
    readerOptions.set_column_names(fileColNames);

    auto reader = cudf::io::chunked_parquet_reader(
        batchSizeBytes, 0, readerOptions, stream, mr);

    while (reader.has_next()) {
      auto tableWithMetadata = reader.read_chunk();
      auto& tbl = tableWithMetadata.tbl;
      if (tbl && tbl->num_rows() > 0) {
        auto numRows = tbl->num_rows();
        allBatches.push_back(
            std::make_shared<cudf_velox::CudfVector>(
                pool, outputType, numRows, std::move(tbl), stream));
      }
    }
    stream.synchronize();
  }
  return allBatches;
}

namespace {

class GpuValuesAdapter : public cudf_velox::OperatorAdapter {
 public:
  GpuValuesAdapter() : OperatorAdapter("Values") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Values*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto valuesNode =
        std::dynamic_pointer_cast<const core::ValuesNode>(planNode);
    if (!valuesNode || valuesNode->values().empty()) {
      return false;
    }
    return std::dynamic_pointer_cast<cudf_velox::CudfVector>(
               valuesNode->values()[0]) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  bool keepOperator() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }
};

} // namespace

void registerGpuValuesAdapter() {
  cudf_velox::OperatorAdapterRegistry::getInstance().registerAdapter(
      std::make_unique<GpuValuesAdapter>(), /*overwrite=*/true);
}

} // namespace facebook::velox::cudf_velox
