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
#include <folly/executors/IOThreadPoolExecutor.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>

#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/vector/BaseVector.h"

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwrf;

DEFINE_string(
    file_path,
    "",
    "Path to the file to scan. Required.");
DEFINE_string(
    format,
    "",
    "File format: orc, dwrf, parquet (or pq). Required.");
DEFINE_string(
    config_path,
    "",
    "Path to the configuration file. Required.");
DEFINE_int32(
    batch_size,
    10000,
    "Number of rows to read per batch.");
DEFINE_bool(
    show_throughput,
    true,
    "Display throughput statistics after scanning.");
DEFINE_bool(
    print_rows,
    true,
    "Print row contents to stdout. Set to false to only measure throughput.");
DEFINE_string(
    buffer_type,
    "standard",
    "Type of buffered input to use: standard, direct, or cached.");
DEFINE_int32(
    io_threads,
    0,
    "Number of IO threads for async operations. 0 uses hardware_concurrency.");

namespace {

std::shared_ptr<config::ConfigBase> readConfig(const std::string& filePath) {
  std::ifstream configFile(filePath);
  if (!configFile.is_open()) {
    throw std::runtime_error(
        fmt::format("Couldn't open config file {} for reading.", filePath));
  }

  std::unordered_map<std::string, std::string> properties;
  std::string line;
  while (getline(configFile, line)) {
    line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
    if (line[0] == '#' || line.empty()) {
      continue;
    }
    auto delimiterPos = line.find('=');
    auto name = line.substr(0, delimiterPos);
    auto value = line.substr(delimiterPos + 1);
    properties.emplace(name, value);
  }

  return std::make_shared<config::ConfigBase>(std::move(properties));
}

FileFormat inferFileFormat(std::string_view format) {
  if (format == "orc") {
    return FileFormat::ORC;
  }
  if (format == "parquet" || format == "pq") {
    return FileFormat::PARQUET;
  }
  if (format == "dwrf") {
    return FileFormat::DWRF;
  }
  return FileFormat::UNKNOWN;
}

} // namespace

// Usage: velox_example_scan_file --file_path=<path> --format=<format> --config_path=<config>
int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Validate required flags
  if (FLAGS_file_path.empty()) {
    std::cerr << "Error: --file_path is required" << std::endl;
    std::cerr << "Usage: velox_example_scan_file --file_path=<path> --format=<format> --config_path=<config>" << std::endl;
    return 1;
  }

  if (FLAGS_format.empty()) {
    std::cerr << "Error: --format is required" << std::endl;
    std::cerr << "Usage: velox_example_scan_file --file_path=<path> --format=<format> --config_path=<config>" << std::endl;
    return 1;
  }

  if (FLAGS_config_path.empty()) {
    std::cerr << "Error: --config_path is required" << std::endl;
    std::cerr << "Usage: velox_example_scan_file --file_path=<path> --format=<format> --config_path=<config>" << std::endl;
    return 1;
  }

  // Validate buffer_type flag
  if (FLAGS_buffer_type != "standard" && FLAGS_buffer_type != "direct" && 
      FLAGS_buffer_type != "cached") {
    std::cerr << "Error: Invalid --buffer_type '" << FLAGS_buffer_type << "'" << std::endl;
    std::cerr << "Valid options: standard, direct, cached" << std::endl;
    return 1;
  }

  const auto format = inferFileFormat(FLAGS_format);
  if (format == FileFormat::UNKNOWN) {
    std::cerr << "Unsupported file format " << FLAGS_format << std::endl;
    std::cerr << "Supported formats: orc, dwrf, parquet (or pq)" << std::endl;
    return 1;
  }

  // To be able to read local files, we need to register the local file
  // filesystem. We also need to register the dwrf and parquet reader
  // factories.
  filesystems::registerLocalFileSystem();
  filesystems::registerS3FileSystem();
  auto config = readConfig(FLAGS_config_path);
  auto fs = filesystems::getFileSystem(FLAGS_file_path, config);
  std::shared_ptr<ReadFile> readFile = fs->openFileForRead(FLAGS_file_path);
  dwrf::registerDwrfReaderFactory();
  parquet::registerParquetReaderFactory();
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  auto pool = facebook::velox::memory::memoryManager()->addLeafPool();

  // Create an IO executor for async I/O operations
  const int32_t ioThreads = FLAGS_io_threads > 0 
      ? FLAGS_io_threads 
      : std::thread::hardware_concurrency();
  std::shared_ptr<folly::Executor> ioExecutor =
      std::make_shared<folly::IOThreadPoolExecutor>(ioThreads);

  auto dataIoStats = std::make_shared<io::IoStatistics>();
  dwio::common::ReaderOptions readerOpts(pool.get());
  readerOpts.setFileFormat(format);
  // ORC files are served by the DWRF reader factory; Parquet has its own.
  const auto factoryFormat =
      format == FileFormat::ORC ? FileFormat::DWRF : format;
  
  auto startTime = std::chrono::high_resolution_clock::now();
  std::unique_ptr<dwio::common::BufferedInput> bufferedInput;
  if (FLAGS_buffer_type == "direct") {
    // Use DirectBufferedInput for potentially better performance with async I/O
    bufferedInput = std::make_unique<dwio::common::DirectBufferedInput>(
        std::move(readFile),
        MetricsLog::voidLog(),
        StringIdLease(fileIds(), ""),
        nullptr, // tracker
        StringIdLease(fileIds(), ""),
        dataIoStats,
        nullptr,
        ioExecutor.get(), // executor for async I/O
        readerOpts);
  } else if (FLAGS_buffer_type == "cached") {
    // Use CachedBufferedInput for coalescing small reads
    cache::AsyncDataCache::Options cacheOptions;
    auto asyncDataCache = cache::AsyncDataCache::create(
        memory::memoryManager()->allocator(),
        nullptr,
        cacheOptions);
    bufferedInput = std::make_unique<dwio::common::CachedBufferedInput>(
        std::move(readFile),
        MetricsLog::voidLog(),
        StringIdLease(fileIds(), ""),
        asyncDataCache.get(), // async data cache
        nullptr, // tracker
        StringIdLease(fileIds(), ""),
        dataIoStats,
        nullptr,
        ioExecutor.get(), // executor for async I/O
        readerOpts);
  } else {
    // Use standard BufferedInput (default)
    bufferedInput = std::make_unique<BufferedInput>(
        std::move(readFile),
        readerOpts.memoryPool(),
        MetricsLog::voidLog(),
        dataIoStats.get());
  }
  
  auto reader = dwio::common::getReaderFactory(factoryFormat)
                    ->createReader(std::move(bufferedInput), readerOpts);

  // The Parquet reader expects the caller to provide an allocated result
  // vector and fills it in place; the DWRF reader allocates it on first read
  // when null. Pre-allocating here works for both.
  const vector_size_t kBatchSize = FLAGS_batch_size;
  VectorPtr batch =
      BaseVector::create(reader->rowType(), kBatchSize, pool.get());
  RowReaderOptions rowReaderOptions;
  rowReaderOptions.setRequestedType(reader->rowType());
  auto scanSpec = std::make_shared<common::ScanSpec>("");
  scanSpec->addAllChildFields(*reader->rowType());
  rowReaderOptions.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOptions);
  
  // Track timing and row count
  uint64_t totalRows = 0;
  
  while (rowReader->next(kBatchSize, batch)) {
    auto rowVector = batch->as<RowVector>();
    totalRows += rowVector->size();
    rowVector->loadedVector();
    if (FLAGS_print_rows) {
      for (vector_size_t i = 0; i < rowVector->size(); ++i) {
        std::cout << rowVector->toString(i) << std::endl;
      }
    }
  }
  
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime).count();

  // Display throughput statistics
  if (FLAGS_show_throughput) {
    const auto dataBytes = dataIoStats->rawBytesRead();
    const double durationSec = duration / 1000.0;
    const double throughputMBps = (dataBytes / (1024.0 * 1024.0)) / durationSec;
    
    std::cerr << "\n=== Scan Statistics ===" << std::endl;
    std::cerr << "Total rows: " << totalRows << std::endl;
    std::cerr << "Duration: " << std::fixed << std::setprecision(2) 
              << durationSec << " seconds" << std::endl;
    std::cerr << "Data read bytes: " << dataBytes 
              << " (" << (dataBytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cerr << "Throughput: " << std::fixed << std::setprecision(2) 
              << throughputMBps << " MB/s" << std::endl;
    std::cerr << "Rows/sec: " << std::fixed << std::setprecision(0) 
              << (totalRows / durationSec) << std::endl;
  }

  fs.reset();
  filesystems::finalizeS3FileSystem();
  
  return 0;
}
