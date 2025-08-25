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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3WriteFile.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"

#include <gtest/gtest.h>

namespace {
using namespace facebook::velox::filesystems;
class S3FileSystemBenchmark {
 public:
  S3FileSystemBenchmark() {
    minioServer_ = std::make_unique<MinioServer>();
    minioServer_->start();
    ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(3);
    filesystems::initializeS3("Info", kLogLocation_);
  }
  ~S3FileSystemBenchmark() {
    minioServer_->stop();
    filesystems::finalizeS3();
  }
  std::unique_ptr<MinioServer> minioServer_;
  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
  std::string_view kLogLocation_ = "/tmp/foobar/";

  std::string localPath(const char* directory) {
    return minioServer_->path() + "/" + directory;
  }
  void addBucket(const char* bucket) {
    minioServer_->addBucket(bucket);
  }
  void
  run(const std::string& name, bool enableUploadPartAsync, int32_t size_MiB) {
    folly::BenchmarkSuspender suspender;
    const auto bucketName = "writedata";
    const auto file = fmt::format("test_{}_{}.txt", name, size_MiB);
    const auto filename = localPath(bucketName) + "/" + file.c_str();
    addBucket(bucketName);
    const auto s3File = s3URI(bucketName, file.c_str());
    auto hiveConfig = minioServer_->hiveConfig(
        {{"hive.s3.uploadPartAsync",
          enableUploadPartAsync ? "true" : "false"}});
    filesystems::S3FileSystem s3fs(bucketName, hiveConfig);
    suspender.dismiss();
    auto pool = memory::memoryManager()->addLeafPool("S3FileSystemBenchmark");
    auto writeFile =
        s3fs.openFileForWrite(s3File, {{}, pool.get(), std::nullopt});
    auto s3WriteFile = dynamic_cast<filesystems::S3WriteFile*>(writeFile.get());
    // 1024
    std::string dataContent(1024, 'a');

    EXPECT_EQ(writeFile->size(), 0);
    std::int64_t contentSize = dataContent.length();
    // dataContent length is 1024.
    EXPECT_EQ(contentSize, 1024);

    // Append and flush a small batch of data.
    writeFile->append(dataContent.substr(0, 10));
    EXPECT_EQ(writeFile->size(), 10);
    writeFile->append(dataContent.substr(10, contentSize - 10));
    EXPECT_EQ(writeFile->size(), contentSize);
    writeFile->flush();
    // No parts must have been uploaded.
    EXPECT_EQ(s3WriteFile->numPartsUploaded(), 0);

    // Append data
    for (int i = 0; i < 1024 * size_MiB - 1; ++i) {
      writeFile->append(dataContent);
    }
    writeFile->close();
    EXPECT_EQ(
        writeFile->size(), static_cast<std::int64_t>(1024) * 1024 * size_MiB);
  }
};

auto benchmark = S3FileSystemBenchmark();

#define DEFINE_BENCHMARKS(size)                     \
  BENCHMARK(non_async_upload_##size##M) {           \
    benchmark.run("non_async_upload", false, size); \
  }                                                 \
  BENCHMARK_RELATIVE(async_upload_##size##M) {      \
    benchmark.run("async_upload", true, size);      \
  }

DEFINE_BENCHMARKS(4)
DEFINE_BENCHMARKS(8)
DEFINE_BENCHMARKS(16)
DEFINE_BENCHMARKS(32)
DEFINE_BENCHMARKS(64)
DEFINE_BENCHMARKS(128)
DEFINE_BENCHMARKS(256)
DEFINE_BENCHMARKS(512)
DEFINE_BENCHMARKS(1024)
DEFINE_BENCHMARKS(2048)
} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
