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

#include <utility>

#include "velox/common/base/Fs.h"
#include "velox/dwio/parquet/tests/ParquetWriterTestBase.h"
#include "velox/dwio/parquet/writer/Writer.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

class SinkTest : public ParquetWriterTestBase {
 protected:
  std::pair<std::unique_ptr<parquet::Writer>, FileSink*>
  createWriterWithSinkPtr(
      const std::string& filePath,
      std::function<std::unique_ptr<DefaultFlushPolicy>()> flushPolicy) {
    auto sink = createSink(filePath);
    auto sinkPtr = sink.get();
    return {createWriter(std::move(sink), std::move(flushPolicy)), sinkPtr};
  }

  static constexpr uint64_t kRowsInRowGroup = 10'000;
  static constexpr uint64_t kBytesInRowGroup = 128 * 1'024 * 1'024;
};

TEST_F(SinkTest, close) {
  auto batches = createBatches(ROW({INTEGER(), VARCHAR()}), 2, 3);
  auto filePath = createTempPath();
  auto [writer, sinkPtr] = createWriterWithSinkPtr(filePath.string(), [&]() {
    return std::make_unique<LambdaFlushPolicy>(
        kRowsInRowGroup, kBytesInRowGroup, [&]() { return false; });
  });

  for (auto& batch : batches) {
    writer->write(batch);
  }
  writer->flush();

  auto size = sinkPtr->size();
  EXPECT_EQ(size, fs::file_size(filePath));

  for (auto& batch : batches) {
    writer->write(batch);
  }

  // Close would flush
  writer->close();
  EXPECT_LT(size, fs::file_size(filePath));
}

TEST_F(SinkTest, abort) {
  auto batches = createBatches(ROW({INTEGER(), VARCHAR()}), 2, 3);
  auto filePath = createTempPath();
  auto [writer, sinkPtr] = createWriterWithSinkPtr(filePath.string(), [&]() {
    return std::make_unique<LambdaFlushPolicy>(
        kRowsInRowGroup, kBytesInRowGroup, [&]() { return false; });
  });

  for (auto& batch : batches) {
    writer->write(batch);
  }
  writer->flush();

  auto size = sinkPtr->size();
  EXPECT_EQ(size, fs::file_size(filePath));

  for (auto& batch : batches) {
    writer->write(batch);
  }

  // Abort would not flush.
  writer->abort();
  EXPECT_EQ(size, fs::file_size(filePath));
}
