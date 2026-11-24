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

#include "velox/dwio/parquet/writer/arrow/Writer.h"

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::parquet::arrow {
namespace {

// Fails the first write after arm() and passes every other write through to
// an in-memory buffer.
class FailOnceOutputStream : public ::arrow::io::OutputStream {
 public:
  FailOnceOutputStream()
      : buffer_(::arrow::io::BufferOutputStream::Create().ValueOrDie()) {}

  void arm() {
    armed_ = true;
  }

  ::arrow::Status Close() override {
    return buffer_->Close();
  }

  bool closed() const override {
    return buffer_->closed();
  }

  ::arrow::Result<int64_t> Tell() const override {
    return buffer_->Tell();
  }

  using ::arrow::io::OutputStream::Write;

  ::arrow::Status Write(const void* data, int64_t numBytes) override {
    if (armed_) {
      armed_ = false;
      return ::arrow::Status::IOError("Injected write failure");
    }
    return buffer_->Write(data, numBytes);
  }

 private:
  std::shared_ptr<::arrow::io::BufferOutputStream> buffer_;
  bool armed_{false};
};

TEST(WriterTest, closeReturnsRowGroupError) {
  auto schema = ::arrow::schema({::arrow::field("c0", ::arrow::int64())});
  ::arrow::Int64Builder builder;
  ASSERT_TRUE(builder.AppendValues({1, 2, 3}).ok());
  auto batch =
      ::arrow::RecordBatch::Make(schema, 3, {builder.Finish().ValueOrDie()});

  auto sink = std::make_shared<FailOnceOutputStream>();
  auto writer =
      arrow::FileWriter::open(*schema, ::arrow::default_memory_pool(), sink)
          .ValueOrDie();
  ASSERT_TRUE(writer->writeRecordBatch(*batch).ok());

  // The batch is buffered, so its column data reaches the sink only when
  // close() finishes the row group. The footer that close() writes next
  // succeeds.
  sink->arm();
  const auto status = writer->close();
  EXPECT_TRUE(status.IsIOError()) << status.ToString();
  EXPECT_THAT(status.message(), testing::HasSubstr("Injected write failure"));
}

} // namespace
} // namespace facebook::velox::parquet::arrow
