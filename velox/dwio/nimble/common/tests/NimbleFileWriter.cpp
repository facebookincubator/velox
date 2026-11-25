/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"

#include <deque>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"

namespace facebook::nimble::test {
namespace {

// Emits every buffered input as its own stripe, so writes of 3 and 2 rows give
// stripes of 3 and 2 rows.
class StripePerWriteBufferPolicy final : public BufferPolicy {
 public:
  void bufferInput(velox::VectorPtr input) override {
    NIMBLE_USER_CHECK_NOT_NULL(input, "Buffered input must not be null");
    NIMBLE_USER_CHECK_GT(input->size(), 0, "Buffered input must not be empty");
    BufferRange range;
    range.rowRanges.emplace_back(0, static_cast<uint32_t>(input->size()));
    range.inputs.push_back(std::move(input));
    ready_.push_back(std::move(range));
  }

  BufferRange flushInput() override {
    if (ready_.empty()) {
      return {};
    }
    auto range = std::move(ready_.front());
    ready_.pop_front();
    return range;
  }

  void finalize() override {}

 private:
  // Stripes not yet returned by flushInput(), oldest first.
  std::deque<BufferRange> ready_;
};

} // namespace

std::function<std::unique_ptr<BufferPolicy>()>
stripePerWriteBufferPolicyFactory() {
  return [] { return std::make_unique<StripePerWriteBufferPolicy>(); };
}

std::string createNimbleFile(
    velox::memory::MemoryPool& memoryPool,
    const velox::VectorPtr& vector,
    nimble::WriterOptions writerOptions,
    bool flushAfterWrite) {
  return createNimbleFile(
      memoryPool,
      std::vector<facebook::velox::VectorPtr>{vector},
      std::move(writerOptions),
      flushAfterWrite);
}

std::string createNimbleFile(
    velox::memory::MemoryPool& memoryPool,
    const std::vector<facebook::velox::VectorPtr>& vectors,
    nimble::WriterOptions writerOptions,
    bool flushAfterWrite) {
  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  NIMBLE_CHECK_GT(vectors.size(), 0, "Expecting at least one input vector.");
  auto& type = vectors[0]->type();

  for (int i = 1; i < vectors.size(); ++i) {
    NIMBLE_CHECK(
        vectors[i]->type()->kindEquals(type),
        "All vectors should have the same schema.");
  }

  nimble::Writer writer(
      type, std::move(writeFile), memoryPool, std::move(writerOptions));
  for (const auto& vector : vectors) {
    writer.write(vector);
    if (flushAfterWrite) {
      writer.flush();
    }
  }
  writer.close();

  return file;
}

} // namespace facebook::nimble::test
