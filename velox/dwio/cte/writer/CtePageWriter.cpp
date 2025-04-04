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

#include "velox/dwio/cte/writer/CtePageWriter.h"
#include "velox/dwio/common/DataBuffer.h"

namespace {
// Spilling currently uses the default PrestoSerializer which by default
// serializes timestamp with millisecond precision to maintain compatibility
// with presto. Since velox's native timestamp implementation supports
// nanosecond precision, we use this serde option to ensure the serializer
// preserves precision.
static const bool kDefaultUseLosslessTimestamp = true;
} // namespace

namespace facebook::velox::pagefile {

CtePageWriter::CtePageWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const dwio::common::WriterOptions& options,
    std::shared_ptr<memory::MemoryPool> pool)
    : sink_(std::move(sink)),
      schema_{dwio::common::TypeWithId::create(options.schema)},
      pool_(pool),
      options_(options),
      serde_(getNamedVectorSerde(VectorSerde::Kind::kPresto)) {}

void CtePageWriter::write(const VectorPtr& input) {
  auto rowVector = std::dynamic_pointer_cast<RowVector>(input);
  VELOX_CHECK_NOT_NULL(rowVector);
  if (batch_ == nullptr) {
    serializer::presto::PrestoVectorSerde::PrestoOptions options = {
        kDefaultUseLosslessTimestamp,
        options_.compressionKind.value(),
        true /*nullsFirst*/};
    batch_ = std::make_unique<VectorStreamGroup>(pool_.get(), serde_);
    batch_->createStreamTree(
        std::static_pointer_cast<const RowType>(rowVector->type()),
        1'000,
        &options);
  }
  batch_->append(rowVector);
  flush();
}

void CtePageWriter::flush() {
  if (batch_ == nullptr) {
    return;
  }
  IOBufOutputStream out(
      *pool_, nullptr, std::max<int64_t>(64 * 1024, batch_->size()));
  batch_->flush(&out);
  batch_.reset();

  auto iobuf = out.getIOBuf();
  auto size = iobuf->length();
  dwio::common::DataBuffer<char> buf{*pool_, size};
  std::memcpy(buf.data(), iobuf->data(), size);
  sink_->write(std::move(buf));
}

void CtePageWriter::close() {
  if (batch_) {
    flush();
  }
  VELOX_CHECK_NOT_NULL(sink_);
  sink_->close();
  sink_ = nullptr;
}

void CtePageWriter::abort() {
  close();
}

std::unique_ptr<dwio::common::Writer> CtePageWriterFactory::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const std::shared_ptr<dwio::common::WriterOptions>& options) {
  return std::make_unique<CtePageWriter>(
      std::move(sink),
      *options,
      options->memoryPool->addAggregateChild(fmt::format(
          "{}.pagefile.{}",
          options->memoryPool->name(),
          folly::to<std::string>(folly::Random::rand64()))));
}

void registerCtePageWriterFactory() {
  dwio::common::registerWriterFactory(std::make_shared<CtePageWriterFactory>());
}

void unregisterCtePageWriterFactory() {
  dwio::common::unregisterWriterFactory(dwio::common::FileFormat::PAGEFILE);
}
} // namespace facebook::velox::pagefile
