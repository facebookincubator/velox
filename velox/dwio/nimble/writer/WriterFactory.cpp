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
#include "velox/dwio/nimble/writer/WriterFactory.h"

#include <fmt/format.h>
#include <folly/Conv.h>
#include <folly/Random.h>

#include "dwio/utils/BufferedWriteFile.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"
#include "velox/dwio/nimble/velox/NimbleConfig.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/dwio/nimble/writer/fb/NimbleWriterOptionBuilder.h"

namespace facebook::velox::nimble {

// Table serde param > session property > connector config > default.
// emplace preserves existing serde params (table-level), and getWithFallback
// checks session before connector config.
void NimbleWriterOptions::processConfigs(
    const velox::config::ConfigBase& connectorConfig,
    const velox::config::ConfigBase& session) {
  using NimbleConfig = facebook::nimble::Config;

  auto value = session.getWithFallback<std::string>(
      NimbleConfig::kNimbleWriteTargetRawStripeSize, connectorConfig);

  if (value.has_value()) {
    serdeParameters.emplace(
        NimbleConfig::RAW_STRIPE_SIZE.key,
        std::to_string(
            config::toCapacity(value.value(), config::CapacityUnit::BYTE)));
  }

  // [BufferedWriteFile control] Honor a user-set Spark conf for the buffered
  // write file capacity. Gluten's existing conf bridge auto-forwards Spark
  // confs prefixed with "spark.gluten.sql.columnar.backend.velox." into both
  // the Velox session config and the Hive connector config (prefix stripped).
  // So a user can set this from a Spark conf without any per-table change:
  //   spark.conf.set(
  //     "spark.gluten.sql.columnar.backend.velox.nimble.write.file_buffer_size_bytes",
  //     "8388608")
  // The value is propagated here as a serde param under the same key, where
  // WriterFactory::createWriter (below) reads it. Table-level serde
  // params still take precedence (emplace below is a no-op if already set).
  auto bufferBytes = session.getWithFallback<std::string>(
      std::string(kNimbleWriteFileBufferSizeBytesKey), connectorConfig);
  if (bufferBytes.has_value()) {
    serdeParameters.emplace(
        std::string(kNimbleWriteFileBufferSizeBytesKey), bufferBytes.value());
  }
}

std::unique_ptr<dwio::common::Writer> WriterFactory::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const std::shared_ptr<dwio::common::WriterOptions>& options) {
  auto builder =
      facebook::dwio::api::NimbleWriterOptionBuilder()
          .withSerdeParams(asRowType(options->schema), options->serdeParameters)
          .withReclaimerFactory(options->memoryReclaimerFactory)
          .withSpillConfig(options->spillConfig);

  auto* nimbleOptions = dynamic_cast<NimbleWriterOptions*>(options.get());
  if (nimbleOptions && nimbleOptions->encodingExecutor) {
    builder.withEncodingExecutor(nimbleOptions->encodingExecutor);
  }
  if (nimbleOptions && !nimbleOptions->schemaAttributes.empty()) {
    builder.withSchemaAttributes(nimbleOptions->schemaAttributes);
  }

  // NOTE: Index configuration is parsed automatically by withSerdeParams()
  // via getIndexConfigFromSerdeParams() in NimbleWriterOptionBuilder.cpp

  // Optionally wrap the raw WriteFile with BufferedWriteFile so that many
  // small Nimble stream chunk appends coalesce into fewer, larger downstream
  // pwrite calls. The buffer size is opt-in via the serde parameter
  // `nimble.write.file_buffer_size_bytes`; a value of 0 (the default) keeps
  // the legacy behavior of passing the raw WriteFile directly. Without
  // buffering, every Nimble append incurs the full per-pwrite WarmStorage
  // scuba-logging + TraceContext malloc cost — note that the JVM/AlphaJNI
  // Nimble write path has always wrapped with BufferedWriteFile (default
  // 8MB capacity, see JniAlphaWriterOptions.java), so this brings parity.
  uint64_t writeFileBufferBytes = 0;
  auto bufIt = options->serdeParameters.find(
      std::string(kNimbleWriteFileBufferSizeBytesKey));
  if (bufIt != options->serdeParameters.end()) {
    auto parsed = folly::tryTo<uint64_t>(bufIt->second);
    VELOX_USER_CHECK(
        parsed.hasValue(),
        "{} must be a non-negative integer byte count, got: '{}'",
        kNimbleWriteFileBufferSizeBytesKey,
        bufIt->second);
    writeFileBufferBytes = parsed.value();
  }

  auto maybeBuffer = [&](std::unique_ptr<velox::WriteFile> file)
      -> std::unique_ptr<velox::WriteFile> {
    if (writeFileBufferBytes == 0) {
      return file;
    }
    auto bufferedFilePool = options->memoryPool->addAggregateChild(
        fmt::format(
            "{}.nimble.buffered_write_file.{}",
            options->memoryPool->name(),
            folly::to<std::string>(folly::Random::rand64())));
    auto bufferedFileLeaf =
        bufferedFilePool->addLeafChild("buffered_write_file_buffer");
    return std::make_unique<facebook::dwio::api::BufferedWriteFile>(
        std::move(bufferedFileLeaf), writeFileBufferBytes, std::move(file));
  };

  auto writerOptions = builder.build();

  // TODO: Pass the sink directly to writer.
  if (auto* writeFileSinkWrapper =
          dynamic_cast<dwio::common::WriteFileSink*>(sink.get())) {
    writerOptions.ioStatistics = writeFileSinkWrapper->getIoStatistics();
    return std::make_unique<facebook::nimble::Writer>(
        options->schema,
        maybeBuffer(writeFileSinkWrapper->toWriteFile()),
        *options->memoryPool,
        std::move(writerOptions));
  } else if (
      auto* localFileSinkWrapper =
          dynamic_cast<dwio::common::LocalFileSink*>(sink.get())) {
    writerOptions.ioStatistics = localFileSinkWrapper->getIoStatistics();
    return std::make_unique<facebook::nimble::Writer>(
        options->schema,
        maybeBuffer(localFileSinkWrapper->toWriteFile()),
        *options->memoryPool,
        std::move(writerOptions));
  } else {
    NIMBLE_FAIL("Expected WriteFileSink, got {}", typeid(*sink).name());
  }
}

std::unique_ptr<dwio::common::WriterOptions>
WriterFactory::createWriterOptions() {
  return std::make_unique<NimbleWriterOptions>();
}

} // namespace facebook::velox::nimble
