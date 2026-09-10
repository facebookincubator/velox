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
#pragma once

#include <folly/Range.h>

#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/WriterFactory.h"

#include <folly/container/F14Map.h>

namespace facebook::velox::nimble {

// Serde-parameter key controlling the BufferedWriteFile capacity (bytes)
// wrapped around the WriteFile passed to the underlying nimble::Writer.
// Default behavior (key absent or value 0): no buffering wrapper, raw
// WriteFile passed directly to the writer. Setting to a positive value
// (e.g. "4194304" for 4MB) coalesces many small Nimble stream-chunk appends
// into fewer larger downstream pwrite calls, drastically reducing per-pwrite
// WarmStorage logging/tracing malloc overhead.
//
// Users can set this from a Spark conf via Gluten's existing config bridge
// (no OSS changes needed):
//   spark.conf.set(
//     "spark.gluten.sql.columnar.backend.velox.nimble.write.file_buffer_size_bytes",
//     "8388608")
constexpr folly::StringPiece kNimbleWriteFileBufferSizeBytesKey =
    "nimble.write.file_buffer_size_bytes";

// Nimble writer options exposed to Velox library users. We shold only expose
// engine options (like thread pools) but not query-specific ones. The latter
// should use query writer serde options.
struct NimbleWriterOptions : public dwio::common::WriterOptions {
  folly::Executor::KeepAlive<> encodingExecutor;

  /// Per-type attributes routed through to
  /// `facebook::nimble::WriterOptions::schemaAttributes` when this
  /// dwio::common::WriterOptions instance is used to spawn a NIMBLE writer.
  /// Keyed by pre-order schema node id (matching `TypeWithId::id()`). Empty by
  /// default (no-op for legacy callers that never populate it; resulting files
  /// are byte-identical to pre-attributes output).
  ///
  /// Populated by the Iceberg connector's `NimbleWriterOptionsAdapter` to stamp
  /// Iceberg `iceberg.id` (and V3 type) attributes onto each NIMBLE schema
  /// node.
  folly::F14FastMap<uint32_t, std::vector<std::pair<std::string, std::string>>>
      schemaAttributes;

  void processConfigs(
      const velox::config::ConfigBase& connectorConfig,
      const velox::config::ConfigBase& session) override;
};

class WriterFactory : public dwio::common::WriterFactory {
 public:
  // The base must be named explicitly: an unqualified WriterFactory(...) here
  // resolves to this class, making the initializer a delegating constructor to
  // itself rather than a call to the base.
  WriterFactory()
      : dwio::common::WriterFactory(dwio::common::FileFormat::NIMBLE) {}

  std::unique_ptr<dwio::common::Writer> createWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      const std::shared_ptr<dwio::common::WriterOptions>& options) override;

  std::unique_ptr<dwio::common::WriterOptions> createWriterOptions() override;
};

inline void registerNimbleWriterFactory() {
  dwio::common::registerWriterFactory(std::make_shared<WriterFactory>());
}

inline void unregisterNimbleWriterFactory() {
  dwio::common::unregisterWriterFactory(dwio::common::FileFormat::NIMBLE);
}

} // namespace facebook::velox::nimble
