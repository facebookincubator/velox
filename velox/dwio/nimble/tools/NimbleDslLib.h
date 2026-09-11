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
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"

#include "velox/dwio/nimble/tools/NimbleDumpLib.h"

namespace facebook::nimble::tools {

/// NimbleDslLib provides SQL-like query operations on Nimble files.
///
/// It powers the `nimble_dsl` interactive REPL, allowing users to inspect
/// Nimble file contents using familiar SQL-like syntax instead of working
/// at the raw stream/encoding level (which is what `nimble_dump` does).
///
/// Usage:
///   // From a file path (uses FileSystem::openForRead internally):
///   NimbleDslLib sql(std::cout, enableColors, "/path/to/data.nimble");
///
///   // From an in-memory ReadFile (useful for testing):
///   auto readFile = std::make_shared<velox::InMemoryReadFile>(data);
///   NimbleDslLib sql(std::cout, enableColors, readFile);
///
///   sql.describe();                             // DESCRIBE
///   sql.select({}, 10, 0, std::nullopt);        // SELECT * LIMIT 10
///   sql.select({"col1"}, 5, 0, std::nullopt);   // SELECT col1 LIMIT 5
///   sql.showInfo();                              // SHOW INFO
///
/// All output is written to the ostream provided at construction time.
/// When enableColors is true, ANSI color codes are used for table headers.
class NimbleDslLib {
 public:
  /// Construct from a file path. The file is opened via the velox
  /// filesystem interface.
  NimbleDslLib(
      std::ostream& ostream,
      bool enableColors,
      const std::string& filePath);

  /// Construct from a pre-opened ReadFile. Useful for in-memory testing
  /// with velox::InMemoryReadFile.
  NimbleDslLib(
      std::ostream& ostream,
      bool enableColors,
      std::shared_ptr<velox::ReadFile> file);

  /// Read and display row data from the file.
  ///
  /// @param columns Column names to project. Empty vector means all columns.
  /// @param limit   Maximum number of rows to display.
  /// @param offset  Number of rows to skip from the beginning of the file
  ///                (or from the beginning of the specified stripe).
  /// @param stripeId If set, restrict reading to this stripe only.
  void select(
      const std::vector<std::string>& columns,
      uint64_t limit,
      uint64_t offset,
      std::optional<uint32_t> stripeId);

  /// Display a table of top-level column names, their Velox types, and
  /// the corresponding Nimble stream offsets.
  void describe();

  /// Display the full Nimble schema tree with stream offsets, type kinds,
  /// and scalar kinds. This shows the internal Nimble type hierarchy
  /// including nested types (arrays, maps, rows, flat maps).
  void showSchema();

  /// Display file-level metadata: Nimble version, file size, checksum,
  /// stripe count, row count, and user-defined metadata key-value pairs.
  void showInfo();

  /// Display per-column statistics from the vectorized stats optional
  /// section: value count, null count, min/max values, logical size,
  /// and physical size. Requires the file to have been written with
  /// enableVectorizedStats = true.
  void showStats();

  /// Display stripe-level information: stripe ID, byte offset, byte size,
  /// and row count for each stripe in the file.
  void showStripes();

  /// Display stream-level information: stream ID, byte offset, byte size,
  /// item count, and human-readable stream label for each stream.
  ///
  /// @param stripeId If set, show streams for this stripe only.
  ///                 Otherwise, show streams for all stripes.
  void showStreams(std::optional<uint32_t> stripeId);

  /// Display per-stream encoding information: the encoding tree (encoding
  /// types, data types, compression) for each stream in each stripe.
  ///
  /// @param stripeId If set, show encodings for this stripe only.
  ///                 Otherwise, show encodings for all stripes.
  void showEncoding(std::optional<uint32_t> stripeId);

  /// Display index information: index columns with sort orders,
  /// index group metadata, and key stream regions per stripe.
  void showIndex();

  /// Display encoding type distribution with instance counts and
  /// storage percentage of the file size.
  ///
  /// @param topLevel If true, only show top-level encodings (don't recurse).
  /// @param stripeId If set, show histogram for this stripe only.
  void showHistogram(bool topLevel, std::optional<uint32_t> stripeId);

  /// Dump raw decoded stream values for a given stream ID.
  ///
  /// @param streamId The stream to decode and display.
  /// @param stripeId If set, show content for this stripe only.
  void showContent(uint32_t streamId, std::optional<uint32_t> stripeId);

  /// Display the physical file layout: offsets, sizes, and compression
  /// for each section (stripes, stripe groups, optional sections,
  /// footer, postscript).
  void showFileLayout();

  /// Display stripes metadata section details: offset, size,
  /// and compression type.
  void showStripesMetadata();

  /// Display stripe groups metadata: group ID, offset, size,
  /// and compression type for each stripe group.
  void showStripeGroupsMetadata();

  /// Display optional sections metadata: name, compression,
  /// offset, and size for each optional section.
  void showOptionalSections();

 private:
  const std::shared_ptr<velox::memory::MemoryPool> pool_;
  const std::shared_ptr<velox::ReadFile> file_;
  const bool enableColors_;
  std::ostream& ostream_;
  NimbleDumpLib dumpLib_;
};

} // namespace facebook::nimble::tools
