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

// Fixture files for the options-off tests (OptionsOffGoldenTest.cpp and an
// internal test): one deterministic set of files that reaches every
// FixedBitWidth decision the typed-lanes options can change, written through
// every writer path.
//
// These fixtures use only writer APIs that existed before the typed-lanes
// options, so a build of that earlier code can write them too. That is how
// the digests in OptionsOffGolden.h are produced: the earlier build and this
// one write every fixture, and the digests must agree. Keep it that way: the
// typed-lanes options (WriterOptions::fixedBitWidthTypedLanes and the options
// added with it) are set only by the tests, never here.
//
// Every fixture file is a function of its inputs alone:
// - No Managed Compression, whose output depends on configs served at run
//   time. Compression is either off, or Zstd with a minimum compression size
//   no payload reaches. The second runs every compression-enabled encoder
//   path, but no compressor library touches the stored bytes.
// - Footer metadata is never compressed, and vectorized statistics are off:
//   statistics streams compress with a policy of their own.
// - Encoding is serial, every batch is one chunk, and stripes end after fixed
//   batches.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble::test::options_off {

/// How a fixture compresses.
enum class FixtureCompression {
  /// No compression.
  kUncompressed,
  /// Zstd, with a minimum compression size no payload reaches.
  kZstdBelowMinSize,
};

/// The writer path a fixture exercises.
enum class FixturePath {
  /// Fresh selection with the default read factors.
  kFreshDefaultFactors,
  /// Fresh selection, FixedBitWidth for every numeric stream.
  kFreshFixedBitWidth,
  /// The encoding selection cache: chunks after the first are replayed.
  kSelectionCache,
  /// An external encoding layout tree replayed on column i64_24.
  kLayoutTree,
  /// A layout tree that cannot encode i64_24, so replay falls back to a fresh
  /// selection.
  kReplayFallback,
  /// A stripe-scoped shared dictionary on column lowcard.
  kSharedDictionary,
  /// Column split pinned to SubIntSplit, and column fsst pinned to FSST.
  kPinnedEncodings,
};

/// One fixture file.
struct Fixture {
  /// Unique name, also the key of its golden digest.
  std::string name;

  FixtureCompression compression;

  FixturePath path;

  /// WriterOptions::fixedBitWidthUseExactBits.
  bool exactBits;
};

/// A written fixture file.
struct WrittenFile {
  /// The file's bytes.
  std::string bytes;

  /// The names of the writer's runtime statistics (Writer::runtimeStats()).
  std::vector<std::string> runtimeStatNames;
};

/// Returns every fixture: each path under both compressions, with exact bits
/// off and on (28 fixtures).
std::vector<Fixture> allFixtures();

/// Returns the batches every fixture writes, one chunk each: 4,096, 257, 256,
/// 255 and 100 rows, so chunks fall on both sides of the default 256-value
/// floor of typed lanes. The columns cover every stored bit width that lanes
/// treat differently, nested streams (nulls, arrays, maps), dictionary and
/// run-length candidates, strings, floating point and booleans; see the
/// definition for each column.
std::vector<velox::RowVectorPtr> makeBatches(velox::memory::MemoryPool& pool);

/// Returns whether a stripe ends after batch 'batch' (0-based). Stripes hold
/// 4,096 + 257, 256 + 255 and 100 rows.
bool endsStripe(size_t batch);

/// Returns the compression options of 'compression'.
CompressionOptions compressionOptions(FixtureCompression compression);

/// Returns the writer options of 'fixture', with its own compression. No
/// typed-lanes option is touched.
WriterOptions fixtureOptions(const Fixture& fixture);

/// Returns the writer options of 'fixture' with 'compression' in place of the
/// fixture's own, for the encoding selection policies, any replayed layout and
/// WriterOptions::compressionOptions alike.
WriterOptions fixtureOptions(
    const Fixture& fixture,
    const CompressionOptions& compression);

/// Writes 'batches' with 'options', one chunk per batch, ending stripes where
/// endsStripe() says.
WrittenFile writeFile(
    const std::vector<velox::RowVectorPtr>& batches,
    WriterOptions options,
    velox::memory::MemoryPool& rootPool);

/// Returns the digest recorded in OptionsOffGolden.h for a file: XXH3-64 of
/// its bytes.
uint64_t fileDigest(std::string_view bytes);

} // namespace facebook::nimble::test::options_off
