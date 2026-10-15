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

#include <cstdint>
#include <string_view>

namespace facebook::nimble {

/// Selects the on-disk representation for chunk statistics.
enum class ChunkStatsVersion : uint8_t {
  /// Stores statistics as raw FlatBuffer arrays.
  kV1 = 1,
  /// Stores statistics as Nimble-encoded arrays.
  kV2 = 2,
};

constexpr uint16_t kMagicNumber = 0xA1FA;
constexpr uint64_t kInitialFooterSize = 8 * 1024 * 1024; // 8MB
constexpr uint16_t kVersionMajor = 0;
constexpr uint16_t kVersionMinor = 1;

// Total size of the fields after the flatbuffer.
constexpr uint32_t kPostscriptSize = 20;

// The following fields in postscript are included in checksum calculation.
// 4 bytes footer size + 1 byte compression type
constexpr uint32_t kPostscriptChecksumedSize = 5;

constexpr std::string_view kSchemaSection = "columnar.schema";
constexpr std::string_view kMetadataSection = "columnar.metadata";
constexpr std::string_view kStatsSection = "columnar.stats";
constexpr std::string_view kVectorizedStatsSection =
    "columnar.vectorized_stats";
constexpr std::string_view kStripeStatsSection = "columnar.stripe_stats";
constexpr std::string_view kIndexSection = "columnar.indexes";
constexpr std::string_view kChunkStatsSection = "columnar.chunk.stats";
constexpr std::string_view kChunkStatsV2Section = "columnar.chunk.stats.v2";
constexpr std::string_view kPropertiesSection = "columnar.properties";
constexpr std::string_view kDictionarySection = "columnar.dictionaries";
constexpr std::string_view kVectorIndexSection = "columnar.vector.index";
/// Present only in a suspended file, i.e. one closed without being finalized
/// so that a later writer can reopen it and append. Its presence is the signal
/// that the file is not final; a finalized file never carries it.
constexpr std::string_view kCheckpointSection = "columnar.checkpoint";

/// Version written into, and accepted from, the checkpoint section. Versions
/// start at 1 so that a flatbuffers scalar left at its implicit default is
/// recognizable as absent rather than read as version 0.
constexpr uint32_t kCheckpointVersionMin = 1;
constexpr uint32_t kCheckpointVersion = 1;

} // namespace facebook::nimble
