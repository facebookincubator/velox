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

#include <string_view>

namespace facebook::nimble {

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
constexpr std::string_view kIndexSection = "columnar.indexes";
constexpr std::string_view kChunkStatsSection = "columnar.chunk.stats";
constexpr std::string_view kFeaturesSection = "columnar.features";
constexpr std::string_view kSharedDictionarySection =
    "columnar.shared_dictionaries";

} // namespace facebook::nimble
