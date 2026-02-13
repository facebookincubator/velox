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

// Adapted from Apache Arrow.

#pragma once

#include <memory>

#include "velox/dwio/parquet/writer/arrow/util/Compression.h"

namespace facebook::velox::parquet::arrow::util {

// ----------------------------------------------------------------------.
// Internal Codec factories.

namespace internal {

// Brotli compression quality is max (11) by default, which is slow.
// We use 8 as a default as it is the best trade-off for Parquet workload.
constexpr int kBrotliDefaultCompressionLevel = 8;

// Brotli codec.
std::unique_ptr<Codec> makeBrotliCodec(
    int compressionLevel = kBrotliDefaultCompressionLevel,
    std::optional<int> windowBits = std::nullopt);

// BZ2 codec.
constexpr int kBZ2DefaultCompressionLevel = 9;

std::unique_ptr<Codec> makeBZ2Codec(
    int compressionLevel = kBZ2DefaultCompressionLevel);

// GZip.
constexpr int kGZipDefaultCompressionLevel = 9;

std::unique_ptr<Codec> makeGZipCodec(
    int compressionLevel = kGZipDefaultCompressionLevel,
    GZipFormat format = GZipFormat::GZIP,
    std::optional<int> windowBits = std::nullopt);

// Snappy.
std::unique_ptr<Codec> makeSnappyCodec();

// Lz4 Codecs.
constexpr int kLz4DefaultCompressionLevel = 1;

// Lz4 frame format codec.

std::unique_ptr<Codec> makeLz4FrameCodec(
    int compressionLevel = kLz4DefaultCompressionLevel);

// Lz4 "raw" format codec.
std::unique_ptr<Codec> makeLz4RawCodec(
    int compressionLevel = kLz4DefaultCompressionLevel);

// Lz4 "Hadoop" format codec (== Lz4 raw codec prefixed with lengths header)
std::unique_ptr<Codec> makeLz4HadoopRawCodec();

// ZSTD codec.

// XXX level = 1 probably doesn't compress very much.
constexpr int kZSTDDefaultCompressionLevel = 1;

std::unique_ptr<Codec> makeZSTDCodec(
    int compressionLevel = kZSTDDefaultCompressionLevel);

} // namespace internal
} // namespace facebook::velox::parquet::arrow::util
