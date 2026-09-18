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
#include "velox/dwio/nimble/compression/CompressionPolicy.h"

namespace facebook::nimble {

ConfiguredCompressionPolicy::ConfiguredCompressionPolicy(
    CompressionOptions compressionOptions,
    EncodingType encodingType)
    : compressionOptions_{std::move(compressionOptions)},
      effectiveAcceptRatio_{getAcceptRatio(encodingType)} {}

CompressionConfig ConfiguredCompressionPolicy::config() const {
  if (compressionOptions_.compressionType == CompressionType::Uncompressed) {
    return {.compressionType = CompressionType::Uncompressed};
  }

  if (compressionOptions_.compressionType == CompressionType::Zstd) {
    CompressionConfig config{
        .compressionType = CompressionType::Zstd,
        .minCompressionSize = compressionOptions_.zstdMinCompressionSize};
    config.parameters.zstd.compressionLevel =
        compressionOptions_.zstdCompressionLevel;
    return config;
  }

  if (compressionOptions_.compressionType == CompressionType::Lz4) {
    CompressionConfig config{
        .compressionType = CompressionType::Lz4,
        .minCompressionSize = compressionOptions_.lz4MinCompressionSize};
    config.parameters.lz4.accelerationLevel =
        compressionOptions_.lz4AccelerationLevel;
    return config;
  }

  if (compressionOptions_.compressionType == CompressionType::OpenZL) {
    CompressionConfig config{
        .compressionType = CompressionType::OpenZL,
        .minCompressionSize = compressionOptions_.openzlMinCompressionSize};
    config.parameters.openzl.compressionLevel =
        compressionOptions_.openzlCompressionLevel;
    config.parameters.openzl.decompressionLevel =
        compressionOptions_.openzlDecompressionLevel;
    config.parameters.openzl.formatVersion =
        compressionOptions_.openzlFormatVersion;
    return config;
  }

  CompressionConfig config{
      .compressionType = CompressionType::MetaInternal,
      .minCompressionSize = compressionOptions_.internalMinCompressionSize};
  config.parameters.metaInternal.compressionLevel =
      compressionOptions_.internalCompressionLevel;
  config.parameters.metaInternal.decompressionLevel =
      compressionOptions_.internalDecompressionLevel;
  config.parameters.metaInternal.useVariableBitWidthCompressor =
      compressionOptions_.useVariableBitWidthCompressor;
  config.parameters.metaInternal.compressionKey =
      compressionOptions_.metaInternalCompressionKey;
  return config;
}

bool ConfiguredCompressionPolicy::shouldAccept(
    CompressionType /* compressionType */,
    uint64_t uncompressedSize,
    uint64_t compressedSize) const {
  if (uncompressedSize * effectiveAcceptRatio_ < compressedSize) {
    return false;
  }

  return true;
}

float ConfiguredCompressionPolicy::getAcceptRatio(
    EncodingType encodingType) const {
  for (const auto& [enc, ratio] :
       compressionOptions_.compressionAcceptRatioOverrides) {
    if (enc == encodingType) {
      return ratio;
    }
  }
  return compressionOptions_.compressionAcceptRatio;
}

ReplayedCompressionPolicy::ReplayedCompressionPolicy(
    CompressionType compressionType,
    CompressionOptions compressionOptions)
    : compressionType_{compressionType},
      compressionOptions_{std::move(compressionOptions)} {}

CompressionConfig ReplayedCompressionPolicy::config() const {
  if (compressionType_ == CompressionType::Uncompressed) {
    return {.compressionType = CompressionType::Uncompressed};
  }

  if (compressionType_ == CompressionType::Zstd) {
    CompressionConfig config{
        .compressionType = CompressionType::Zstd,
        .minCompressionSize = compressionOptions_.zstdMinCompressionSize};
    config.parameters.zstd.compressionLevel =
        compressionOptions_.zstdCompressionLevel;
    return config;
  }

  if (compressionType_ == CompressionType::Lz4) {
    CompressionConfig config{
        .compressionType = CompressionType::Lz4,
        .minCompressionSize = compressionOptions_.lz4MinCompressionSize};
    config.parameters.lz4.accelerationLevel =
        compressionOptions_.lz4AccelerationLevel;
    return config;
  }

  if (compressionType_ == CompressionType::OpenZL) {
    CompressionConfig config{
        .compressionType = CompressionType::OpenZL,
        .minCompressionSize = compressionOptions_.openzlMinCompressionSize};
    config.parameters.openzl.compressionLevel =
        compressionOptions_.openzlCompressionLevel;
    config.parameters.openzl.decompressionLevel =
        compressionOptions_.openzlDecompressionLevel;
    config.parameters.openzl.formatVersion =
        compressionOptions_.openzlFormatVersion;
    return config;
  }

#ifdef DISABLE_META_INTERNAL_COMPRESSOR
  // Replaying a layout recorded by an internal writer can ask for the Meta
  // internal compressor, which has no OSS implementation and is not put in the
  // registry by Compression.cpp. Falling through to it would throw from
  // getCompressor(), so redirect to Zstd.
  CompressionConfig config{
      .compressionType = CompressionType::Zstd,
      .minCompressionSize = compressionOptions_.zstdMinCompressionSize};
  config.parameters.zstd.compressionLevel =
      compressionOptions_.zstdCompressionLevel;
  return config;
#else
  CompressionConfig config{
      .compressionType = CompressionType::MetaInternal,
      .minCompressionSize = compressionOptions_.internalMinCompressionSize};
  config.parameters.metaInternal.compressionLevel =
      compressionOptions_.internalCompressionLevel;
  config.parameters.metaInternal.decompressionLevel =
      compressionOptions_.internalDecompressionLevel;
  config.parameters.metaInternal.useVariableBitWidthCompressor =
      compressionOptions_.useVariableBitWidthCompressor;
  config.parameters.metaInternal.compressionKey =
      compressionOptions_.metaInternalCompressionKey;
  return config;
#endif
}

bool ReplayedCompressionPolicy::shouldAccept(
    CompressionType /* compressionType */,
    uint64_t uncompressedSize,
    uint64_t compressedSize) const {
  return compressedSize <=
      uncompressedSize * compressionOptions_.compressionAcceptRatio;
}

} // namespace facebook::nimble
