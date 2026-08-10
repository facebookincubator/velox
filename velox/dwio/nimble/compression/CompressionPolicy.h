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
#include <string>
#include <utility>
#include <vector>

#include "folly/json/json.h"
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {

///
/// Compression policy type definitions:
/// A compression policy defines which compression algorithm to apply on the
/// data (if any) and what parameters to use for this compression algorithm. In
/// addition, once compression is applied to the data, the compression policy
/// can decide if the compressed result is statisfactory or if it should be
/// discarded.
struct ZstdCompressionParameters {
  int16_t compressionLevel = 3;
};

struct Lz4CompressionParameters {
  int16_t accelerationLevel = 1;
};

/// An identifier for the meta internal compression policy.
class MetaInternalCompressionKey {
 public:
  MetaInternalCompressionKey() = default;

  MetaInternalCompressionKey(
      std::string ns,
      std::string tableName,
      std::string columnName)
      : ns_{std::move(ns)},
        tableName_{std::move(tableName)},
        columnName_{std::move(columnName)} {}

  const std::string& ns() const {
    return ns_;
  }

  const std::string& tableName() const {
    return tableName_;
  }

  const std::string& columnName() const {
    return columnName_;
  }

  std::string toString() const {
    folly::dynamic json = folly::dynamic::object("ns", ns_)(
        "tableName", tableName_)("columnName", columnName_);
    return folly::toJson(json);
  }

  static MetaInternalCompressionKey fromString(const std::string& str) {
    auto json = folly::parseJson(str);
    return MetaInternalCompressionKey{
        json["ns"].asString(),
        json["tableName"].asString(),
        json["columnName"].asString()};
  }

 private:
  std::string ns_;
  std::string tableName_;
  std::string columnName_;
};

struct MetaInternalCompressionParameters {
  int16_t compressionLevel = 0;
  int16_t decompressionLevel = 0;
  bool useVariableBitWidthCompressor = true;
  MetaInternalCompressionKey compressionKey;
};

struct OpenZLCompressionParameters {
  int compressionLevel = 6;
  int decompressionLevel = 3;
  int formatVersion = 25; // current prod max, as of 2026-06-10
};

struct CompressionParameters {
  ZstdCompressionParameters zstd{};
  Lz4CompressionParameters lz4{};
  MetaInternalCompressionParameters metaInternal{};
  OpenZLCompressionParameters openzl{};
};

struct CompressionConfig {
  CompressionType compressionType{};
  CompressionParameters parameters{};
  uint64_t minCompressionSize = 0;
};

struct CompressionOptions {
  /// Rejects compression when compressedSize exceeds uncompressedSize
  /// multiplied by this ratio. Enforced by every compressor through
  /// CompressionPolicy::shouldAccept().
  float compressionAcceptRatio = 0.98f;
#ifndef DISABLE_META_INTERNAL_COMPRESSOR
  CompressionType compressionType = CompressionType::MetaInternal;
#else
  CompressionType compressionType = CompressionType::Zstd;
#endif
  uint64_t zstdMinCompressionSize = kZstdMinCompressionSize;
  uint32_t zstdCompressionLevel = 3;
  uint64_t lz4MinCompressionSize = kLz4MinCompressionSize;
  uint32_t lz4AccelerationLevel = 1;
  uint64_t internalMinCompressionSize = kMetaInternalMinCompressionSize;
  uint32_t internalCompressionLevel = 4;
  uint32_t internalDecompressionLevel = 2;
  bool useVariableBitWidthCompressor = false;
  MetaInternalCompressionKey metaInternalCompressionKey;
  uint64_t openzlMinCompressionSize = kOpenZLMinCompressionSize;
  int32_t openzlCompressionLevel = 6;
  int32_t openzlDecompressionLevel = 3;
  int32_t openzlFormatVersion = 25;
  /// Per-encoding overrides for compressionAcceptRatio.
  /// BlockBitPacking default 0.7: data is already well-packed via per-block
  /// baselines, so compression must save at least 30% to justify CPU cost.
  std::vector<std::pair<EncodingType, float>> compressionAcceptRatioOverrides =
      {{EncodingType::BlockBitPacking, 0.7f}};
};

class CompressionPolicy {
 public:
  virtual CompressionConfig config() const = 0;
  virtual bool shouldAccept(
      CompressionType /* compressionType */,
      uint64_t /* uncompressedSize */,
      uint64_t /* compressedSize */) const = 0;

  virtual ~CompressionPolicy() = default;
};

/// Default compression policy. Default behavior (if not compression policy is
/// provided) is to not compress.
class NoCompressionPolicy : public CompressionPolicy {
 public:
  CompressionConfig config() const override {
    return {.compressionType = CompressionType::Uncompressed};
  }

  virtual bool shouldAccept(
      CompressionType /* compressionType */,
      uint64_t /* uncompressedSize */,
      uint64_t /* compressedSize */) const override {
    return false;
  }
};

/// Compression policy backed by CompressionOptions.
/// Requests the configured compression type and uses
/// CompressionOptions::compressionAcceptRatio to decide whether to keep the
/// compressed result.
class ConfiguredCompressionPolicy : public CompressionPolicy {
 public:
  ConfiguredCompressionPolicy(
      CompressionOptions compressionOptions,
      EncodingType encodingType);

  CompressionConfig config() const override;

  virtual bool shouldAccept(
      CompressionType compressionType,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override;

 private:
  float getAcceptRatio(EncodingType encodingType) const;

  const CompressionOptions compressionOptions_;
  const float effectiveAcceptRatio_;
};

/// Compression policy for replaying an encoding layout's captured compression
/// type while using current CompressionOptions parameters.
class ReplayedCompressionPolicy : public CompressionPolicy {
 public:
  ReplayedCompressionPolicy(
      CompressionType compressionType,
      CompressionOptions compressionOptions);

  CompressionConfig config() const override;

  virtual bool shouldAccept(
      CompressionType compressionType,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override;

 private:
  const CompressionType compressionType_;
  const CompressionOptions compressionOptions_;
};

} // namespace facebook::nimble
