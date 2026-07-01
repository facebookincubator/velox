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

#pragma once

#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/Compression.h"
#include "velox/common/config/Config.h"
#include "velox/dwio/common/FormatConfig.h"
#include "velox/dwio/dwrf/common/Common.h"

namespace facebook::velox::dwrf {

class Config : public config::ConfigBase {
 public:
  template <typename T>
  using Entry = config::ConfigBase::Entry<T>;

  static Entry<WriterVersion> WRITER_VERSION;
  static Entry<common::CompressionKind> COMPRESSION;
  static Entry<int32_t> ZLIB_COMPRESSION_LEVEL;
  static Entry<int32_t> ZSTD_COMPRESSION_LEVEL;
  static Entry<uint64_t> COMPRESSION_BLOCK_SIZE;
  static Entry<uint64_t> COMPRESSION_BLOCK_SIZE_MIN;
  static Entry<float> COMPRESSION_BLOCK_SIZE_EXTEND_RATIO;
  static Entry<uint32_t> COMPRESSION_THRESHOLD;
  static Entry<bool> CREATE_INDEX;
  static Entry<uint32_t> ROW_INDEX_STRIDE;
  static Entry<proto::ChecksumAlgorithm> CHECKSUM_ALGORITHM;
  static Entry<StripeCacheMode> STRIPE_CACHE_MODE;
  static Entry<uint32_t> STRIPE_CACHE_SIZE;
  static Entry<uint32_t> DICTIONARY_ENCODING_INTERVAL;
  static Entry<bool> USE_VINTS;
  static Entry<float> DICTIONARY_NUMERIC_KEY_SIZE_THRESHOLD;
  static Entry<float> DICTIONARY_STRING_KEY_SIZE_THRESHOLD;
  static Entry<bool> DICTIONARY_SORT_KEYS;
  static Entry<float> ENTROPY_KEY_STRING_SIZE_THRESHOLD;
  static Entry<uint32_t> ENTROPY_STRING_MIN_SAMPLES;
  static Entry<float> ENTROPY_STRING_DICT_SAMPLE_FRACTION;
  static Entry<uint32_t> ENTROPY_STRING_THRESHOLD;
  static Entry<uint32_t> STRING_STATS_LIMIT;
  static Entry<bool> FLATTEN_MAP;
  static Entry<bool> MAP_FLAT_DISABLE_DICT_ENCODING;
  static Entry<bool> MAP_FLAT_DISABLE_DICT_ENCODING_STRING;
  static Entry<bool> MAP_FLAT_DICT_SHARE;
  static Entry<const std::vector<uint32_t>> MAP_FLAT_COLS;
  static Entry<const std::vector<std::vector<std::string>>>
      MAP_FLAT_COLS_STRUCT_KEYS;
  static Entry<uint32_t> MAP_FLAT_MAX_KEYS;
  static Entry<uint64_t> MAX_DICTIONARY_SIZE;
  static Entry<bool> INTEGER_DICTIONARY_ENCODING_ENABLED;
  static Entry<bool> STRING_DICTIONARY_ENCODING_ENABLED;
  static Entry<uint64_t> STRIPE_SIZE;
  static Entry<bool> LINEAR_STRIPE_SIZE_HEURISTICS;
  /// With this config, we don't even try the more memory intensive encodings on
  /// writer start up.
  static Entry<bool> FORCE_LOW_MEMORY_MODE;
  /// Disable low memory mode mostly for test purposes.
  static Entry<bool> DISABLE_LOW_MEMORY_MODE;
  /// Fail the writer, when Stream size is above threshold. Streams greater than
  /// 2GB will be failed to be read by Jolly/Presto reader.
  static Entry<bool> STREAM_SIZE_ABOVE_THRESHOLD_CHECK_ENABLED;
  /// Limit the raw data size per batch to avoid being forced to write oversized
  /// stripes.
  static Entry<uint64_t> RAW_DATA_SIZE_PER_BATCH;
  static Entry<bool> MAP_STATISTICS;

  VELOX_FORMAT_CONFIG_PROPERTY(
      kOrcMaxCoalesceDistanceSession,
      kOrcMaxCoalesceDistance,
      "max_coalesced_distance",
      "max-coalesced-distance",
      std::string_view,
      "512kB",
      "Maximum merge distance to combine ORC read requests.")

  VELOX_FORMAT_CONFIG_PROPERTY(
      kOrcWriterMaxStripeSizeSession,
      kOrcWriterMaxStripeSize,
      "optimized_writer_max_stripe_size",
      "writer.stripe-max-size",
      std::string_view,
      "64MB",
      "Maximum stripe size in orc writer.")

  VELOX_FORMAT_CONFIG_PROPERTY(
      kOrcWriterMaxDictionaryMemorySession,
      kOrcWriterMaxDictionaryMemory,
      "optimized_writer_max_dictionary_memory",
      "writer.dictionary-max-memory",
      std::string_view,
      "16MB",
      "Maximum dictionary memory that can be used in orc writer.")

  VELOX_FORMAT_CONFIG_PROPERTY(
      kOrcWriterIntegerDictionaryEncodingEnabledSession,
      kOrcWriterIntegerDictionaryEncodingEnabled,
      "optimized_writer_integer_dictionary_encoding_enabled",
      "writer.integer-dictionary-encoding-enabled",
      bool,
      true,
      "Whether or not dictionary encoding of integer types should be used "
      "by the ORC writer.")

  VELOX_FORMAT_CONFIG_PROPERTY(
      kOrcWriterStringDictionaryEncodingEnabledSession,
      kOrcWriterStringDictionaryEncodingEnabled,
      "optimized_writer_string_dictionary_encoding_enabled",
      "writer.string-dictionary-encoding-enabled",
      bool,
      true,
      "Whether or not dictionary encoding of string types should be used "
      "by the ORC writer.")

  VELOX_FORMAT_CONFIG_PROPERTY(
      kOrcWriterLinearStripeSizeHeuristicsSession,
      kOrcWriterLinearStripeSizeHeuristics,
      "writer_linear_stripe_size_heuristics",
      "writer.linear-stripe-size-heuristics",
      bool,
      true,
      "Enables historical based stripe size estimation after compression.")

  VELOX_FORMAT_CONFIG_PROPERTY(
      kOrcWriterMinCompressionSizeSession,
      kOrcWriterMinCompressionSize,
      "writer_min_compression_size",
      "writer.min-compression-size",
      uint64_t,
      1024,
      "Minimal number of items in an encoded stream.")

  /// The compression level to use with ZLIB and ZSTD.
  static constexpr const char* kOrcWriterCompressionLevel =
      "writer.compression-level";
  static constexpr const char* kOrcWriterCompressionLevelSession =
      "optimized_writer_compression_level";

  static int32_t maxCoalesceDistance(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) {
    const auto distance = config::toCapacity(
        session
            .getLegacyWithFallback<std::string>(
                kOrcMaxCoalesceDistanceSession,
                connectorConfig,
                kOrcMaxCoalesceDistance)
            .value_or(kOrcMaxCoalesceDistanceSessionProperty::defaultValue),
        config::CapacityUnit::BYTE);
    VELOX_USER_CHECK_LE(
        distance,
        std::numeric_limits<int32_t>::max(),
        "The max merge distance to combine read requests must be less than 2GB."
        " Got {} bytes.",
        distance);
    return int32_t(distance);
  }

  static void registerProperties(
      std::vector<config::ConfigProperty>& properties,
      std::string_view sessionPrefix) {
    dwio::common::registerFormatConfigProperty<
        kOrcMaxCoalesceDistanceSessionProperty>(properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kOrcWriterMaxStripeSizeSessionProperty>(properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kOrcWriterMaxDictionaryMemorySessionProperty>(
        properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kOrcWriterIntegerDictionaryEncodingEnabledSessionProperty>(
        properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kOrcWriterStringDictionaryEncodingEnabledSessionProperty>(
        properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kOrcWriterLinearStripeSizeHeuristicsSessionProperty>(
        properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kOrcWriterMinCompressionSizeSessionProperty>(properties, sessionPrefix);
    properties.push_back({
        std::string(sessionPrefix) + kOrcWriterCompressionLevelSession,
        config::ConfigPropertyType::kInteger,
        std::nullopt,
        "The compression level to use with ZLIB and ZSTD.",
    });
  }

  static std::shared_ptr<Config> fromMap(
      const std::map<std::string, std::string>& map) {
    auto config = std::make_shared<Config>();
    for (const auto& pair : map) {
      config->set(pair.first, pair.second);
    }
    return config;
  }

  Config() : ConfigBase({}, true) {}

  std::map<std::string, std::string> toSerdeParams() const {
    return std::map{configs_.cbegin(), configs_.cend()};
  }
};

} // namespace facebook::velox::dwrf
