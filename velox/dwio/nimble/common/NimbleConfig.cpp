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
#include <folly/json/dynamic.h>
#include <folly/json/json.h>

#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/common/NimbleConfig.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

DEFINE_string(
    nimble_selection_read_factors,
    "Constant=1.0;Trivial=0.5;FixedBitWidth=0.8;MainlyConstant=1.0;SparseBool=1.0;Dictionary=1.0;RLE=1.0",
    "Encoding selection read factors, in the format: "
    "<EncodingName>=<FactorFloatValue>;<EncodingName>=<FactorFloatValue>;...");

DEFINE_double(
    nimble_selection_compression_accept_ratio,
    0.99,
    "Encoding selection compression accept ratio.");

DEFINE_bool(
    nimble_zstrong_enable_variable_bit_width_compressor,
    false,
    "Enable zstrong variable bit width compressor at write time. Transparent at read time.");

DEFINE_string(
    nimble_writer_input_buffer_default_growth_config,
    "{\"32\":4.0,\"512\":1.414,\"4096\":1.189}",
    "Default growth config for writer input buffers, each entry in the format of {range_start,growth_factor}");

DEFINE_string(
    nimble_writer_string_buffer_growth_config,
    "{\"32000\":4.0,\"2000000\":2.5,\"4000000\":1.5}",
    "Growth config for string buffers when disableSharedStringBuffers is enabled. "
    "Format: {range_start_bytes:growth_factor}. Ranges: 32KB->4X, 2MB->2.5X, 4MB->1.5X");

namespace facebook::nimble {
namespace {
template <typename T>
std::vector<T> parseVector(const std::string& str) {
  std::vector<T> result;
  if (!str.empty()) {
    std::vector<folly::StringPiece> pieces;
    folly::split(',', str, pieces, true);
    for (auto& p : pieces) {
      const auto& trimmedCol = folly::trimWhitespace(p);
      if (!trimmedCol.empty()) {
        result.push_back(folly::to<T>(trimmedCol));
      }
    }
  }
  return result;
}

std::map<uint64_t, float> parseGrowthConfigMap(const std::string& str) {
  std::map<uint64_t, float> ret;
  NIMBLE_CHECK(!str.empty(), "Can't supply an empty growth config.");
  folly::dynamic json = folly::parseJson(str);
  for (const auto& pair : json.items()) {
    auto [_, inserted] = ret.emplace(
        folly::to<uint64_t>(pair.first.asString()), pair.second.asDouble());
    NIMBLE_CHECK(
        inserted, fmt::format("Duplicate key: {}.", pair.first.asString()));
  }
  return ret;
}
} // namespace

/* static */ Config::Entry<bool> Config::FLATTEN_MAP("orc.flatten.map", false);

/* static */ Config::Entry<const std::vector<uint32_t>> Config::MAP_FLAT_COLS(
    "orc.map.flat.cols",
    {},
    [](const std::vector<uint32_t>& val) { return folly::join(",", val); },
    [](const std::string& /* key */, const std::string& val) {
      return parseVector<uint32_t>(val);
    });

// Maximum number of distinct flat-map keys allowed per file; 0 means unlimited.
/* static */ Config::Entry<uint32_t> Config::MAP_FLAT_MAX_KEYS(
    "orc.map.flat.max.keys",
    kDefaultMaxFlatMapKeys);

/* static */ Config::Entry<const std::vector<uint32_t>>
    Config::DEDUPLICATED_COLS(
        "alpha.map.deduplicated.cols",
        {},
        [](const std::vector<uint32_t>& val) { return folly::join(",", val); },
        [](const std::string& /* key */, const std::string& val) {
          return parseVector<uint32_t>(val);
        });

/* static */ Config::Entry<const std::vector<uint32_t>>
    Config::BATCH_REUSE_COLS(
        "alpha.dictionaryarray.cols",
        {},
        [](const std::vector<uint32_t>& val) { return folly::join(",", val); },
        [](const std::string& /* key */, const std::string& val) {
          return parseVector<uint32_t>(val);
        });

/// 384MB is a relatively good safe default, some tables might OOM during
/// ingestion with higher stripe sizes.
/// One downside of smaller stripe sizes is that tables with good compression
/// ratio might get many small stripes resulting in higher IO. Second downside
/// is that smaller stripes migh have worse dictionary utilization resulting in
/// increased file size.
/* static */ Config::Entry<uint64_t> Config::RAW_STRIPE_SIZE(
    "alpha.raw.stripe.size",
    384 * 1024L * 1024L);

// currently sets the encoding executor to folly::globalCPUexecutor
/* static */ Config::Entry<bool> Config::ENABLE_DEFAULT_ENCODING_EXECUTOR(
    "alpha.encodingexecutor.enable.default",
    false);

/* static */ Config::Entry<bool> Config::ENABLE_STREAM_DEDUPLICATION(
    "nimble.enable.stream.deduplication",
    false);

/* static */ Config::Entry<bool> Config::DISABLE_SHARED_STRING_BUFFERS(
    "nimble.disable.shared.string.buffers",
    false);

/* static */ Config::Entry<const std::vector<std::pair<EncodingType, float>>>
    Config::MANUAL_ENCODING_SELECTION_READ_FACTORS(
        "alpha.encodingselection.read.factors",
        ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
            FLAGS_nimble_selection_read_factors),
        [](const std::vector<std::pair<EncodingType, float>>& readFactors) {
          std::vector<std::string> encodingFactorStrings;
          std::transform(
              readFactors.cbegin(),
              readFactors.cend(),
              std::back_inserter(encodingFactorStrings),
              [](const auto& readFactor) {
                return fmt::format(
                    "{}={}", toString(readFactor.first), readFactor.second);
              });
          return folly::join(";", encodingFactorStrings);
        },
        [](const std::string& /* key */, const std::string& readFactorsConfig) {
          return ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
              readFactorsConfig);
        });

/* static */ Config::Entry<float>
    Config::ENCODING_SELECTION_COMPRESSION_ACCEPT_RATIO(
        "alpha.encodingselection.compression.accept.ratio",
        FLAGS_nimble_selection_compression_accept_ratio);

/* static */ Config::Entry<const std::vector<std::pair<EncodingType, float>>>
    Config::COMPRESSION_ACCEPT_RATIO_OVERRIDES(
        "alpha.encodingselection.compression.accept.ratio.overrides",
        {},
        [](const std::vector<std::pair<EncodingType, float>>& readFactors) {
          std::vector<std::string> parts;
          std::transform(
              readFactors.cbegin(),
              readFactors.cend(),
              std::back_inserter(parts),
              [](const auto& p) {
                return fmt::format("{}={}", toString(p.first), p.second);
              });
          return folly::join(";", parts);
        },
        [](const std::string& /* key */, const std::string& readFactorsConfig) {
          if (readFactorsConfig.empty()) {
            return std::vector<std::pair<EncodingType, float>>{};
          }
          return ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
              readFactorsConfig);
        });

// Attempt to compress the data only if the data size is equal or greater than
// this threshold.
/* static */ Config::Entry<uint64_t> Config::ZSTD_COMPRESSION_MIN_SIZE(
    "alpha.zstd.compression.size.min",
    kZstdMinCompressionSize);

// Attempt to compress the data only if the data size is equal or greater than
// this threshold.
/* static */ Config::Entry<uint64_t> Config::ZSTRONG_COMPRESSION_MIN_SIZE(
    "alpha.zstrong.compression.size.min",
    kMetaInternalMinCompressionSize);

/* static */ Config::Entry<uint32_t> Config::ZSTRONG_COMPRESSION_LEVEL(
    "alpha.zstrong.compression.level",
    4);

/* static */ Config::Entry<uint32_t> Config::ZSTRONG_DECOMPRESSION_LEVEL(
    "alpha.zstrong.decompression.level",
    2);

/* static */ Config::Entry<bool>
    Config::ENABLE_ZSTRONG_VARIABLE_BITWIDTH_COMPRESSOR(
        "alpha.zstrong.enable.variable.bit.width.compressor",
        FLAGS_nimble_zstrong_enable_variable_bit_width_compressor);

/* static */ Config::Entry<bool> Config::USE_MANAGED_COMPRESSION(
    "alpha.openzl.use.managed.compression",
    false);

/* static */ Config::Entry<MetaInternalCompressionKey>
    Config::MANAGED_COMPRESSION_KEY(
        "alpha.openzl.managed.compression.key",
        MetaInternalCompressionKey{"", "", ""},
        [](const MetaInternalCompressionKey& key) { return key.toString(); },
        [](const std::string& /* key */, const std::string& val) {
          return MetaInternalCompressionKey::fromString(val);
        });

/* static */ Config::Entry<const std::map<uint64_t, float>>
    Config::INPUT_BUFFER_DEFAULT_GROWTH_CONFIGS(
        "alpha.writer.input.buffer.default.growth.configs",
        parseGrowthConfigMap(
            FLAGS_nimble_writer_input_buffer_default_growth_config),
        [](const std::map<uint64_t, float>& val) {
          folly::dynamic obj = folly::dynamic::object;
          for (const auto& [rangeStart, growthFactor] : val) {
            obj[folly::to<std::string>(rangeStart)] = growthFactor;
          }
          return folly::toJson(obj);
        },
        [](const std::string& /* key */, const std::string& val) {
          return parseGrowthConfigMap(val);
        });

/* static */ Config::Entry<const std::map<uint64_t, float>>
    Config::STRING_BUFFER_GROWTH_CONFIGS(
        "alpha.writer.string.buffer.growth.configs",
        parseGrowthConfigMap(FLAGS_nimble_writer_string_buffer_growth_config),
        [](const std::map<uint64_t, float>& val) {
          folly::dynamic obj = folly::dynamic::object;
          for (const auto& [rangeStart, growthFactor] : val) {
            obj[folly::to<std::string>(rangeStart)] = growthFactor;
          }
          return folly::toJson(obj);
        },
        [](const std::string& /* key */, const std::string& val) {
          return parseGrowthConfigMap(val);
        });

/// Enable chunking to manage writer and reader memory by splitting large data
/// streams into smaller chunks. When enabled, streams are encoded incrementally
/// instead of as single large blobs, reducing peak memory usage and OOMs.
/* static */ Config::Entry<bool> Config::ENABLE_CHUNKING(
    "nimble.chunking.enabled",
    true);

/// Enable chunk index for chunk-level seeking and per-chunk statistics
/// for filter pushdown. When enabled, the chunk stats optional section is
/// written alongside chunk position data in the file.
// EXPERIMENTAL: Not production-ready. Do not enable for production tables
// without consulting the Nimble team (oncall: dwios).
/* static */ Config::Entry<bool> Config::ENABLE_CHUNK_INDEX(
    "nimble.chunk.index.enabled",
    false);

/// Threshold to trigger chunking to relieve memory pressure.
/* static */ Config::Entry<uint64_t>
    Config::CHUNKING_WRITER_MEMORY_HIGH_THRESHOLD(
        "nimble.chunking.writer.memory.high.threshold",
        kChunkingWriterMemoryHighThreshold);

/// Threshold below which chunking stops and stripe size optimization resumes.
/* static */ Config::Entry<uint64_t>
    Config::CHUNKING_WRITER_MEMORY_LOW_THRESHOLD(
        "nimble.chunking.writer.memory.low.threshold",
        kChunkingWriterMemoryLowThreshold);

/// Target physical size for encoded stripes.
/* static */ Config::Entry<uint64_t>
    Config::CHUNKING_WRITER_TARGET_STRIPE_STORAGE_SIZE(
        "nimble.chunking.writer.target.stripe.storage.size",
        kChunkingWriterTargetStripeStorageSize);

/// Expected ratio of raw to encoded data.
/* static */ Config::Entry<double>
    Config::CHUNKING_WRITER_ESTIMATED_COMPRESSION_FACTOR(
        "nimble.chunking.writer.estimated.compression.factor",
        kChunkingWriterEstimatedCompressionFactor);

/// When flushing data streams into chunks, streams with raw data size smaller
/// than this threshold will not be flushed.
/// Note: this threshold is ignored when it is time to flush a stripe.
/* static */ Config::Entry<uint64_t> Config::CHUNKING_WRITER_MIN_CHUNK_SIZE(
    "nimble.chunking.writer.min.chunk.size",
    kChunkingWriterMinChunkSize);

/// When flushing data streams into chunks, streams with raw data size larger
/// than this threshold will be broken down into multiple smaller chunks. Each
/// chunk will be at most this size.
/* static */ Config::Entry<uint64_t> Config::CHUNKING_WRITER_MAX_CHUNK_SIZE(
    "nimble.chunking.writer.max.chunk.size",
    kChunkingWriterMaxChunkSize);

/// Used in place of CHUNKING_WRITER_MAX_CHUNK_SIZE for large schema tables.
/* static */ Config::Entry<uint64_t>
    Config::CHUNKING_WRITER_WIDE_SCHEMA_MAX_CHUNK_SIZE(
        "nimble.chunking.writer.wide.schema.max.chunk.size",
        kChunkingWriterWideSchemaMaxChunkSize);

/* static */ Config::Entry<std::string> Config::FLUSH_POLICY_CONFIG(
    "nimble.flush_policy_config",
    "");

/* static */ Config::Entry<std::string> Config::BUFFER_POLICY_CONFIG(
    "nimble.buffer_policy_config",
    "");

/* static */ Config::Entry<std::string> Config::ENCODING_SELECTION_CONFIG(
    "nimble.encoding_selection_config",
    "");

// EXPERIMENTAL: Cluster index is not production-ready. Do not enable for
// production tables without consulting the Nimble team (oncall: dwios).
/* static */ Config::Entry<const std::vector<std::string>>
    Config::INDEX_COLUMNS(
        "nimble.index.columns",
        {},
        [](const std::vector<std::string>& val) {
          return folly::join(",", val);
        },
        [](const std::string& /* key */, const std::string& val) {
          return parseVector<std::string>(val);
        });

/* static */ Config::Entry<const std::vector<std::string>>
    Config::INDEX_SORT_ORDERS(
        "nimble.index.sort_orders",
        {},
        [](const std::vector<std::string>& val) {
          return folly::join(",", val);
        },
        [](const std::string& /* key */, const std::string& val) {
          return parseVector<std::string>(val);
        });

/* static */ Config::Entry<bool> Config::INDEX_ENFORCE_KEY_ORDER(
    "nimble.index.enforce_key_order",
    false);

/* static */ Config::Entry<bool> Config::INDEX_NO_DUPLICATE_KEY(
    "nimble.index.no_duplicate_key",
    false);

/* static */ Config::Entry<std::string> Config::INDEX_ENCODING_TYPE(
    "nimble.index.encoding_type",
    "prefix");

/* static */ Config::Entry<uint64_t> Config::INDEX_MAX_ROWS_PER_KEY_CHUNK(
    "nimble.index.max_rows_per_key_chunk",
    10'000);

/* static */ Config::Entry<bool> Config::INDEX_OMIT_KEY_COLUMN_STORAGE(
    "nimble.index.omit_key_column_storage",
    false);

// EXPERIMENTAL: BlockBitPacking encoding is not production-ready. Do not
// enable for production tables without consulting the Nimble team
// (oncall: dwios).
/* static */ Config::Entry<uint16_t> Config::BLOCK_BIT_PACKING_BLOCK_SIZE(
    "nimble.blockbitpacking.block.size",
    kBlockBitPackingBlockSize);

/* static */ Config::Entry<bool> Config::ENABLE_STATS_COLLECTION(
    "nimble.stats.enable",
    true);

/* static */ Config::Entry<bool> Config::ENABLE_VECTORIZED_STATS(
    "nimble.stats.enable_vectorized",
    true);

/* static */ Config::Entry<bool> Config::ENABLE_ENCODING_SELECTION_CACHE(
    "nimble.encoding.enable_selection_cache",
    false);
} // namespace facebook::nimble
