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
#include <optional>
#include <string>
#include <unordered_map>

#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {
struct WriterOptions;
}

namespace facebook::nimble::fuzzer {

/// Flush regimes exercised by the shared writer configuration randomizer.
enum class FuzzedFlushMode {
  kRawSize,
  kPerBatch,
  kRandom,
};

/// Controls random encoding-selection frequency for each fuzzer user.
struct WriterConfigRandomizationOptions {
  /// One-in-N probability of installing random encoding selection. Zero
  /// disables it; one always installs it.
  uint32_t randomEncodingSelectionOneIn{1};
};

/// Reproducible, serializable writer settings shared by Nimble fuzzers.
struct RandomizedWriterConfig {
  /// Distribution version. Increment when a seed maps to different settings.
  uint32_t version{2};

  /// Root seed from which domain-separated draws are derived.
  uint64_t seed{0};

  /// Maximum accepted compressed-to-uncompressed size ratio.
  float compressionAcceptRatio{1.0};

  /// Minimum sizes at which shared compression codecs are attempted.
  uint64_t zstdMinCompressionSize{0};
  uint64_t internalMinCompressionSize{0};

  /// MetaInternal compression level.
  uint32_t internalCompressionLevel{1};

  /// Chunking enablement and raw-size thresholds.
  bool enableChunking{true};
  uint64_t minStreamChunkRawSize{0};
  uint64_t maxStreamChunkRawSize{1'024};
  uint64_t wideSchemaMaxStreamChunkRawSize{1'024};

  /// Optional file-layout features.
  bool enableChunkIndex{false};
  bool enableStreamDeduplication{false};

  /// Encoding-selection tuning exposed through serde.
  uint16_t blockBitPackingBlockSize{32};

  /// Flush regime and its reproducible parameters.
  FuzzedFlushMode flushMode{FuzzedFlushMode::kRawSize};
  uint64_t stripeRawSize{4'096};
  uint64_t flushSeed{0};

  /// Seed for random encoding selection; absent keeps manual selection.
  std::optional<uint64_t> encodingSelectionSeed;

  bool operator==(const RandomizedWriterConfig&) const = default;

  /// Applies scalar and flush settings directly to the low-level writer
  /// options. Apply encoding selection after any direct-only compression
  /// overrides so the policy captures the final compression options.
  void applyTo(WriterOptions& options) const;

  /// Applies random encoding selection to the low-level writer options when
  /// enabled, optionally restricting the policy to one encoding.
  void applyEncodingSelectionTo(
      WriterOptions& options,
      std::optional<EncodingType> forcedEncoding = std::nullopt) const;

  /// Serializes this configuration for TableEvolution's serde parameter hook.
  std::unordered_map<std::string, std::string> toSerdeParams() const;

  /// Returns the random-policy string, optionally restricted to one encoding.
  std::string encodingSelectionConfig(
      std::optional<EncodingType> forcedEncoding = std::nullopt) const;

  /// Returns a stable, ordered representation suitable for failure logs.
  std::string debugString() const;
};

/// Generates a writer configuration solely from 'seed' and 'options'.
RandomizedWriterConfig randomizeWriterConfig(
    uint64_t seed,
    WriterConfigRandomizationOptions options = {});

} // namespace facebook::nimble::fuzzer
