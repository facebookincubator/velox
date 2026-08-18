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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"

namespace facebook::nimble::benchmarks {

/// Caps serialized input before it reaches the encoding factory.
inline constexpr size_t kMaxEncodingArtifactBytes = 64 * 1024 * 1024;

/// Configures one deterministic runner invocation.
struct EncodingRunnerConfig {
  /// Selects the encoding and operation lane.
  std::string taskId;
  /// Sets the number of logical rows in the deterministic corpus.
  uint32_t rowCount{4096};
  /// Seeds corpus and access-pattern generation.
  uint64_t seed{0xC0FFEE};
  /// Sets the number of untimed samples before measurement.
  uint32_t warmups{3};
  /// Sets the number of raw timing observations.
  uint32_t samples{15};
  /// Sets the minimum calibrated duration for each observation.
  uint32_t minSampleTimeMicros{250'000};
  /// Sets the initial operations per observation before calibration.
  uint32_t innerIterations{1};
};

/// Carries one implementation's validated artifact and raw timing samples.
struct EncodingRunnerMeasurement {
  /// Identifies the manifest task.
  std::string taskId;
  /// Identifies the root encoding.
  std::string encoding;
  /// Identifies the measured operation.
  std::string lane;
  /// Identifies the representative physical type.
  std::string dataType;
  /// Records the corpus seed.
  uint64_t seed{0};
  /// Records the logical row count.
  uint32_t rowCount{0};
  /// Records the serialized artifact size.
  uint32_t encodedBytes{0};
  /// Fingerprints the canonical input semantics.
  std::string inputDigest;
  /// Fingerprints the decoded output semantics.
  std::string outputDigest;
  /// Fingerprints the serialized artifact bytes.
  std::string artifactDigest;
  /// Stores seconds per measured operation.
  std::vector<double> samplesSeconds;
  /// Owns the artifact referenced by the measurement.
  std::string encodedArtifact;
};

/// Carries peer-artifact compatibility facts from one runner binary.
struct EncodingArtifactVerification {
  /// Identifies the manifest task.
  std::string taskId;
  /// Identifies the root encoding.
  std::string encoding;
  /// Identifies the representative physical type.
  std::string dataType;
  /// Records the corpus seed.
  uint64_t seed{0};
  /// Records the logical row count.
  uint32_t rowCount{0};
  /// Records the verified artifact size.
  uint32_t encodedBytes{0};
  /// Fingerprints the canonical input semantics.
  std::string inputDigest;
  /// Fingerprints the decoded output semantics.
  std::string outputDigest;
  /// Fingerprints the verified artifact bytes.
  std::string artifactDigest;
};

/// Runs one operation and optionally consumes a pristine canonical artifact.
EncodingRunnerMeasurement runEncodingBenchmark(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool,
    std::optional<std::string_view> benchmarkArtifact = std::nullopt);

/// Validates a peer artifact against the deterministic task corpus.
EncodingArtifactVerification verifyEncodingArtifact(
    const EncodingRunnerConfig& config,
    std::string_view encodedArtifact,
    velox::memory::MemoryPool& pool);

/// Serializes a validated raw measurement as schema-version-1 JSON.
std::string measurementToJson(const EncodingRunnerMeasurement& measurement);

/// Serializes a validated peer verification as schema-version-1 JSON.
std::string verificationToJson(
    const EncodingArtifactVerification& verification);

} // namespace facebook::nimble::benchmarks
