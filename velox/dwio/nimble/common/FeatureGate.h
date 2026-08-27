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

#include <memory>
#include <string_view>

namespace facebook::nimble {

/// Runtime enablement gate for optional writer features (e.g. rollout
/// killswitches). The base implementation applies no runtime override:
/// enabled() returns the caller-provided default, so OSS builds honor the
/// static writer config alone. Internal builds install a dynamic-config-backed
/// implementation via registerFeatureGate() to gate gradual rollouts.
class FeatureGate {
 public:
  /// Stable identifiers for the runtime-gated writer features. OSS-neutral: no
  /// dynamic-config (e.g. JustKnobs) names leak into open source. The internal
  /// FeatureGate implementation maps each identifier to its backing knob.
  class FeatureSet {
   public:
    static constexpr std::string_view kChunkedEncoding = "chunked_encoding";
    static constexpr std::string_view kStreamDeduplication =
        "stream_deduplication";
    static constexpr std::string_view kDisableSharedStringBuffers =
        "disable_shared_string_buffers";
  };

  virtual ~FeatureGate() = default;

  /// Resolves whether `feature` is enabled at runtime, given the caller's
  /// requested `defaultValue`. With no gate installed (the OSS default),
  /// returns `defaultValue` unchanged. A registered gate may consult dynamic
  /// config to hold a feature back during rollout (return false) or force it on
  /// (return true), independent of `defaultValue`.
  virtual bool enabled(std::string_view feature, bool defaultValue) const {
    return defaultValue;
  }
};

/// Installs the process-wide FeatureGate, replacing any previously registered
/// one. Intended to be called once at startup by internal (non-OSS) code with a
/// dynamic-config-backed gate. Passing nullptr restores the default no-op gate.
/// Thread-safe.
void registerFeatureGate(std::shared_ptr<FeatureGate> gate);

/// Returns the process-wide FeatureGate: the one installed via
/// registerFeatureGate(), or a default no-op gate when none has been registered
/// (the OSS case). The returned pointer is never null and owns the gate, so it
/// stays valid even if a different gate is registered concurrently.
/// Thread-safe.
std::shared_ptr<const FeatureGate> featureGate();

} // namespace facebook::nimble
