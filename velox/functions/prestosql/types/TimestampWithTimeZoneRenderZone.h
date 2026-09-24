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

#include <string>

#include "velox/functions/lib/TimeUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox {

/// Selects the zone used to render a packed TIMESTAMP WITH TIME ZONE.
class TimestampWithTimeZoneRenderZone {
 public:
  /// Controls when a non-legacy session zone is validated and resolved.
  enum class SessionZoneResolution { kEager, kOnDemand };

  /// Captures the query's render-zone policy.
  explicit TimestampWithTimeZoneRenderZone(const core::QueryConfig& config)
      : TimestampWithTimeZoneRenderZone(config, SessionZoneResolution::kEager) {
  }

  /// Captures the policy with configurable session-zone resolution.
  TimestampWithTimeZoneRenderZone(
      const core::QueryConfig& config,
      SessionZoneResolution resolution)
      : legacyTimestampWithTimeZone_(config.legacyTimestampWithTimezone()),
        sessionTimeZoneName_(config.sessionTimezone()),
        sessionRenderZone_(
            !legacyTimestampWithTimeZone_ &&
                    resolution == SessionZoneResolution::kEager
                ? functions::getSessionTimeZone(sessionTimeZoneName_)
                : nullptr) {}

  /// Returns the value's embedded zone in legacy mode and the session zone
  /// otherwise.
  const tz::TimeZone* get(int64_t timestampWithTimeZone) const {
    if (legacyTimestampWithTimeZone_) {
      return tz::locateZone(unpackZoneKeyId(timestampWithTimeZone));
    }
    if (sessionRenderZone_ == nullptr) {
      sessionRenderZone_ = functions::getSessionTimeZone(sessionTimeZoneName_);
    }
    return sessionRenderZone_;
  }

 private:
  // Preserves embedded-zone rendering when true.
  const bool legacyTimestampWithTimeZone_;
  // Retains the configured name for on-demand resolution.
  const std::string sessionTimeZoneName_;
  // Caches the first successful eager or on-demand session-zone resolution.
  mutable const tz::TimeZone* sessionRenderZone_;
};

} // namespace facebook::velox
