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

#include <type_traits>

#include <folly/Expected.h>

#include "velox/common/base/Status.h"
#include "velox/expression/StringWriter.h"
#include "velox/type/CppToType.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::exec {

enum PolicyType {
  LegacyCastPolicy = 1,
  PrestoCastPolicy,
  SparkCastPolicy,
  SparkTryCastPolicy
};

fmt::underlying_t<PolicyType> format_as(PolicyType f);

/// This class provides cast hooks to allow different behaviors of CastExpr and
/// SparkCastExpr. The main purpose is to create customized cast implementation
/// by taking full usage of existing cast expression.
class CastHooks {
 public:
  virtual ~CastHooks() = default;

  virtual Expected<Timestamp> castStringToTimestamp(
      const StringView& view) const = 0;

  virtual Expected<Timestamp> castIntToTimestamp(int64_t seconds) const = 0;

  virtual Expected<int64_t> castTimestampToBigint(
      Timestamp timestamp) const = 0;

  /// Returns the converted value as int64_t and reports overflow against the
  /// target type 'To' as an error. setResultOrStatus then either throws (ANSI)
  /// or sets NULL (try_cast) for the overflow case.
  template <typename T>
  Expected<int64_t> castTimestampToInt(Timestamp timestamp) const {
    auto result = castTimestampToBigint(timestamp);
    if (result.hasError()) {
      return result;
    }

    const int64_t value = result.value();
    if (value != static_cast<int64_t>(static_cast<T>(value))) {
      return folly::makeUnexpected(
          Status::UserError(
              "The value {} of the type \"{}\" cannot be cast to \"{}\" due to an "
              "overflow. Use `try_cast` to tolerate overflow and return NULL "
              "instead.",
              value,
              TypeTraits<TypeKind::TIMESTAMP>::name,
              CppToType<T>::name));
    }
    return value;
  }

  virtual Expected<std::optional<Timestamp>> castDoubleToTimestamp(
      double seconds) const = 0;

  virtual Expected<int32_t> castStringToDate(
      const StringView& dateString) const = 0;

  /// 'data' is guaranteed to be non-empty and has been processed by
  /// removeWhiteSpaces.
  virtual Expected<float> castStringToReal(const StringView& data) const = 0;

  /// 'data' is guaranteed to be non-empty and has been processed by
  /// removeWhiteSpaces.
  virtual Expected<double> castStringToDouble(const StringView& data) const = 0;

  /// Trims all leading and trailing UTF8 whitespaces.
  virtual StringView removeWhiteSpaces(const StringView& view) const = 0;

  /// Returns the options to cast from timestamp to string.
  virtual const TimestampToStringOptions& timestampToStringOptions() const = 0;

  /// Returns the options to cast from TIMESTAMP_UTC to string, with
  /// timeZone = nullptr since TIMESTAMP_UTC is not subject to session
  /// timezone adjustment.
  virtual TimestampToStringOptions timestampUtcToStringOptions() const = 0;

  /// Returns whether to cast to int by truncate.
  virtual bool truncate() const = 0;

  /// Returns whether to apply try_cast recursively rather than only at the top
  /// level. E.g. if true, an element inside an array would be null rather than
  /// the entire array if the cast of that element fails.
  virtual bool applyTryCastRecursively() const = 0;

  virtual PolicyType getPolicy() const = 0;

  /// Returns true if TIMESTAMP_UTC casts are supported.
  /// Spark supports them; Presto does not.
  virtual bool supportsTimestampUtc() const = 0;

  /// Converts boolean to timestamp type.
  virtual Expected<Timestamp> castBooleanToTimestamp(bool seconds) const = 0;

  /// Returns whether to format small magnitude decimals using scientific
  /// notation. Example: with scale 20 and value 1, the output is
  /// "1E-20" when isScientific() is true, and "0.00000000000000000001" when
  /// false.
  virtual bool isScientific() const = 0;

  /// Converts a local timestamp to GMT using the given timezone.
  /// The default implementation calls timestamp.toGMT() which throws on
  /// nonexistent local times (timezone gaps). Spark overrides this to adjust
  /// gap times to the post-transition time instead of throwing.
  virtual void castDateTimestampToGMT(
      Timestamp& timestamp,
      const tz::TimeZone& timeZone) const {
    timestamp.toGMT(timeZone);
  }
};
} // namespace facebook::velox::exec
