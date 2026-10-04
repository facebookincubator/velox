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

#include <concepts>
#include <cstdint>
#include <optional>
#include <string>

#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::aggregate::sparksql {

/// Converts histogram_numeric inputs and stored centers independently of SQL
/// cast/ANSI settings. Callers skip null inputs; heights always remain DOUBLE.
/// Legacy output uses the stored center unchanged, not a propagated conversion.
class HistogramNumericTypes {
 public:
  /// Normalizes primitive numeric storage. DATE and INTERVAL_YEAR_MONTH use
  /// int32 days and months.
  template <typename T>
    requires(
        std::same_as<T, int8_t> || std::same_as<T, int16_t> ||
        std::same_as<T, int32_t> || std::same_as<T, int64_t> ||
        std::same_as<T, float> || std::same_as<T, double>)
  static double numericToDouble(T value) {
    return static_cast<double>(value);
  }

  /// Normalizes TIMESTAMP or TIMESTAMP_UTC as signed microseconds without a
  /// timezone adjustment. Rejects sub-microsecond nanos and int64 overflow.
  static double timestampToDouble(const Timestamp& value);

  /// Applies Java double-to-int followed by signed low-eight-bit narrowing.
  static int8_t javaDoubleToByte(double value);

  /// Applies Java double-to-int followed by signed low-sixteen-bit narrowing.
  static int16_t javaDoubleToShort(double value);

  /// Truncates toward zero, maps NaN to zero, and saturates at signed limits.
  /// Also reconstructs DATE epoch days and INTERVAL_YEAR_MONTH total months.
  static int32_t javaDoubleToInt(double value);

  /// Truncates toward zero, maps NaN to zero, and saturates at signed limits.
  /// Also reconstructs BIGINT centers.
  static int64_t javaDoubleToLong(double value);

  /// Rounds to binary32, ties to even, including subnormals and overflow to
  /// infinity, independently of the host FP rounding mode. Quiets NaNs while
  /// keeping sign/high payload bits; Java only guarantees NaN classification.
  static float javaDoubleToFloat(double value);

  /// Reconstructs TIMESTAMP and TIMESTAMP_UTC via Java-toLong microseconds,
  /// including INT64_MIN, without overflowing negative-epoch intermediates.
  static Timestamp javaDoubleToTimestamp(double value);
};

/// Binds Decimal metadata once for typed and legacy normalization and output.
/// Pins textual selection to the Java 21 Double.toString specification and
/// Scala 2.13.17 BigDecimal behavior used by the target Spark implementation.
class HistogramNumericDecimal {
 public:
  /// Rejects precision outside [1,38] and scale outside [0,precision], before
  /// narrowing metadata to the native Decimal type representation.
  HistogramNumericDecimal(int32_t precision, int32_t scale);

  /// Converts native unscaled storage with one correctly rounded division when
  /// both operands are exact doubles (magnitude <= 2^53 and scale <= 22).
  /// Otherwise uses exact DecimalUtil text and correctly rounded parsing to
  /// avoid double rounding.
  double decimalToDouble(int128_t unscaled) const;

  /// Returns the fully reconstructed fixed-scale value after MathContext(p)
  /// HALF_UP rounding followed by setScale(s, HALF_UP). Rejects non-finite
  /// centers. Normalizes negative zero and retains values wider than p.
  std::string reconstructDecimal(double center) const;

  /// Returns the same reconstructed value as unscaled native Decimal storage,
  /// or nullopt when it does not fit p and Spark materializes a null field.
  /// Rejects non-finite centers. Results with p <= 18 fit int64 and may be
  /// narrowed by the typed writer.
  std::optional<int128_t> tryReconstructNativeDecimal(double center) const;

 private:
  // Retains validated metadata for both independent Spark rounding stages.
  int32_t precision_;
  int32_t scale_;
  // Reuses native Decimal text conversion for both physical storage widths.
  TypePtr type_;
};

} // namespace facebook::velox::functions::aggregate::sparksql
