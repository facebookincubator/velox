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
#include "velox/connectors/paimon/PaimonFieldAggregator.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <limits>

#include "velox/connectors/paimon/PaimonTableHandle.h"
#include "velox/functions/sparksql/JavaFloatingPoint.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::connector::paimon {
namespace {

using int256_t = boost::multiprecision::int256_t;

[[noreturn]] void failRetract(
    const std::string& function,
    const std::string& column) {
  VELOX_USER_FAIL(
      "Paimon aggregate function {} cannot retract a row unless '{}' is true",
      function,
      PaimonTableOptions::fieldIgnoreRetract(column));
}

// Compares two non-null values as Paimon Java's InternalRowUtils.compare does.
int32_t compareValues(
    const BaseVector& left,
    vector_size_t leftRow,
    const BaseVector& right,
    vector_size_t rightRow) {
  switch (left.typeKind()) {
    case TypeKind::REAL:
      return functions::sparksql::JavaFloatingPoint::compare(
          left.asUnchecked<SimpleVector<float>>()->valueAt(leftRow),
          right.asUnchecked<SimpleVector<float>>()->valueAt(rightRow));
    case TypeKind::DOUBLE:
      return functions::sparksql::JavaFloatingPoint::compare(
          left.asUnchecked<SimpleVector<double>>()->valueAt(leftRow),
          right.asUnchecked<SimpleVector<double>>()->valueAt(rightRow));
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::HUGEINT:
    case TypeKind::TIMESTAMP:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      // Integers and decimals compare by value, timestamps by time, and
      // strings and binaries by unsigned bytes, as in Java.
      return left.compare(&right, leftRow, rightRow, CompareFlags{}).value();
    case TypeKind::BOOLEAN:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::UNKNOWN:
    case TypeKind::FUNCTION:
    case TypeKind::OPAQUE:
    case TypeKind::INVALID:
      // isComparable() rejects these types.
      VELOX_UNREACHABLE();
    default:
      VELOX_FAIL("Unknown TypeKind: {}", static_cast<int>(left.typeKind()));
  }
}

// Base of the functions that pick an input value. Keeps a reference to the
// row of the value.
class PickAggregator : public PaimonFieldAggregator {
 public:
  PickAggregator(std::string function, std::string column)
      : function_(std::move(function)), column_(std::move(column)) {}

  void reset() override {
    clear();
  }

  void retract(const BaseVector& /*input*/, vector_size_t /*row*/) override {
    failRetract(function_, column_);
  }

  void set(const BaseVector& input, vector_size_t row) override {
    pick(input, row);
  }

  void write(vector_size_t row, BaseVector& output) const override {
    if (vector_ == nullptr) {
      output.setNull(row, true);
      return;
    }
    output.copy(vector_, row, row_, 1);
  }

 protected:
  // Vector of the accumulator, or nullptr if the accumulator is null.
  const BaseVector* accumulatorVector() const {
    return vector_;
  }

  vector_size_t accumulatorRow() const {
    return row_;
  }

  // Makes row 'row' of 'input' the accumulator. A null value clears it.
  void pick(const BaseVector& input, vector_size_t row) {
    if (input.isNullAt(row)) {
      clear();
      return;
    }
    vector_ = &input;
    row_ = row;
  }

  void clear() {
    vector_ = nullptr;
  }

  const std::string function_;
  const std::string column_;

 private:
  const BaseVector* vector_{nullptr};
  vector_size_t row_{0};
};

// FieldPrimaryKeyAgg: takes every value, retracted or not.
class PrimaryKeyAggregator final : public PickAggregator {
 public:
  using PickAggregator::PickAggregator;

  void add(const BaseVector& input, vector_size_t row) override {
    pick(input, row);
  }

  void retract(const BaseVector& input, vector_size_t row) override {
    pick(input, row);
  }
};

// FieldLastValueAgg: takes every value, and a retract clears it.
class LastValueAggregator final : public PickAggregator {
 public:
  using PickAggregator::PickAggregator;

  void add(const BaseVector& input, vector_size_t row) override {
    pick(input, row);
  }

  void retract(const BaseVector& /*input*/, vector_size_t /*row*/) override {
    clear();
  }
};

// FieldLastNonNullValueAgg: takes every non-null value, and retracting one
// clears it.
class LastNonNullValueAggregator final : public PickAggregator {
 public:
  using PickAggregator::PickAggregator;

  void add(const BaseVector& input, vector_size_t row) override {
    if (!input.isNullAt(row)) {
      pick(input, row);
    }
  }

  void retract(const BaseVector& input, vector_size_t row) override {
    if (!input.isNullAt(row)) {
      clear();
    }
  }
};

// FieldFirstValueAgg and FieldFirstNonNullValueAgg: take the first value, or
// the first non-null value, of a key.
class FirstValueAggregator final : public PickAggregator {
 public:
  FirstValueAggregator(std::string function, std::string column, bool nonNull)
      : PickAggregator(std::move(function), std::move(column)),
        nonNull_(nonNull) {}

  void reset() override {
    PickAggregator::reset();
    initialized_ = false;
  }

  void add(const BaseVector& input, vector_size_t row) override {
    if (initialized_ || (nonNull_ && input.isNullAt(row))) {
      return;
    }
    initialized_ = true;
    pick(input, row);
  }

 private:
  const bool nonNull_;
  // Whether the function took its value. A DELETE row that replaces the
  // merged row does not reset it.
  bool initialized_{false};
};

// FieldMinAgg and FieldMaxAgg. On ties, min takes the input and max keeps the
// accumulator.
class MinMaxAggregator final : public PickAggregator {
 public:
  MinMaxAggregator(std::string function, std::string column, bool min)
      : PickAggregator(std::move(function), std::move(column)), min_(min) {}

  void add(const BaseVector& input, vector_size_t row) override {
    if (input.isNullAt(row)) {
      return;
    }
    if (accumulatorVector() == nullptr) {
      pick(input, row);
      return;
    }
    const auto result =
        compareValues(*accumulatorVector(), accumulatorRow(), input, row);
    if (min_ ? result >= 0 : result < 0) {
      pick(input, row);
    }
  }

 private:
  const bool min_;
};

// Base of the functions that compute a value of type T.
template <typename T>
class ValueAggregator : public PaimonFieldAggregator {
 public:
  ValueAggregator(std::string function, std::string column)
      : function_(std::move(function)), column_(std::move(column)) {}

  void reset() override {
    value_.reset();
  }

  void retract(const BaseVector& /*input*/, vector_size_t /*row*/) override {
    failRetract(function_, column_);
  }

  void set(const BaseVector& input, vector_size_t row) override {
    value_ = read(input, row);
  }

  void write(vector_size_t row, BaseVector& output) const override {
    auto* flatOutput = output.asChecked<FlatVector<T>>();
    if (value_.has_value()) {
      flatOutput->set(row, *value_);
    } else {
      flatOutput->setNull(row, true);
    }
  }

 protected:
  static std::optional<T> read(const BaseVector& input, vector_size_t row) {
    if (input.isNullAt(row)) {
      return std::nullopt;
    }
    return input.asUnchecked<SimpleVector<T>>()->valueAt(row);
  }

  const std::string function_;
  const std::string column_;
  // Null when the accumulator is null.
  std::optional<T> value_;
};

// Unsigned type in which T, an integer type, wraps around as Java integers
// do. Java promotes byte and short to int.
template <typename T>
using JavaUnsigned =
    std::conditional_t<sizeof(T) <= sizeof(uint32_t), uint32_t, uint64_t>;

template <typename T>
T javaAdd(T left, T right) {
  if constexpr (std::is_integral_v<T>) {
    using U = JavaUnsigned<T>;
    return static_cast<T>(static_cast<U>(left) + static_cast<U>(right));
  } else {
    return left + right;
  }
}

template <typename T>
T javaSubtract(T left, T right) {
  if constexpr (std::is_integral_v<T>) {
    using U = JavaUnsigned<T>;
    return static_cast<T>(static_cast<U>(left) - static_cast<U>(right));
  } else {
    return left - right;
  }
}

template <typename T>
T javaMultiply(T left, T right) {
  if constexpr (std::is_integral_v<T>) {
    using U = JavaUnsigned<T>;
    return static_cast<T>(static_cast<U>(left) * static_cast<U>(right));
  } else {
    return left * right;
  }
}

template <typename T>
T javaNegate(T value) {
  if constexpr (std::is_integral_v<T>) {
    using U = JavaUnsigned<T>;
    return static_cast<T>(U{0} - static_cast<U>(value));
  } else {
    return -value;
  }
}

// FieldSumAgg for integer and floating-point types.
template <typename T>
class SumAggregator final : public ValueAggregator<T> {
 public:
  using ValueAggregator<T>::ValueAggregator;

  void add(const BaseVector& input, vector_size_t row) override {
    const auto value = this->read(input, row);
    if (!this->value_.has_value()) {
      this->value_ = value;
    } else if (value.has_value()) {
      this->value_ = javaAdd(*this->value_, *value);
    }
  }

  void retract(const BaseVector& input, vector_size_t row) override {
    const auto value = this->read(input, row);
    if (!value.has_value()) {
      return;
    }
    this->value_ = this->value_.has_value()
        ? javaSubtract(*this->value_, *value)
        : javaNegate(*value);
  }
};

// FieldProductAgg for integer and floating-point types. Retracting divides.
template <typename T>
class ProductAggregator final : public ValueAggregator<T> {
 public:
  using ValueAggregator<T>::ValueAggregator;

  void add(const BaseVector& input, vector_size_t row) override {
    const auto value = this->read(input, row);
    if (!this->value_.has_value()) {
      this->value_ = value;
    } else if (value.has_value()) {
      this->value_ = javaMultiply(*this->value_, *value);
    }
  }

  void retract(const BaseVector& input, vector_size_t row) override {
    const auto value = this->read(input, row);
    if (!this->value_.has_value() || !value.has_value()) {
      return;
    }
    if constexpr (std::is_integral_v<T>) {
      VELOX_USER_CHECK_NE(
          *value,
          0,
          "Paimon aggregate function product cannot retract a zero: {}",
          this->column_);
      // Java wraps the minimum value divided by -1 around to itself.
      this->value_ = *value == -1 ? javaNegate(*this->value_)
                                  : static_cast<T>(*this->value_ / *value);
    } else {
      this->value_ = *this->value_ / *value;
    }
  }
};

// Returns true if 'value' has more digits than 'precision'.
bool exceedsPrecision(int128_t value, uint8_t precision) {
  return value >= DecimalUtil::kPowersOfTen[precision] ||
      value <= -DecimalUtil::kPowersOfTen[precision];
}

// FieldSumAgg for decimals, as Paimon Java's DecimalUtils.add and subtract
// do. Decimals of precision up to 18 add as longs and return null only on
// long overflow, while wider decimals return null when the result exceeds the
// precision. Negating checks the precision either way.
template <typename T>
class DecimalSumAggregator final : public ValueAggregator<T> {
 public:
  DecimalSumAggregator(
      std::string function,
      std::string column,
      uint8_t precision)
      : ValueAggregator<T>(std::move(function), std::move(column)),
        precision_(precision) {}

  void add(const BaseVector& input, vector_size_t row) override {
    const auto value = this->read(input, row);
    if (!this->value_.has_value()) {
      this->value_ = value;
    } else if (value.has_value()) {
      T result;
      this->value_ = __builtin_add_overflow(*this->value_, *value, &result)
          ? std::nullopt
          : checkPrecision(result);
    }
  }

  void retract(const BaseVector& input, vector_size_t row) override {
    const auto value = this->read(input, row);
    if (!value.has_value()) {
      return;
    }
    if (!this->value_.has_value()) {
      this->value_ = exceedsPrecision(*value, precision_)
          ? std::nullopt
          : std::optional<T>(-*value);
      return;
    }
    T result;
    this->value_ = __builtin_sub_overflow(*this->value_, *value, &result)
        ? std::nullopt
        : checkPrecision(result);
  }

 private:
  std::optional<T> checkPrecision(T value) const {
    if constexpr (std::is_same_v<T, int64_t>) {
      return value;
    } else {
      return exceedsPrecision(value, precision_) ? std::nullopt
                                                 : std::optional<T>(value);
    }
  }

  const uint8_t precision_;
};

int256_t toInt256(int128_t value) {
  const bool negative = value < 0;
  const auto magnitude = negative ? -static_cast<unsigned __int128>(value)
                                  : static_cast<unsigned __int128>(value);
  int256_t result = static_cast<uint64_t>(magnitude >> 64);
  result <<= 64;
  result |= static_cast<uint64_t>(magnitude);
  return negative ? int256_t(-result) : result;
}

// Converts 'value', which must fit in 128 bits.
int128_t toInt128(const int256_t& value) {
  const bool negative = value < 0;
  const int256_t magnitude = negative ? int256_t(-value) : value;
  const auto high = static_cast<uint64_t>(magnitude >> 64);
  const auto low =
      static_cast<uint64_t>(magnitude & std::numeric_limits<uint64_t>::max());
  const auto bits = (static_cast<unsigned __int128>(high) << 64) | low;
  return negative ? -static_cast<int128_t>(bits) : static_cast<int128_t>(bits);
}

int256_t powerOfTen(uint8_t exponent) {
  int256_t result = 1;
  for (uint8_t i = 0; i < exponent; ++i) {
    result *= 10;
  }
  return result;
}

// Divides and rounds half away from zero, as Java's RoundingMode.HALF_UP does.
int256_t divideRoundHalfUp(const int256_t& dividend, const int256_t& divisor) {
  int256_t quotient = dividend / divisor;
  const int256_t remainder = dividend % divisor;
  if (2 * boost::multiprecision::abs(remainder) >=
      boost::multiprecision::abs(divisor)) {
    quotient += (dividend < 0) != (divisor < 0) ? -1 : 1;
  }
  return quotient;
}

int256_t greatestCommonDivisor(int256_t left, int256_t right) {
  while (right != 0) {
    int256_t remainder = left % right;
    left = std::move(right);
    right = std::move(remainder);
  }
  return left;
}

// FieldProductAgg for decimals. Java multiplies and divides as BigDecimal and
// rounds the result half up to the scale, which returns null when it exceeds
// the precision. A division whose quotient does not terminate fails.
template <typename T>
class DecimalProductAggregator final : public ValueAggregator<T> {
 public:
  DecimalProductAggregator(
      std::string function,
      std::string column,
      uint8_t precision,
      uint8_t scale)
      : ValueAggregator<T>(std::move(function), std::move(column)),
        limit_(powerOfTen(precision)),
        scaleFactor_(powerOfTen(scale)) {}

  void add(const BaseVector& input, vector_size_t row) override {
    const auto value = this->read(input, row);
    if (!this->value_.has_value()) {
      this->value_ = value;
    } else if (value.has_value()) {
      this->value_ = toValue(divideRoundHalfUp(
          toInt256(*this->value_) * toInt256(*value), scaleFactor_));
    }
  }

  void retract(const BaseVector& input, vector_size_t row) override {
    const auto value = this->read(input, row);
    if (!this->value_.has_value() || !value.has_value()) {
      return;
    }
    VELOX_USER_CHECK_NE(
        *value,
        0,
        "Paimon aggregate function product cannot retract a zero: {}",
        this->column_);
    const auto dividend = toInt256(*this->value_);
    const auto divisor = toInt256(*value);
    // The quotient terminates when its reduced denominator has no prime
    // factors other than 2 and 5.
    int256_t denominator =
        abs(divisor) / greatestCommonDivisor(abs(dividend), abs(divisor));
    while (denominator % 2 == 0) {
      denominator /= 2;
    }
    while (denominator % 5 == 0) {
      denominator /= 5;
    }
    const bool terminates = denominator == 1;
    VELOX_USER_CHECK(
        terminates,
        "Paimon aggregate function product cannot retract a decimal whose "
        "quotient does not terminate: {}",
        this->column_);
    this->value_ = toValue(divideRoundHalfUp(dividend * scaleFactor_, divisor));
  }

 private:
  std::optional<T> toValue(const int256_t& unscaled) const {
    if (boost::multiprecision::abs(unscaled) >= limit_) {
      return std::nullopt;
    }
    return static_cast<T>(toInt128(unscaled));
  }

  // 10^precision, the smallest unscaled value that exceeds the precision.
  const int256_t limit_;
  // 10^scale.
  const int256_t scaleFactor_;
};

// FieldBoolAndAgg and FieldBoolOrAgg.
class BooleanAggregator final : public ValueAggregator<bool> {
 public:
  BooleanAggregator(std::string function, std::string column, bool isAnd)
      : ValueAggregator<bool>(std::move(function), std::move(column)),
        isAnd_(isAnd) {}

  void add(const BaseVector& input, vector_size_t row) override {
    const auto value = read(input, row);
    if (!value_.has_value()) {
      value_ = value;
    } else if (value.has_value()) {
      value_ = isAnd_ ? *value_ && *value : *value_ || *value;
    }
  }

 private:
  const bool isAnd_;
};

// FieldIgnoreRetractAgg: keeps the accumulator on retracts.
class IgnoreRetractAggregator final : public PaimonFieldAggregator {
 public:
  explicit IgnoreRetractAggregator(
      std::unique_ptr<PaimonFieldAggregator> aggregator)
      : aggregator_(std::move(aggregator)) {}

  void reset() override {
    aggregator_->reset();
  }

  void add(const BaseVector& input, vector_size_t row) override {
    aggregator_->add(input, row);
  }

  void retract(const BaseVector& /*input*/, vector_size_t /*row*/) override {}

  void set(const BaseVector& input, vector_size_t row) override {
    aggregator_->set(input, row);
  }

  void write(vector_size_t row, BaseVector& output) const override {
    aggregator_->write(row, output);
  }

 private:
  const std::unique_ptr<PaimonFieldAggregator> aggregator_;
};

// Returns true for the numeric types of Paimon Java's sum and product.
bool isNumeric(const Type& type) {
  if (type.isDecimal()) {
    return true;
  }
  switch (type.kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
      // Excludes DATE and other logical types of these kinds.
      return type.name() == TypeKindName::toName(type.kind());
    case TypeKind::BOOLEAN:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::UNKNOWN:
    case TypeKind::FUNCTION:
    case TypeKind::OPAQUE:
    case TypeKind::INVALID:
      return false;
    default:
      VELOX_FAIL("Unknown TypeKind: {}", static_cast<int>(type.kind()));
  }
}

// Returns true for the types Paimon Java's min and max compare.
bool isComparable(const Type& type) {
  switch (type.kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::HUGEINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::TIMESTAMP:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return true;
    case TypeKind::BOOLEAN:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::UNKNOWN:
    case TypeKind::FUNCTION:
    case TypeKind::OPAQUE:
    case TypeKind::INVALID:
      return false;
    default:
      VELOX_FAIL("Unknown TypeKind: {}", static_cast<int>(type.kind()));
  }
}

std::unique_ptr<PaimonFieldAggregator> createSum(
    const std::string& function,
    const std::string& column,
    const TypePtr& type) {
  if (type->isDecimal()) {
    const auto [precision, _] = getDecimalPrecisionScale(*type);
    if (type->isShortDecimal()) {
      return std::make_unique<DecimalSumAggregator<int64_t>>(
          function, column, precision);
    }
    return std::make_unique<DecimalSumAggregator<int128_t>>(
        function, column, precision);
  }
  switch (type->kind()) {
    case TypeKind::TINYINT:
      return std::make_unique<SumAggregator<int8_t>>(function, column);
    case TypeKind::SMALLINT:
      return std::make_unique<SumAggregator<int16_t>>(function, column);
    case TypeKind::INTEGER:
      return std::make_unique<SumAggregator<int32_t>>(function, column);
    case TypeKind::BIGINT:
      return std::make_unique<SumAggregator<int64_t>>(function, column);
    case TypeKind::REAL:
      return std::make_unique<SumAggregator<float>>(function, column);
    case TypeKind::DOUBLE:
      return std::make_unique<SumAggregator<double>>(function, column);
    case TypeKind::BOOLEAN:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::UNKNOWN:
    case TypeKind::FUNCTION:
    case TypeKind::OPAQUE:
    case TypeKind::INVALID:
      // isNumeric() rejects these types.
      VELOX_UNREACHABLE();
    default:
      VELOX_FAIL("Unknown TypeKind: {}", static_cast<int>(type->kind()));
  }
}

std::unique_ptr<PaimonFieldAggregator> createProduct(
    const std::string& function,
    const std::string& column,
    const TypePtr& type) {
  if (type->isDecimal()) {
    const auto [precision, scale] = getDecimalPrecisionScale(*type);
    if (type->isShortDecimal()) {
      return std::make_unique<DecimalProductAggregator<int64_t>>(
          function, column, precision, scale);
    }
    return std::make_unique<DecimalProductAggregator<int128_t>>(
        function, column, precision, scale);
  }
  switch (type->kind()) {
    case TypeKind::TINYINT:
      return std::make_unique<ProductAggregator<int8_t>>(function, column);
    case TypeKind::SMALLINT:
      return std::make_unique<ProductAggregator<int16_t>>(function, column);
    case TypeKind::INTEGER:
      return std::make_unique<ProductAggregator<int32_t>>(function, column);
    case TypeKind::BIGINT:
      return std::make_unique<ProductAggregator<int64_t>>(function, column);
    case TypeKind::REAL:
      return std::make_unique<ProductAggregator<float>>(function, column);
    case TypeKind::DOUBLE:
      return std::make_unique<ProductAggregator<double>>(function, column);
    case TypeKind::BOOLEAN:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::UNKNOWN:
    case TypeKind::FUNCTION:
    case TypeKind::OPAQUE:
    case TypeKind::INVALID:
      // isNumeric() rejects these types.
      VELOX_UNREACHABLE();
    default:
      VELOX_FAIL("Unknown TypeKind: {}", static_cast<int>(type->kind()));
  }
}

// Creates the aggregator of 'function' without the ignore-retract wrapper.
std::unique_ptr<PaimonFieldAggregator> createAggregator(
    const std::string& function,
    const std::string& column,
    const TypePtr& type) {
  const auto checkType = [&](bool supported) {
    VELOX_USER_CHECK(
        supported,
        "Paimon aggregate function {} does not support the type of column {}: "
        "{}",
        function,
        column,
        type->toString());
  };
  if (function == PaimonAggregateFunction::kPrimaryKey) {
    return std::make_unique<PrimaryKeyAggregator>(function, column);
  }
  if (function == PaimonAggregateFunction::kLastValue) {
    return std::make_unique<LastValueAggregator>(function, column);
  }
  if (function == PaimonAggregateFunction::kLastNonNullValue) {
    return std::make_unique<LastNonNullValueAggregator>(function, column);
  }
  if (function == PaimonAggregateFunction::kFirstValue) {
    return std::make_unique<FirstValueAggregator>(
        function, column, /*nonNull=*/false);
  }
  // kFirstNotNullValue is the legacy name of kFirstNonNullValue.
  if (function == PaimonAggregateFunction::kFirstNonNullValue ||
      function == PaimonAggregateFunction::kFirstNotNullValue) {
    return std::make_unique<FirstValueAggregator>(
        function, column, /*nonNull=*/true);
  }
  if (function == PaimonAggregateFunction::kMin ||
      function == PaimonAggregateFunction::kMax) {
    checkType(isComparable(*type));
    return std::make_unique<MinMaxAggregator>(
        function,
        column,
        /*min=*/function == PaimonAggregateFunction::kMin);
  }
  if (function == PaimonAggregateFunction::kSum) {
    checkType(isNumeric(*type));
    return createSum(function, column, type);
  }
  if (function == PaimonAggregateFunction::kProduct) {
    checkType(isNumeric(*type));
    return createProduct(function, column, type);
  }
  if (function == PaimonAggregateFunction::kBoolAnd ||
      function == PaimonAggregateFunction::kBoolOr) {
    checkType(type->kind() == TypeKind::BOOLEAN);
    return std::make_unique<BooleanAggregator>(
        function,
        column,
        /*isAnd=*/function == PaimonAggregateFunction::kBoolAnd);
  }
  for (const auto* name :
       {PaimonAggregateFunction::kListAgg,
        PaimonAggregateFunction::kCollect,
        PaimonAggregateFunction::kMergeMap,
        PaimonAggregateFunction::kMergeMapWithKeyTime,
        PaimonAggregateFunction::kNestedUpdate,
        PaimonAggregateFunction::kNestedPartialUpdate,
        PaimonAggregateFunction::kRoaringBitmap32,
        PaimonAggregateFunction::kRoaringBitmap64,
        PaimonAggregateFunction::kHllSketch,
        PaimonAggregateFunction::kThetaSketch}) {
    if (function == name) {
      VELOX_NYI(
          "Paimon merge-on-read does not support the aggregate function: {}",
          function);
    }
  }
  VELOX_USER_FAIL("Unknown Paimon aggregate function: {}", function);
}

} // namespace

// static
std::unique_ptr<PaimonFieldAggregator> PaimonFieldAggregator::create(
    const std::string& function,
    const std::string& column,
    const TypePtr& type,
    bool ignoreRetract) {
  auto aggregator = createAggregator(function, column, type);
  if (ignoreRetract) {
    return std::make_unique<IgnoreRetractAggregator>(std::move(aggregator));
  }
  return aggregator;
}

} // namespace facebook::velox::connector::paimon
