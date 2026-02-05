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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/Statistics.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "arrow/array.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_run_reader.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/ubsan.h"
#include "arrow/visit_data_inline.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/Encoding.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"

#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/HugeInt.h"

using arrow::default_memory_pool;
using arrow::MemoryPool;
using arrow::ResizableBuffer;
using arrow::internal::checked_cast;

namespace facebook::velox::parquet::arrow {
namespace {

template <typename U, typename T>
inline std::enable_if_t<
    std::is_trivially_copyable_v<T> && std::is_trivially_copyable_v<U> &&
        sizeof(T) == sizeof(U),
    U>
safeCopy(T value) {
  std::remove_const_t<U> ret;
  std::memcpy(&ret, &value, sizeof(T));
  return ret;
}

template <typename T>
inline std::enable_if_t<std::is_trivially_copyable_v<T>, T> safeLoad(
    const T* unaligned) {
  std::remove_const_t<T> ret;
  std::memcpy(&ret, unaligned, sizeof(T));
  return ret;
}

std::shared_ptr<ResizableBuffer> allocateBuffer(
    MemoryPool* pool,
    int64_t size) {
  PARQUET_ASSIGN_OR_THROW(
      auto result, ::arrow::AllocateResizableBuffer(size, pool));
  return std::move(result);
}

// ----------------------------------------------------------------------.
// Comparator implementations.

constexpr int valueLength(int valueLength, const ByteArray& value) {
  return value.len;
}
constexpr int valueLength(int typeLength, const FLBA& value) {
  return typeLength;
}

template <typename DType, bool isSigned>
struct CompareHelper {
  using T = typename DType::CType;

  static_assert(
      !std::is_unsigned<T>::value || std::is_same<T, bool>::value,
      "T is an unsigned numeric");

  constexpr static T defaultMin() {
    if constexpr (std::is_floating_point_v<T>) {
      return std::numeric_limits<T>::infinity();
    }
    return std::numeric_limits<T>::max();
  }
  constexpr static T defaultMax() {
    if constexpr (std::is_floating_point_v<T>) {
      return -std::numeric_limits<T>::infinity();
    }
    return std::numeric_limits<T>::min();
  }

  // MSVC17 fix, isnan is not overloaded for IntegralType as per C++11.
  // Standard requirements.
  template <typename T1 = T>
  static ::arrow::enable_if_t<std::is_floating_point<T1>::value, T> coalesce(
      T val,
      T fallback) {
    return std::isnan(val) ? fallback : val;
  }

  template <typename T1 = T>
  static ::arrow::enable_if_t<!std::is_floating_point<T1>::value, T> coalesce(
      T val,
      T fallback) {
    return val;
  }

  static inline bool compare(int typeLength, const T& a, const T& b) {
    return a < b;
  }

  static T min(int typeLength, T a, T b) {
    return a < b ? a : b;
  }
  static T max(int typeLength, T a, T b) {
    return a < b ? b : a;
  }
};

template <typename DType>
struct UnsignedCompareHelperBase {
  using T = typename DType::CType;
  using UCType = typename std::make_unsigned<T>::type;

  static_assert(!std::is_same<T, UCType>::value, "T is unsigned");
  static_assert(sizeof(T) == sizeof(UCType), "T and UCType not the same size");

  // NOTE: according to the C++ spec, unsigned-to-signed conversion is.
  // Implementation-defined if the original value does not fit in the signed.
  // Type (i.e., two's complement cannot be assumed even on mainstream
  // machines,. Because the compiler may decide otherwise).  Hence the use of
  // `SafeCopy`. Below for deterministic bit-casting. (See "Integer conversions"
  // in.
  //  https://en.cppreference.com/w/cpp/language/implicit_conversion)

  static const T defaultMin() {
    return safeCopy<T>(std::numeric_limits<UCType>::max());
  }
  static const T defaultMax() {
    return 0;
  }

  static T coalesce(T val, T fallback) {
    return val;
  }

  static bool compare(int typeLength, T a, T b) {
    return safeCopy<UCType>(a) < safeCopy<UCType>(b);
  }

  static T min(int typeLength, T a, T b) {
    return compare(typeLength, a, b) ? a : b;
  }
  static T max(int typeLength, T a, T b) {
    return compare(typeLength, a, b) ? b : a;
  }
};

template <>
struct CompareHelper<Int32Type, false>
    : public UnsignedCompareHelperBase<Int32Type> {};

template <>
struct CompareHelper<Int64Type, false>
    : public UnsignedCompareHelperBase<Int64Type> {};

template <bool isSigned>
struct CompareHelper<Int96Type, isSigned> {
  using T = typename Int96Type::CType;
  using MsbType = typename std::conditional<isSigned, int32_t, uint32_t>::type;

  static T defaultMin() {
    uint32_t kMsbMax = safeCopy<uint32_t>(std::numeric_limits<MsbType>::max());
    uint32_t kMax = std::numeric_limits<uint32_t>::max();
    return {kMax, kMax, kMsbMax};
  }
  static T defaultMax() {
    uint32_t kMsbMin = safeCopy<uint32_t>(std::numeric_limits<MsbType>::min());
    uint32_t kMin = std::numeric_limits<uint32_t>::min();
    return {kMin, kMin, kMsbMin};
  }
  static T coalesce(T val, T fallback) {
    return val;
  }

  static inline bool compare(int typeLength, const T& a, const T& b) {
    if (a.value[2] != b.value[2]) {
      // Only the MSB bit is by Signed comparison. For little-endian, this is.
      // The last bit of Int96 type.
      return safeCopy<MsbType>(a.value[2]) < safeCopy<MsbType>(b.value[2]);
    } else if (a.value[1] != b.value[1]) {
      return (a.value[1] < b.value[1]);
    }
    return (a.value[0] < b.value[0]);
  }

  static T min(int typeLength, const T& a, const T& b) {
    return compare(0, a, b) ? a : b;
  }
  static T max(int typeLength, const T& a, const T& b) {
    return compare(0, a, b) ? b : a;
  }
};

template <typename T, bool isSigned>
struct BinaryLikeComparer {};

template <typename T>
struct BinaryLikeComparer<T, /*is_signed=*/false> {
  static bool compare(int typeLength, const T& a, const T& b) {
    int aLength = valueLength(typeLength, a);
    int bLength = valueLength(typeLength, b);
    // Unsigned comparison is used for non-numeric types so straight.
    // Lexicographic comparison makes sense. (a.ptr is always unsigned)....
    return std::lexicographical_compare(
        a.ptr, a.ptr + aLength, b.ptr, b.ptr + bLength);
  }
};

template <typename T>
struct BinaryLikeComparer<T, /*is_signed=*/true> {
  static bool compare(int typeLength, const T& a, const T& b) {
    // Is signed is used for integers encoded as big-endian twos.
    // Complement integers. (e.g. decimals).
    int aLength = valueLength(typeLength, a);
    int bLength = valueLength(typeLength, b);

    // At least of the lengths is zero.
    if (aLength == 0 || bLength == 0) {
      return aLength == 0 && bLength > 0;
    }

    int8_t firstA = *a.ptr;
    int8_t firstB = *b.ptr;
    // We can short circuit for different signed numbers or.
    // For equal length bytes arrays that have different first bytes.
    // The equality requirement is necessary for sign extension cases.
    // 0XFF10 should be equal to 0x10 (due to big endian sign extension).
    if ((0x80 & firstA) != (0x80 & firstB) ||
        (aLength == bLength && firstA != firstB)) {
      return firstA < firstB;
    }
    // When the lengths are unequal and the numbers are of the same.
    // Sign we need to do comparison by sign extending the shorter.
    // Value first, and once we get to equal sized arrays, lexicographical.
    // Unsigned comparison of everything but the first byte is sufficient.
    const uint8_t* aStart = a.ptr;
    const uint8_t* bStart = b.ptr;
    if (aLength != bLength) {
      const uint8_t* leadStart = nullptr;
      const uint8_t* leadEnd = nullptr;
      if (aLength > bLength) {
        int leadLength = aLength - bLength;
        leadStart = a.ptr;
        leadEnd = a.ptr + leadLength;
        aStart += leadLength;
      } else {
        VELOX_DCHECK_LT(aLength, bLength);
        int leadLength = bLength - aLength;
        leadStart = b.ptr;
        leadEnd = b.ptr + leadLength;
        bStart += leadLength;
      }
      // Compare extra bytes to the sign extension of the first.
      // Byte of the other number.
      uint8_t extension = firstA < 0 ? 0xFF : 0;
      bool notEqual = std::any_of(leadStart, leadEnd, [extension](uint8_t a) {
        return extension != a;
      });
      if (notEqual) {
        // Since sign extension are extrema values for unsigned bytes:
        //
        // Four cases exist:
        //    Negative values:
        //      B is the longer value.
        //        B must be the lesser value: return false.
        //      Else:
        //        A must be the lesser value: return true.
        //
        //    Positive values:
        //      B is the longer value.
        //        Values in b must be greater than a: return true.
        //      Else:
        //        Values in a must be greater than b: return false.
        bool negativeValues = firstA < 0;
        bool bLonger = aLength < bLength;
        return negativeValues != bLonger;
      }
    } else {
      aStart++;
      bStart++;
    }
    return std::lexicographical_compare(
        aStart, a.ptr + aLength, bStart, b.ptr + bLength);
  }
};

template <typename DType, bool isSigned>
struct BinaryLikeCompareHelperBase {
  using T = typename DType::CType;

  static T defaultMin() {
    return {};
  }
  static T defaultMax() {
    return {};
  }
  static T coalesce(T val, T fallback) {
    return val;
  }

  static inline bool compare(int typeLength, const T& a, const T& b) {
    return BinaryLikeComparer<T, isSigned>::compare(typeLength, a, b);
  }
  static T min(int typeLength, const T& a, const T& b) {
    if (a.ptr == nullptr)
      return b;
    if (b.ptr == nullptr)
      return a;
    return compare(typeLength, a, b) ? a : b;
  }

  static T max(int typeLength, const T& a, const T& b) {
    if (a.ptr == nullptr)
      return b;
    if (b.ptr == nullptr)
      return a;
    return compare(typeLength, a, b) ? b : a;
  }
};

template <bool isSigned>
struct CompareHelper<ByteArrayType, isSigned>
    : public BinaryLikeCompareHelperBase<ByteArrayType, isSigned> {};

template <bool isSigned>
struct CompareHelper<FLBAType, isSigned>
    : public BinaryLikeCompareHelperBase<FLBAType, isSigned> {};

using ::std::optional;

template <typename T>
::arrow::enable_if_t<std::is_integral<T>::value, optional<std::pair<T, T>>>
cleanStatistic(std::pair<T, T> minMax) {
  return minMax;
}

// In case of floating point types, the following rules are applied (as per.
// Upstream parquet-mr):
// - If any of min/max is NaN, return nothing.
// - If min is infinity and max is -infinity, return nothing.
// - If min is 0.0f, replace with -0.0f.
// - If max is -0.0f, replace with 0.0f.
template <typename T>
::arrow::
    enable_if_t<std::is_floating_point<T>::value, optional<std::pair<T, T>>>
    cleanStatistic(std::pair<T, T> minMax) {
  T min = minMax.first;
  T max = minMax.second;

  // Ignore if one of the value is nan.
  if (std::isnan(min) || std::isnan(max)) {
    return ::std::nullopt;
  }

  if (min == std::numeric_limits<T>::infinity() &&
      max == -std::numeric_limits<T>::infinity()) {
    return ::std::nullopt;
  }

  T zero{};

  if (min == zero && !std::signbit(min)) {
    min = -min;
  }

  if (max == zero && std::signbit(max)) {
    max = -max;
  }

  return {{min, max}};
}

optional<std::pair<FLBA, FLBA>> cleanStatistic(std::pair<FLBA, FLBA> minMax) {
  if (minMax.first.ptr == nullptr || minMax.second.ptr == nullptr) {
    return ::std::nullopt;
  }
  return minMax;
}

optional<std::pair<ByteArray, ByteArray>> cleanStatistic(
    std::pair<ByteArray, ByteArray> minMax) {
  if (minMax.first.ptr == nullptr || minMax.second.ptr == nullptr) {
    return ::std::nullopt;
  }
  return minMax;
}

template <bool isSigned, typename DType>
class TypedComparatorImpl : virtual public TypedComparator<DType> {
 public:
  using T = typename DType::CType;
  using Helper = CompareHelper<DType, isSigned>;

  explicit TypedComparatorImpl(int typeLength = -1) : typeLength_(typeLength) {}

  bool compareInline(const T& a, const T& b) const {
    return Helper::compare(typeLength_, a, b);
  }

  bool compare(const T& a, const T& b) override {
    return compareInline(a, b);
  }

  std::pair<T, T> getMinMax(const T* values, int64_t length) override {
    VELOX_DCHECK_GT(length, 0);

    T min = Helper::defaultMin();
    T max = Helper::defaultMax();

    for (int64_t i = 0; i < length; i++) {
      const auto val = safeLoad(values + i);
      min = Helper::min(
          typeLength_, min, Helper::coalesce(val, Helper::defaultMin()));
      max = Helper::max(
          typeLength_, max, Helper::coalesce(val, Helper::defaultMax()));
    }

    return {min, max};
  }

  std::pair<T, T> getMinMaxSpaced(
      const T* values,
      int64_t length,
      const uint8_t* validBits,
      int64_t validBitsOffset) override {
    VELOX_DCHECK_GT(length, 0);

    T min = Helper::defaultMin();
    T max = Helper::defaultMax();

    ::arrow::internal::VisitSetBitRunsVoid(
        validBits,
        validBitsOffset,
        length,
        [&](int64_t position, int64_t length) {
          for (int64_t i = 0; i < length; i++) {
            const auto val = safeLoad(values + i + position);
            min = Helper::min(
                typeLength_, min, Helper::coalesce(val, Helper::defaultMin()));
            max = Helper::max(
                typeLength_, max, Helper::coalesce(val, Helper::defaultMax()));
          }
        });

    return {min, max};
  }

  std::pair<T, T> getMinMax(const ::arrow::Array& values) override;

 private:
  int typeLength_;
};

// ARROW-11675: A hand-written version of GetMinMax(), to work around.
// What looks like a MSVC code generation bug.
// This does not seem to be required for GetMinMaxSpaced().
template <>
std::pair<int32_t, int32_t>
TypedComparatorImpl</*is_signed=*/false, Int32Type>::getMinMax(
    const int32_t* values,
    int64_t length) {
  VELOX_DCHECK_GT(length, 0);

  const uint32_t* unsignedValues = reinterpret_cast<const uint32_t*>(values);
  uint32_t min = std::numeric_limits<uint32_t>::max();
  uint32_t max = std::numeric_limits<uint32_t>::lowest();

  for (int64_t i = 0; i < length; i++) {
    const auto val = unsignedValues[i];
    min = std::min<uint32_t>(min, val);
    max = std::max<uint32_t>(max, val);
  }

  return {safeCopy<int32_t>(min), safeCopy<int32_t>(max)};
}

template <bool isSigned, typename DType>
std::pair<typename DType::CType, typename DType::CType>
TypedComparatorImpl<isSigned, DType>::getMinMax(const ::arrow::Array& values) {
  ParquetException::NYI(values.type()->ToString());
}

template <bool isSigned>
std::pair<ByteArray, ByteArray> getMinMaxBinaryHelper(
    const TypedComparatorImpl<isSigned, ByteArrayType>& Comparator,
    const ::arrow::Array& values) {
  using Helper = CompareHelper<ByteArrayType, isSigned>;

  ByteArray min = Helper::defaultMin();
  ByteArray max = Helper::defaultMax();
  constexpr int typeLength = -1;

  const auto validFunc = [&](std::string_view val) {
    ByteArray ba{std::string_view(val.data(), val.size())};
    min = Helper::min(typeLength, ba, min);
    max = Helper::max(typeLength, ba, max);
  };
  const auto nullFunc = [&]() {};

  if (::arrow::is_binary_like(values.type_id())) {
    ::arrow::VisitArraySpanInline<::arrow::BinaryType>(
        *values.data(), std::move(validFunc), std::move(nullFunc));
  } else {
    VELOX_DCHECK(::arrow::is_large_binary_like(values.type_id()));
    ::arrow::VisitArraySpanInline<::arrow::LargeBinaryType>(
        *values.data(), std::move(validFunc), std::move(nullFunc));
  }

  return {min, max};
}

template <>
std::pair<ByteArray, ByteArray>
TypedComparatorImpl<true, ByteArrayType>::getMinMax(
    const ::arrow::Array& values) {
  return getMinMaxBinaryHelper<true>(*this, values);
}

template <>
std::pair<ByteArray, ByteArray>
TypedComparatorImpl<false, ByteArrayType>::getMinMax(
    const ::arrow::Array& values) {
  return getMinMaxBinaryHelper<false>(*this, values);
}

template <typename T>
std::string encodeDecimalToBigEndian(T value) {
  uint8_t buffer[sizeof(T)];
  if constexpr (std::is_same_v<T, int64_t>) {
    *reinterpret_cast<int64_t*>(buffer) = ::arrow::bit_util::ToBigEndian(value);
  } else if constexpr (std::is_same_v<T, int128_t>) {
    *reinterpret_cast<int128_t*>(buffer) = DecimalUtil::bigEndian(value);
  }
  return std::string(reinterpret_cast<const char*>(buffer), sizeof(T));
}

template <typename DType>
class TypedStatisticsImpl : public TypedStatistics<DType> {
 public:
  using T = typename DType::CType;

  // Create an empty stats.
  TypedStatisticsImpl(const ColumnDescriptor* descr, MemoryPool* pool)
      : descr_(descr),
        pool_(pool),
        minBuffer_(allocateBuffer(pool_, 0)),
        maxBuffer_(allocateBuffer(pool_, 0)) {
    auto comp = Comparator::make(descr);
    comparator_ = std::static_pointer_cast<TypedComparator<DType>>(comp);
    TypedStatisticsImpl::reset();
  }

  // Create stats from provided values.
  TypedStatisticsImpl(
      const T& min,
      const T& max,
      int64_t numValues,
      int64_t nullCount,
      int64_t distinctCount)
      : pool_(default_memory_pool()),
        minBuffer_(allocateBuffer(pool_, 0)),
        maxBuffer_(allocateBuffer(pool_, 0)) {
    TypedStatisticsImpl::incrementNumValues(numValues);
    TypedStatisticsImpl::incrementNullCount(nullCount);
    setDistinctCount(distinctCount);

    copy(min, &min_, minBuffer_.get());
    copy(max, &max_, maxBuffer_.get());
    hasMinMax_ = true;
  }

  // Create stats from a thrift Statistics object.
  TypedStatisticsImpl(
      const ColumnDescriptor* descr,
      const std::string& encodedMin,
      const std::string& encodedMax,
      int64_t numValues,
      int64_t nullCount,
      int64_t distinctCount,
      bool hasMinMax,
      bool hasNullCount,
      bool hasDistinctCount,
      bool hasNaNCount,
      int64_t nanCount,
      MemoryPool* pool)
      : TypedStatisticsImpl(descr, pool) {
    TypedStatisticsImpl::incrementNumValues(numValues);
    if (hasNullCount) {
      TypedStatisticsImpl::incrementNullCount(nullCount);
    } else {
      hasNullCount_ = false;
    }
    if (hasDistinctCount) {
      setDistinctCount(distinctCount);
    } else {
      hasDistinctCount_ = false;
    }

    if (hasNaNCount) {
      incrementNaNValues(nanCount);
    } else {
      hasNanCount_ = false;
    }

    if (!encodedMin.empty()) {
      plainDecode(encodedMin, &min_);
    }
    if (!encodedMax.empty()) {
      plainDecode(encodedMax, &max_);
    }
    hasMinMax_ = hasMinMax;
  }

  bool hasDistinctCount() const override {
    return hasDistinctCount_;
  };
  bool hasMinMax() const override {
    return hasMinMax_;
  }
  bool hasNullCount() const override {
    return hasNullCount_;
  };

  bool hasNaNCount() const override {
    return hasNanCount_;
  }

  void incrementNullCount(int64_t n) override {
    statistics_.nullCount += n;
    hasNullCount_ = true;
  }

  void incrementNumValues(int64_t n) override {
    numValues_ += n;
  }

  void incrementNaNValues(int64_t n) override {
    if (n > 0) {
      nanCount_ += n;
      hasNanCount_ = true;
    }
  }

  bool equals(const Statistics& rawOther) const override {
    if (physicalType() != rawOther.physicalType())
      return false;

    const auto& other = checked_cast<const TypedStatisticsImpl&>(rawOther);

    if (hasMinMax_ != other.hasMinMax_)
      return false;
    if (hasMinMax_) {
      if (!minMaxEqual(other))
        return false;
    }

    return nullCount() == other.nullCount() &&
        distinctCount() == other.distinctCount() &&
        numValues() == other.numValues();
  }

  bool minMaxEqual(const TypedStatisticsImpl& other) const;

  void reset() override {
    resetCounts();
    resetHasFlags();
  }

  void setMinMax(const T& argMin, const T& argMax) override {
    setMinMaxPair({argMin, argMax});
  }

  void merge(const TypedStatistics<DType>& other) override {
    this->numValues_ += other.numValues();
    // Null_count is always valid when merging page statistics into.
    // Column chunk statistics.
    if (other.hasNullCount()) {
      this->statistics_.nullCount += other.nullCount();
    } else {
      this->hasNullCount_ = false;
    }
    if (other.hasNaNCount()) {
      this->nanCount_ += other.nanCount();
      this->hasNanCount_ = true;
    }
    if (hasDistinctCount_ && other.hasDistinctCount() &&
        (distinctCount() == 0 || other.distinctCount() == 0)) {
      // We can merge distinct counts if either side is zero.
      statistics_.distinctCount =
          std::max(statistics_.distinctCount, other.distinctCount());
    } else {
      // Otherwise clear has_distinct_count_ as distinct count cannot be merged.
      this->hasDistinctCount_ = false;
    }
    // Do not clear min/max here if the other side does not provide.
    // Min/max which may happen when other is an empty stats or all.
    // Its values are null and/or NaN.
    if (other.hasMinMax()) {
      setMinMax(other.min(), other.max());
    }
  }

  void update(const T* values, int64_t numValues, int64_t nullCount) override;
  void updateSpaced(
      const T* values,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      int64_t numSpacedValues,
      int64_t numValues,
      int64_t nullCount) override;

  void update(const ::arrow::Array& values, bool updateCounts) override {
    if (updateCounts) {
      incrementNullCount(values.null_count());
      incrementNumValues(values.length() - values.null_count());
    }

    if (values.null_count() == values.length()) {
      return;
    }

    setMinMaxPair(comparator_->getMinMax(values));
  }

  const T& min() const override {
    return min_;
  }

  const T& max() const override {
    return max_;
  }

  Type::type physicalType() const override {
    return descr_->physicalType();
  }

  const ColumnDescriptor* descr() const override {
    return descr_;
  }

  std::string encodeMin() const override {
    std::string s;
    if (hasMinMax())
      this->plainEncode(min_, &s);
    return s;
  }

  std::string encodeMax() const override {
    std::string s;
    if (hasMinMax())
      this->plainEncode(max_, &s);
    return s;
  }

  std::string icebergLowerBoundInclusive(int32_t truncateTo) const override {
    if constexpr (std::is_same_v<T, int64_t>) {
      if (descr_->logicalType()->isDecimal()) {
        return encodeDecimalToBigEndian(min_);
      }
    }
    if constexpr (std::is_same_v<T, int128_t>) {
      return encodeDecimalToBigEndian(min_);
    }
    if constexpr (std::is_same_v<T, ByteArray>) {
      const auto truncatedMin = functions::stringImpl::truncateUtf8(
          std::string_view(min_), truncateTo);
      std::string s;
      this->plainEncode(
          ByteArray(
              truncatedMin.size(),
              reinterpret_cast<const uint8_t*>(truncatedMin.data())),
          &s);
      return s;
    }
    return encodeMin();
  }

  std::optional<std::string> icebergUpperBoundExclusive(
      int32_t truncateTo) const override {
    if constexpr (std::is_same_v<T, int64_t>) {
      if (descr_->logicalType()->isDecimal()) {
        return encodeDecimalToBigEndian(max_);
      }
    }
    if constexpr (std::is_same_v<T, int128_t>) {
      return encodeDecimalToBigEndian(max_);
    }
    if constexpr (std::is_same_v<T, ByteArray>) {
      const auto truncatedMax = functions::stringImpl::roundUpUtf8(
          std::string_view(max_), truncateTo);
      if (!truncatedMax.has_value()) {
        return std::nullopt;
      }
      std::string s;
      this->plainEncode(
          ByteArray(
              truncatedMax->size(),
              reinterpret_cast<const uint8_t*>(truncatedMax->data())),
          &s);
      return s;
    }
    return encodeMax();
  }

  EncodedStatistics encode() override {
    EncodedStatistics s;
    if (hasMinMax()) {
      s.setMin(this->encodeMin());
      s.setMax(this->encodeMax());
    }
    if (hasNullCount()) {
      s.setNullCount(this->nullCount());
      // Num_values_ is reliable and it means number of non-null values.
      s.allNullValue = numValues_ == 0;
    }
    if (hasDistinctCount()) {
      s.setDistinctCount(this->distinctCount());
    }
    if (hasNanCount_) {
      s.set_nan_count(nanCount_);
    }
    return s;
  }

  int64_t nullCount() const override {
    return statistics_.nullCount;
  }
  int64_t distinctCount() const override {
    return statistics_.distinctCount;
  }
  int64_t numValues() const override {
    return numValues_;
  }

  int64_t nanCount() const override {
    return nanCount_;
  }

  bool maxGreaterThan(const Statistics& other) const override {
    const auto* typedOther =
        dynamic_cast<const TypedStatisticsImpl<DType>*>(&other);
    return comparator_->compare(max_, typedOther->max_) ? false : true;
  }

  bool minLessThan(const Statistics& other) const override {
    const auto* typedOther =
        dynamic_cast<const TypedStatisticsImpl<DType>*>(&other);
    return comparator_->compare(min_, typedOther->min_) ? true : false;
  }

 private:
  const ColumnDescriptor* descr_;
  bool hasMinMax_ = false;
  bool hasNullCount_ = false;
  bool hasDistinctCount_ = false;
  bool hasNanCount_ = false;
  T min_;
  T max_;
  ::arrow::MemoryPool* pool_;
  // Number of non-null values.
  // Please note that num_values_ is reliable when has_null_count_ is set.
  // When has_null_count_ is not set, e.g. a page statistics created from.
  // a statistics thrift message which doesn't have the optional null_count,.
  // `num_values_` may include null values.
  int64_t numValues_ = 0;
  // NaN count is tracked separately since it's not written to the parquet file.
  int64_t nanCount_ = 0;
  EncodedStatistics statistics_;
  std::shared_ptr<TypedComparator<DType>> comparator_;
  std::shared_ptr<ResizableBuffer> minBuffer_, maxBuffer_;

  void plainEncode(const T& src, std::string* dst) const;
  void plainDecode(const std::string& src, T* dst) const;

  void copy(const T& src, T* dst, ResizableBuffer*) {
    *dst = src;
  }

  void setDistinctCount(int64_t n) {
    // Distinct count can only be "set", and cannot be incremented.
    statistics_.distinctCount = n;
    hasDistinctCount_ = true;
  }

  void resetCounts() {
    this->statistics_.nullCount = 0;
    this->statistics_.distinctCount = 0;
    this->numValues_ = 0;
    this->nanCount_ = 0;
  }

  void resetHasFlags() {
    // Has_min_max_ will only be set when it meets any valid value.
    this->hasMinMax_ = false;
    // has_distinct_count_ will only be set once SetDistinctCount()
    // Is called because distinct count calculation is not cheap and.
    // Disabled by default.
    this->hasDistinctCount_ = false;
    // Null count calculation is cheap and enabled by default.
    this->hasNullCount_ = true;
    this->hasNanCount_ = false;
  }

  void setMinMaxPair(std::pair<T, T> minMax) {
    // CleanStatistic can return a nullopt in case of erroneous values, e.g.
    // NaN.
    auto maybeMinMax = cleanStatistic(minMax);
    if (!maybeMinMax)
      return;

    auto min = maybeMinMax.value().first;
    auto max = maybeMinMax.value().second;

    if (!hasMinMax_) {
      hasMinMax_ = true;
      copy(min, &min_, minBuffer_.get());
      copy(max, &max_, maxBuffer_.get());
    } else {
      copy(
          comparator_->compare(min_, min) ? min_ : min,
          &min_,
          minBuffer_.get());
      copy(
          comparator_->compare(max_, max) ? max : max_,
          &max_,
          maxBuffer_.get());
    }
  }

  int64_t countNaN(const T* values, int64_t length) {
    if constexpr (!std::is_floating_point_v<T>) {
      return 0;
    } else {
      int64_t count = 0;
      for (auto i = 0; i < length; i++) {
        const auto val = safeLoad(values + i);
        if (std::isnan(val)) {
          count++;
        }
      }
      return count;
    }
  }

  int64_t countNaNSpaced(
      const T* values,
      int64_t length,
      const uint8_t* valid_bits,
      int64_t valid_bits_offset) {
    if constexpr (!std::is_floating_point_v<T>) {
      return 0;
    } else {
      int64_t count = 0;
      ::arrow::internal::VisitSetBitRunsVoid(
          valid_bits,
          valid_bits_offset,
          length,
          [&](int64_t position, int64_t run_length) {
            for (auto i = 0; i < run_length; i++) {
              const auto val = safeLoad(values + i + position);
              if (std::isnan(val)) {
                count++;
              }
            }
          });
      return count;
    }
  }
};

template <>
inline bool TypedStatisticsImpl<FLBAType>::minMaxEqual(
    const TypedStatisticsImpl<FLBAType>& other) const {
  uint32_t len = descr_->typeLength();
  return std::memcmp(min_.ptr, other.min_.ptr, len) == 0 &&
      std::memcmp(max_.ptr, other.max_.ptr, len) == 0;
}

template <typename DType>
bool TypedStatisticsImpl<DType>::minMaxEqual(
    const TypedStatisticsImpl<DType>& other) const {
  return min_ == other.min_ && max_ == other.max_;
}

template <>
inline void TypedStatisticsImpl<FLBAType>::copy(
    const FLBA& src,
    FLBA* dst,
    ResizableBuffer* buffer) {
  if (dst->ptr == src.ptr)
    return;
  uint32_t len = descr_->typeLength();
  PARQUET_THROW_NOT_OK(buffer->Resize(len, false));
  std::memcpy(buffer->mutable_data(), src.ptr, len);
  *dst = FLBA(buffer->data());
}

template <>
inline void TypedStatisticsImpl<ByteArrayType>::copy(
    const ByteArray& src,
    ByteArray* dst,
    ResizableBuffer* buffer) {
  if (dst->ptr == src.ptr)
    return;
  PARQUET_THROW_NOT_OK(buffer->Resize(src.len, false));
  std::memcpy(buffer->mutable_data(), src.ptr, src.len);
  *dst = ByteArray(src.len, buffer->data());
}

template <typename DType>
void TypedStatisticsImpl<DType>::update(
    const T* values,
    int64_t numValues,
    int64_t nullCount) {
  VELOX_DCHECK_GE(numValues, 0);
  VELOX_DCHECK_GE(nullCount, 0);

  incrementNullCount(nullCount);
  incrementNumValues(numValues);

  if (numValues == 0)
    return;
  setMinMaxPair(comparator_->getMinMax(values, numValues));
  incrementNaNValues(countNaN(values, numValues));
}

template <typename DType>
void TypedStatisticsImpl<DType>::updateSpaced(
    const T* values,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    int64_t numSpacedValues,
    int64_t numValues,
    int64_t nullCount) {
  VELOX_DCHECK_GE(numValues, 0);
  VELOX_DCHECK_GE(nullCount, 0);

  incrementNullCount(nullCount);
  incrementNumValues(numValues);

  if (numValues == 0)
    return;
  setMinMaxPair(comparator_->getMinMaxSpaced(
      values, numSpacedValues, validBits, validBitsOffset));
  incrementNaNValues(
      countNaNSpaced(values, numSpacedValues, validBits, validBitsOffset));
}

template <typename DType>
void TypedStatisticsImpl<DType>::plainEncode(const T& src, std::string* dst)
    const {
  auto encoder =
      makeTypedEncoder<DType>(Encoding::kPlain, false, descr_, pool_);
  encoder->put(&src, 1);
  auto buffer = encoder->flushValues();
  auto ptr = reinterpret_cast<const char*>(buffer->data());
  dst->assign(ptr, buffer->size());
}

template <typename DType>
void TypedStatisticsImpl<DType>::plainDecode(const std::string& src, T* dst)
    const {
  auto decoder = makeTypedDecoder<DType>(Encoding::kPlain, descr_);
  decoder->setData(
      1,
      reinterpret_cast<const uint8_t*>(src.c_str()),
      static_cast<int>(src.size()));
  decoder->decode(dst, 1);
}

template <>
void TypedStatisticsImpl<ByteArrayType>::plainEncode(
    const T& src,
    std::string* dst) const {
  dst->assign(reinterpret_cast<const char*>(src.ptr), src.len);
}

template <>
void TypedStatisticsImpl<ByteArrayType>::plainDecode(
    const std::string& src,
    T* dst) const {
  dst->len = static_cast<uint32_t>(src.size());
  dst->ptr = reinterpret_cast<const uint8_t*>(src.c_str());
}

} // namespace

// ----------------------------------------------------------------------.
// Public factory functions.

std::shared_ptr<Comparator> Comparator::make(
    Type::type physicalType,
    SortOrder::type sortOrder,
    int typeLength) {
  if (SortOrder::kSigned == sortOrder) {
    switch (physicalType) {
      case Type::kBoolean:
        return std::make_shared<TypedComparatorImpl<true, BooleanType>>();
      case Type::kInt32:
        return std::make_shared<TypedComparatorImpl<true, Int32Type>>();
      case Type::kInt64:
        return std::make_shared<TypedComparatorImpl<true, Int64Type>>();
      case Type::kInt96:
        return std::make_shared<TypedComparatorImpl<true, Int96Type>>();
      case Type::kFloat:
        return std::make_shared<TypedComparatorImpl<true, FloatType>>();
      case Type::kDouble:
        return std::make_shared<TypedComparatorImpl<true, DoubleType>>();
      case Type::kByteArray:
        return std::make_shared<TypedComparatorImpl<true, ByteArrayType>>();
      case Type::kFixedLenByteArray:
        return std::make_shared<TypedComparatorImpl<true, FLBAType>>(
            typeLength);
      default:
        ParquetException::NYI("Signed Compare not implemented");
    }
  } else if (SortOrder::kUnsigned == sortOrder) {
    switch (physicalType) {
      case Type::kInt32:
        return std::make_shared<TypedComparatorImpl<false, Int32Type>>();
      case Type::kInt64:
        return std::make_shared<TypedComparatorImpl<false, Int64Type>>();
      case Type::kInt96:
        return std::make_shared<TypedComparatorImpl<false, Int96Type>>();
      case Type::kByteArray:
        return std::make_shared<TypedComparatorImpl<false, ByteArrayType>>();
      case Type::kFixedLenByteArray:
        return std::make_shared<TypedComparatorImpl<false, FLBAType>>(
            typeLength);
      default:
        ParquetException::NYI("Unsigned Compare not implemented");
    }
  } else {
    throw ParquetException("UNKNOWN Sort Order");
  }
  return nullptr;
}

std::shared_ptr<Comparator> Comparator::make(const ColumnDescriptor* descr) {
  return make(descr->physicalType(), descr->sortOrder(), descr->typeLength());
}

std::shared_ptr<Statistics> Statistics::make(
    const ColumnDescriptor* descr,
    ::arrow::MemoryPool* pool) {
  switch (descr->physicalType()) {
    case Type::kBoolean:
      return std::make_shared<TypedStatisticsImpl<BooleanType>>(descr, pool);
    case Type::kInt32:
      return std::make_shared<TypedStatisticsImpl<Int32Type>>(descr, pool);
    case Type::kInt64:
      return std::make_shared<TypedStatisticsImpl<Int64Type>>(descr, pool);
    case Type::kFloat:
      return std::make_shared<TypedStatisticsImpl<FloatType>>(descr, pool);
    case Type::kDouble:
      return std::make_shared<TypedStatisticsImpl<DoubleType>>(descr, pool);
    case Type::kByteArray:
      return std::make_shared<TypedStatisticsImpl<ByteArrayType>>(descr, pool);
    case Type::kFixedLenByteArray:
      return std::make_shared<TypedStatisticsImpl<FLBAType>>(descr, pool);
    default:
      ParquetException::NYI("Statistics not implemented");
  }
}

std::shared_ptr<Statistics> Statistics::make(
    Type::type physicalType,
    const void* min,
    const void* max,
    int64_t numValues,
    int64_t nullCount,
    int64_t distinctCount) {
#define MAKE_STATS(CAP_TYPE, KLASS)                           \
  case Type::CAP_TYPE:                                        \
    return std::make_shared<TypedStatisticsImpl<KLASS>>(      \
        *reinterpret_cast<const typename KLASS::CType*>(min), \
        *reinterpret_cast<const typename KLASS::CType*>(max), \
        numValues,                                            \
        nullCount,                                            \
        distinctCount)

  switch (physicalType) {
    MAKE_STATS(kBoolean, BooleanType);
    MAKE_STATS(kInt32, Int32Type);
    MAKE_STATS(kInt64, Int64Type);
    MAKE_STATS(kFloat, FloatType);
    MAKE_STATS(kDouble, DoubleType);
    MAKE_STATS(kByteArray, ByteArrayType);
    MAKE_STATS(kFixedLenByteArray, FLBAType);
    default:
      break;
  }
#undef MAKE_STATS
  VELOX_DCHECK(false, "Cannot reach here");
  return nullptr;
}

std::shared_ptr<Statistics> Statistics::make(
    const ColumnDescriptor* descr,
    const EncodedStatistics* encodedStats,
    int64_t numValues,
    ::arrow::MemoryPool* pool) {
  VELOX_DCHECK(encodedStats != nullptr);
  return make(
      descr,
      encodedStats->min(),
      encodedStats->max(),
      numValues,
      encodedStats->nullCount,
      encodedStats->distinctCount,
      encodedStats->hasMin && encodedStats->hasMax,
      encodedStats->hasNullCount,
      encodedStats->hasDistinctCount,
      encodedStats->hasNanCount,
      encodedStats->nanCount,
      pool);
}

std::shared_ptr<Statistics> Statistics::make(
    const ColumnDescriptor* descr,
    const std::string& encodedMin,
    const std::string& encodedMax,
    int64_t numValues,
    int64_t nullCount,
    int64_t distinctCount,
    bool hasMinMax,
    bool hasNullCount,
    bool hasDistinctCount,
    bool hasNaNCount,
    int64_t nanCount,
    ::arrow::MemoryPool* pool) {
#define MAKE_STATS(CAP_TYPE, KLASS)                      \
  case Type::CAP_TYPE:                                   \
    return std::make_shared<TypedStatisticsImpl<KLASS>>( \
        descr,                                           \
        encodedMin,                                      \
        encodedMax,                                      \
        numValues,                                       \
        nullCount,                                       \
        distinctCount,                                   \
        hasMinMax,                                       \
        hasNullCount,                                    \
        hasDistinctCount,                                \
        hasNaNCount,                                     \
        nanCount,                                        \
        pool)

  switch (descr->physicalType()) {
    MAKE_STATS(kBoolean, BooleanType);
    MAKE_STATS(kInt32, Int32Type);
    MAKE_STATS(kInt64, Int64Type);
    MAKE_STATS(kFloat, FloatType);
    MAKE_STATS(kDouble, DoubleType);
    MAKE_STATS(kByteArray, ByteArrayType);
    MAKE_STATS(kFixedLenByteArray, FLBAType);
    default:
      break;
  }
#undef MAKE_STATS
  VELOX_DCHECK(false, "Cannot reach here");
  return nullptr;
}

} // namespace facebook::velox::parquet::arrow
