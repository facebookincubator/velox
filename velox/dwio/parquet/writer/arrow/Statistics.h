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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

namespace arrow {

class Array;
class BinaryArray;

} // namespace arrow

namespace facebook::velox::parquet::arrow {

class ColumnDescriptor;

// ----------------------------------------------------------------------.
// Value Comparator interfaces.

/// \brief Base class for value Comparators. Generally used with.
/// TypedComparator<T>.
class PARQUET_EXPORT Comparator {
 public:
  virtual ~Comparator() {}

  /// \brief Create a Comparator explicitly from physical type and.
  /// Sort order.
  /// \param[in] physical_type the physical type for the typed.
  /// Comparator.
  /// \param[in] sort_order either SortOrder::kSigned or.
  /// SortOrder::kUnsigned.
  /// \param[in] type_length for FIXED_LEN_BYTE_ARRAY only.
  static std::shared_ptr<Comparator>
  make(Type::type physicalType, SortOrder::type sortOrder, int typeLength = -1);

  /// \brief Create typed Comparator inferring default sort order from.
  /// ColumnDescriptor.
  /// \param[in] descr the Parquet column schema.
  static std::shared_ptr<Comparator> make(const ColumnDescriptor* descr);
};

/// \brief Interface for comparison of physical types according to the.
/// Semantics of a particular logical type.
template <typename DType>
class TypedComparator : public Comparator {
 public:
  using T = typename DType::CType;

  /// \brief Scalar comparison of two elements, return true if first.
  /// Is strictly less than the second.
  virtual bool compare(const T& a, const T& b) = 0;

  /// \brief Compute maximum and minimum elements in a batch of.
  /// Elements without any nulls.
  virtual std::pair<T, T> getMinMax(const T* values, int64_t length) = 0;

  /// \brief Compute minimum and maximum elements from an Arrow array. Only.
  /// Valid for certain Parquet Type / Arrow Type combinations, like BYTE_ARRAY.
  /// / Arrow::BinaryArray.
  virtual std::pair<T, T> getMinMax(const ::arrow::Array& values) = 0;

  /// \brief Compute maximum and minimum elements in a batch of.
  /// Elements with accompanying bitmap indicating which elements are.
  /// included (bit set) and excluded (bit not set)
  ///
  /// \param[in] values the sequence of values.
  /// \param[in] length the length of the sequence.
  /// \param[in] valid_bits a bitmap indicating which elements are.
  /// included (1) or excluded (0)
  /// \param[in] valid_bits_offset the bit offset into the bitmap of.
  /// The first element in the sequence.
  virtual std::pair<T, T> getMinMaxSpaced(
      const T* values,
      int64_t length,
      const uint8_t* validBits,
      int64_t validBitsOffset) = 0;
};

/// \brief Typed version of Comparator::Make.
template <typename DType>
std::shared_ptr<TypedComparator<DType>> makeComparator(
    Type::type physicalType,
    SortOrder::type sortOrder,
    int typeLength = -1) {
  return std::static_pointer_cast<TypedComparator<DType>>(
      Comparator::make(physicalType, sortOrder, typeLength));
}

/// \brief Typed version of Comparator::Make.
template <typename DType>
std::shared_ptr<TypedComparator<DType>> makeComparator(
    const ColumnDescriptor* descr) {
  return std::static_pointer_cast<TypedComparator<DType>>(
      Comparator::make(descr));
}

// ----------------------------------------------------------------------.

/// \brief Structure represented encoded statistics to be written to.
/// And read from Parquet serialized metadata.
class PARQUET_EXPORT EncodedStatistics {
  std::string max_, min_;
  bool isSigned_ = false;

 public:
  EncodedStatistics() = default;

  const std::string& max() const {
    return max_;
  }
  const std::string& min() const {
    return min_;
  }

  int64_t nullCount = 0;
  int64_t distinctCount = 0;
  int64_t nanCount = 0;

  bool hasMin = false;
  bool hasMax = false;
  bool hasNullCount = false;
  bool hasDistinctCount = false;
  bool hasNanCount = false;

  // When all values in the statistics are null, it is set to true.
  // Otherwise, at least one value is not null, or we are not sure at all.
  // Page index requires this information to decide whether a data page.
  // Is a null page or not.
  bool allNullValue = false;

  // From parquet-mr.
  // Don't write stats larger than the max size rather than truncating. The.
  // Rationale is that some engines may use the minimum value in the page as.
  // The true minimum for aggregations and there is no way to mark that a.
  // Value has been truncated and is a lower bound and not in the page.
  void applyStatSizeLimits(size_t length) {
    if (max_.length() > length) {
      hasMax = false;
      max_.clear();
    }
    if (min_.length() > length) {
      hasMin = false;
      min_.clear();
    }
  }

  bool isSet() const {
    return hasMin || hasMax || hasNullCount || hasDistinctCount;
  }

  bool isSigned() const {
    return isSigned_;
  }

  void setIsSigned(bool isSigned) {
    isSigned_ = isSigned;
  }

  EncodedStatistics& setMax(std::string value) {
    max_ = std::move(value);
    hasMax = true;
    return *this;
  }

  EncodedStatistics& setMin(std::string value) {
    min_ = std::move(value);
    hasMin = true;
    return *this;
  }

  EncodedStatistics& setNullCount(int64_t value) {
    nullCount = value;
    hasNullCount = true;
    return *this;
  }

  EncodedStatistics& setDistinctCount(int64_t value) {
    distinctCount = value;
    hasDistinctCount = true;
    return *this;
  }

  EncodedStatistics& set_nan_count(int64_t value) {
    nanCount = value;
    hasNanCount = true;
    return *this;
  }
};

/// \brief Base type for computing column statistics while writing a file.
class PARQUET_EXPORT Statistics {
 public:
  virtual ~Statistics() {}

  /// \brief Create a new statistics instance given a column schema.
  /// Definition.
  /// \param[in] descr the column schema.
  /// \param[in] pool a memory pool to use for any memory allocations, optional.
  static std::shared_ptr<Statistics> make(
      const ColumnDescriptor* descr,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  /// \brief Create a new statistics instance given a column schema.
  /// Definition and pre-existing state.
  /// \param[in] descr the column schema.
  /// \param[in] encodedMin the encoded minimum value.
  /// \param[in] encodedMax the encoded maximum value.
  /// \param[in] numValues total number of values.
  /// \param[in] nullCount number of null values.
  /// \param[in] distinctCount number of distinct values.
  /// \param[in] hasMinMax whether the min/max statistics are set.
  /// \param[in] hasNullCount whether the null_count statistics are set.
  /// \param[in] hasDistinctCount whether the distinct_count statistics are.
  /// Set \param[in] pool a memory pool to use for any memory allocations,.
  /// Optional.
  static std::shared_ptr<Statistics> make(
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
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  // Helper function to convert EncodedStatistics to Statistics.
  // EncodedStatistics does not contain number of non-null values, and it can
  // be. Passed using the num_values parameter.
  static std::shared_ptr<Statistics> make(
      const ColumnDescriptor* descr,
      const EncodedStatistics* encodedStats,
      int64_t numValues = -1,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  /// \brief Return true if the count of null values is set.
  virtual bool hasNullCount() const = 0;

  /// \brief The number of null values, may not be set.
  virtual int64_t nullCount() const = 0;

  /// \brief Return true if the count of distinct values is set.
  virtual bool hasDistinctCount() const = 0;

  /// \brief The number of distinct values, may not be set.
  virtual int64_t distinctCount() const = 0;

  /// \brief The number of non-null values in the column.
  virtual int64_t numValues() const = 0;

  /// \brief Return true if the count of nan values is set
  virtual bool hasNaNCount() const = 0;

  /// \brief The number of NaN values, may not be set
  virtual int64_t nanCount() const = 0;

  /// \brief Return true if the min and max statistics are set. Obtain.
  /// With TypedStatistics<T>::min and max.
  virtual bool hasMinMax() const = 0;

  /// \brief Reset state of object to initial (no data observed) state.
  virtual void reset() = 0;

  /// \brief Plain-encoded minimum value.
  virtual std::string encodeMin() const = 0;

  /// \brief Plain-encoded maximum value.
  virtual std::string encodeMax() const = 0;

  /// \brief Encoded lower bound value compatible with Iceberg.
  ///
  /// Returns an encoded value guaranteed to be <= the actual minimum value.
  /// For string types, truncates to at most \p truncateTo Unicode code
  /// points. For decimal types, encodes the value in big-endian format as
  /// required by. Iceberg's single-value serialization specification. For
  /// all other data types, uses the same plain encoding as Parquet.
  /// (Returns the exact encoded minimum value).
  ///
  /// @param truncateTo Maximum number of Unicode code points for string
  /// types.
  virtual std::string icebergLowerBoundInclusive(int32_t truncateTo) const = 0;

  /// \brief Encoded upper bound value compatible with Iceberg.
  ///
  /// Returns an encoded value guaranteed to be >= the actual maximum value.
  /// For string types:
  /// - If the maximum value has <= \p truncateTo Unicode code points,
  /// returns.
  ///   The exact encoded maximum value (inclusive upper bound).
  /// - If the maximum value has > \p truncateTo Unicode code points,
  /// truncates.
  ///   To \p truncateTo code points and increments the last code point to.
  ///   Produce an exclusive upper bound that is greater than the maximum
  ///   value.
  /// - Returns std::nullopt if no valid upper bound can be computed (e.g.,.
  ///   All code points in the truncated portion are at the maximum Unicode.
  ///   Value U+10FFFF). This allows distinguishing between "upper bound is.
  ///   Empty string" and "no valid upper bound exists".
  /// For decimal types, encodes the value in big-endian format as required
  /// by. Iceberg's single-value serialization specification. For all other
  /// data types, uses the same plain encoding as Parquet. (Returns the
  /// exact encoded maximum value).
  ///
  /// @param truncateTo Maximum number of Unicode code points for string
  /// types.
  /// @return Encoded upper bound value, or std::nullopt if no valid upper.
  ///         Bound can be computed.
  virtual std::optional<std::string> icebergUpperBoundExclusive(
      int32_t truncateTo) const = 0;

  /// \brief The finalized encoded form of the statistics for transport.
  virtual EncodedStatistics encode() = 0;

  /// \brief The physical type of the column schema.
  virtual Type::type physicalType() const = 0;

  /// \brief The full type descriptor from the column schema.
  virtual const ColumnDescriptor* descr() const = 0;

  /// \brief Check two Statistics for equality.
  virtual bool equals(const Statistics& other) const = 0;

  /// \brief Return true if this object's max is greater than the other's
  /// max.
  /// \param[in] other the Statistics object to compare against.
  virtual bool maxGreaterThan(const Statistics& other) const = 0;

  /// \brief Return true if this object's min is less than the other's min.
  /// \param[in] other the Statistics object to compare against.
  virtual bool minLessThan(const Statistics& other) const = 0;

 protected:
  static std::shared_ptr<Statistics> make(
      Type::type physicalType,
      const void* min,
      const void* max,
      int64_t numValues,
      int64_t nullCount,
      int64_t distinctCount);
};

/// \brief A typed implementation of Statistics.
template <typename DType>
class TypedStatistics : public Statistics {
 public:
  using T = typename DType::CType;

  /// \brief The current minimum value.
  virtual const T& min() const = 0;

  /// \brief The current maximum value.
  virtual const T& max() const = 0;

  /// \brief Update state with state of another Statistics object.
  virtual void merge(const TypedStatistics<DType>& other) = 0;

  /// \brief Batch statistics update.
  virtual void
  update(const T* values, int64_t numValues, int64_t nullCount) = 0;

  /// \brief Batch statistics update with supplied validity bitmap.
  /// \param[in] values pointer to column values.
  /// \param[in] valid_bits Pointer to bitmap representing if values are.
  /// Non-null. \param[in] valid_bits_offset Offset offset into valid_bits
  /// where. The slice of.
  ///                              Data begins.
  /// \param[in] num_spaced_values The length of values in values/valid_bits to.
  /// Inspect.
  ///                              When calculating statistics. This can be.
  ///                              Smaller than num_values+null_count as.
  ///                              Null_count can include nulls from parents.
  ///                              While num_spaced_values does not.
  /// \param[in] num_values Number of values that are not null.
  /// \param[in] null_count Number of values that are null.
  virtual void updateSpaced(
      const T* values,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      int64_t numSpacedValues,
      int64_t numValues,
      int64_t nullCount) = 0;

  /// \brief EXPERIMENTAL: Update statistics with an Arrow array without.
  /// Conversion to a primitive Parquet C type. Only implemented for certain.
  /// Parquet type / Arrow type combinations like BYTE_ARRAY /.
  /// Arrow::BinaryArray.
  ///
  /// If update_counts is true then the null_count and num_values will be.
  /// Updated based on the null_count of values.  Set to false if these are.
  /// Updated elsewhere (e.g. when updating a dictionary where the counts are.
  /// taken from the indices and not the values)
  virtual void update(
      const ::arrow::Array& values,
      bool updateCounts = true) = 0;

  /// \brief Set min and max values to particular values.
  virtual void setMinMax(const T& min, const T& max) = 0;

  /// \brief Increments the null count directly.
  /// Use Update to extract the null count from data.  Use this if you
  /// determine. The null count through some other means (e.g. dictionary arrays
  /// where the. null count is determined from the indices)
  virtual void incrementNullCount(int64_t n) = 0;

  /// \brief Increments the number of values directly.
  /// The same note on IncrementNullCount applies here.
  virtual void incrementNumValues(int64_t n) = 0;

  /// \brief Increments the NaN count directly
  virtual void incrementNaNValues(int64_t n) = 0;
};

using BoolStatistics = TypedStatistics<BooleanType>;
using Int32Statistics = TypedStatistics<Int32Type>;
using Int64Statistics = TypedStatistics<Int64Type>;
using FloatStatistics = TypedStatistics<FloatType>;
using DoubleStatistics = TypedStatistics<DoubleType>;
using ByteArrayStatistics = TypedStatistics<ByteArrayType>;
using FLBAStatistics = TypedStatistics<FLBAType>;

/// \brief Typed version of Statistics::Make.
template <typename DType>
std::shared_ptr<TypedStatistics<DType>> makeStatistics(
    const ColumnDescriptor* descr,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  return std::static_pointer_cast<TypedStatistics<DType>>(
      Statistics::make(descr, pool));
}

/// \brief Create Statistics initialized to a particular state.
/// \param[in] min the minimum value.
/// \param[in] max the minimum value.
/// \param[in] numValues number of values.
/// \param[in] nullCount number of null values.
/// \param[in] distinctCount number of distinct values.
template <typename DType>
std::shared_ptr<TypedStatistics<DType>> makeStatistics(
    const typename DType::CType& min,
    const typename DType::CType& max,
    int64_t numValues,
    int64_t nullCount,
    int64_t distinctCount) {
  return std::static_pointer_cast<TypedStatistics<DType>>(Statistics::make(
      DType::typeNum, &min, &max, numValues, nullCount, distinctCount));
}

/// \brief Typed version of Statistics::Make.
template <typename DType>
std::shared_ptr<TypedStatistics<DType>> makeStatistics(
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
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  return std::static_pointer_cast<TypedStatistics<DType>>(Statistics::make(
      descr,
      encodedMin,
      encodedMax,
      numValues,
      nullCount,
      distinctCount,
      hasMinMax,
      hasNullCount,
      hasDistinctCount,
      hasNaNCount,
      nanCount,
      pool));
}

} // namespace facebook::velox::parquet::arrow
