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

#include "velox/functions/sparksql/aggregates/ApproxCountDistinctForIntervalsAggregate.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/VectorReaders.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/lib/HllAccumulator.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

constexpr double kDefaultRelativeSD = 0.05;
constexpr uint64_t kXxHash64Seed = 42;

class SparkXxHash64 final {
 public:
  static uint64_t hashInt32(int32_t input, uint64_t seed) {
    int64_t hash = static_cast<int64_t>(seed + kPrime64_5 + 4L);
    hash ^= static_cast<int64_t>((static_cast<uint32_t>(input)) * kPrime64_1);
    hash = bits::rotateLeft64(hash, 23) * kPrime64_2 + kPrime64_3;
    return fmix(hash);
  }

  static uint64_t hashInt64(int64_t input, uint64_t seed) {
    int64_t hash = static_cast<int64_t>(seed + kPrime64_5 + 8L);
    hash ^= bits::rotateLeft64(input * kPrime64_2, 31) * kPrime64_1;
    hash = bits::rotateLeft64(hash, 27) * kPrime64_1 + kPrime64_4;
    return fmix(hash);
  }

  static uint64_t hashFloat(float input, uint64_t seed) {
    if (input == 0.0f) {
      input = 0.0f;
    }
    uint32_t bits = 0;
    if (std::isnan(input)) {
      bits = 0x7fc00000U;
    } else {
      bits = *reinterpret_cast<uint32_t*>(&input);
    }
    return hashInt32(static_cast<int32_t>(bits), seed);
  }

  static uint64_t hashDouble(double input, uint64_t seed) {
    if (input == 0.0) {
      input = 0.0;
    }
    uint64_t bits = 0;
    if (std::isnan(input)) {
      bits = 0x7ff8000000000000ULL;
    } else {
      bits = *reinterpret_cast<uint64_t*>(&input);
    }
    return hashInt64(static_cast<int64_t>(bits), seed);
  }

  static uint64_t hashBytes(const StringView& input, uint64_t seed) {
    const char* const end = input.data() + input.size();
    uint64_t hash = hashBytesByWords(input, seed);
    auto length = static_cast<uint32_t>(input.size());
    const char* offset = input.data() + (length & -8);
    if (offset + 4L <= end) {
      hash ^= (*reinterpret_cast<const uint64_t*>(offset) & 0xFFFFFFFFL) *
          kPrime64_1;
      hash = bits::rotateLeft64(hash, 23) * kPrime64_2 + kPrime64_3;
      offset += 4L;
    }

    while (offset < end) {
      hash ^= (*reinterpret_cast<const uint64_t*>(offset) & 0xFFL) * kPrime64_5;
      hash = bits::rotateLeft64(hash, 11) * kPrime64_1;
      ++offset;
    }
    return fmix(hash);
  }

  static uint64_t hashDecimal(int128_t input, uint64_t seed) {
    char out[sizeof(int128_t)];
    auto length = DecimalUtil::toByteArray(input, out);
    return hashBytes(StringView(out, length), seed);
  }

 private:
  static constexpr uint64_t kPrime64_1 = 0x9E3779B185EBCA87L;
  static constexpr uint64_t kPrime64_2 = 0xC2B2AE3D27D4EB4FL;
  static constexpr uint64_t kPrime64_3 = 0x165667B19E3779F9L;
  static constexpr uint64_t kPrime64_4 = 0x85EBCA77C2B2AE63L;
  static constexpr uint64_t kPrime64_5 = 0x27D4EB2F165667C5L;

  static uint64_t fmix(uint64_t hash) {
    hash ^= hash >> 33;
    hash *= kPrime64_2;
    hash ^= hash >> 29;
    hash *= kPrime64_3;
    hash ^= hash >> 32;
    return hash;
  }

  static uint64_t hashBytesByWords(const StringView& input, uint64_t seed) {
    const char* i = input.data();
    const char* const end = input.data() + input.size();
    auto length = static_cast<uint32_t>(input.size());
    uint64_t hash;
    if (length >= 32) {
      uint64_t v1 = seed + kPrime64_1 + kPrime64_2;
      uint64_t v2 = seed + kPrime64_2;
      uint64_t v3 = seed;
      uint64_t v4 = seed - kPrime64_1;
      const char* const limit = end - 32;
      while (i <= limit) {
        v1 = bits::rotateLeft64(
                 v1 + *reinterpret_cast<const uint64_t*>(i) * kPrime64_2, 31) *
            kPrime64_1;
        i += 8;
        v2 = bits::rotateLeft64(
                 v2 + *reinterpret_cast<const uint64_t*>(i) * kPrime64_2, 31) *
            kPrime64_1;
        i += 8;
        v3 = bits::rotateLeft64(
                 v3 + *reinterpret_cast<const uint64_t*>(i) * kPrime64_2, 31) *
            kPrime64_1;
        i += 8;
        v4 = bits::rotateLeft64(
                 v4 + *reinterpret_cast<const uint64_t*>(i) * kPrime64_2, 31) *
            kPrime64_1;
        i += 8;
      }
      hash = bits::rotateLeft64(v1, 1) + bits::rotateLeft64(v2, 7) +
          bits::rotateLeft64(v3, 12) + bits::rotateLeft64(v4, 18);
      v1 *= kPrime64_2;
      v1 = bits::rotateLeft64(v1, 31);
      v1 *= kPrime64_1;
      hash ^= v1;
      hash = hash * kPrime64_1 + kPrime64_4;
      v2 *= kPrime64_2;
      v2 = bits::rotateLeft64(v2, 31);
      v2 *= kPrime64_1;
      hash ^= v2;
      hash = hash * kPrime64_1 + kPrime64_4;
      v3 *= kPrime64_2;
      v3 = bits::rotateLeft64(v3, 31);
      v3 *= kPrime64_1;
      hash ^= v3;
      hash = hash * kPrime64_1 + kPrime64_4;
      v4 *= kPrime64_2;
      v4 = bits::rotateLeft64(v4, 31);
      v4 *= kPrime64_1;
      hash ^= v4;
      hash = hash * kPrime64_1 + kPrime64_4;
    } else {
      hash = seed + kPrime64_5;
    }
    return hash + length;
  }
};

int8_t computeIndexBitLength(double relativeSD) {
  VELOX_USER_CHECK(
      std::isfinite(relativeSD) && relativeSD > 0.0,
      "relativeSD must be a positive finite value");
  const double p =
      std::ceil(2.0 * std::log(1.106 / relativeSD) / std::log(2.0));
  VELOX_USER_CHECK(
      p >= 4,
      "HLL++ requires at least 4 bits for addressing. Use a lower error, "
      "at most 39%.");
  VELOX_USER_CHECK(
      p <= 16,
      "HLL++ requires at most 16 bits for addressing. Use a higher error.");
  return static_cast<int8_t>(p);
}

inline double decimalToDouble(int128_t value, uint8_t scale) {
  long double scaled = static_cast<long double>(value);
  long double divisor =
      static_cast<long double>(DecimalUtil::kPowersOfTen[scale]);
  return static_cast<double>(scaled / divisor);
}

class ApproxCountDistinctForIntervalsAggregate : public exec::Aggregate {
 public:
  explicit ApproxCountDistinctForIntervalsAggregate(
      const TypePtr& resultType,
      const std::vector<TypePtr>& argTypes)
      : exec::Aggregate(resultType) {
    if (argTypes.size() >= 2) {
      inputType_ = argTypes[0];
      endpointsElementType_ = argTypes[1]->childAt(0);
    }
  }

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(Accumulator);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(Accumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    VELOX_CHECK_NOT_NULL(inputType_);
    VELOX_CHECK_NOT_NULL(endpointsElementType_);
    ensureRelativeSd(args);
    ensureEndpointsFromRaw(args);
    if (!endpointsSet_) {
      return;
    }

    decodedValues_.decode(*args[0], rows);
    const auto mayHaveNulls = decodedValues_.mayHaveNulls();
    rows.applyToSelected([&](auto row) {
      if (mayHaveNulls && decodedValues_.isNullAt(row)) {
        return;
      }

      const double inputValue = toDouble(decodedValues_, row, inputType_);
      if (inputValue < endpointsMin_ || inputValue > endpointsMax_) {
        return;
      }

      const auto intervalIndex = std::isnan(inputValue)
          ? (intervalCount_ - 1)
          : findIntervalIndex(inputValue);
      auto* group = groups[row];
      auto tracker = trackRowSize(group);
      auto* accumulator = value<Accumulator>(group);
      accumulator->ensureSize(allocator_, intervalCount_, indexBitLength_);
      const uint64_t hash = hashValue(decodedValues_, row, inputType_);
      accumulator->hlls[intervalIndex].insertHash(hash);
      clearNull(group);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    VELOX_CHECK_NOT_NULL(inputType_);
    VELOX_CHECK_NOT_NULL(endpointsElementType_);
    ensureRelativeSd(args);
    ensureEndpointsFromRaw(args);
    if (!endpointsSet_) {
      return;
    }

    decodedValues_.decode(*args[0], rows);
    const auto mayHaveNulls = decodedValues_.mayHaveNulls();
    auto tracker = trackRowSize(group);
    auto* accumulator = value<Accumulator>(group);
    rows.applyToSelected([&](auto row) {
      if (mayHaveNulls && decodedValues_.isNullAt(row)) {
        return;
      }
      const double inputValue = toDouble(decodedValues_, row, inputType_);
      if (inputValue < endpointsMin_ || inputValue > endpointsMax_) {
        return;
      }
      const auto intervalIndex = std::isnan(inputValue)
          ? (intervalCount_ - 1)
          : findIntervalIndex(inputValue);
      accumulator->ensureSize(allocator_, intervalCount_, indexBitLength_);
      const uint64_t hash = hashValue(decodedValues_, row, inputType_);
      accumulator->hlls[intervalIndex].insertHash(hash);
      clearNull(group);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);
    exec::VectorReader<Row<Array<double>, Array<Varbinary>>> reader(
        &decodedIntermediate_);

    rows.applyToSelected([&](auto row) {
      if (!reader.isSet(row)) {
        return;
      }
      auto rowView = reader[row];
      auto endpointsView = rowView.template at<0>();
      auto hllsView = rowView.template at<1>();
      if (!endpointsView.has_value() || !hllsView.has_value()) {
        return;
      }
      ensureEndpointsFromIntermediate(endpointsView.value());
      if (!endpointsSet_) {
        return;
      }

      const auto& hllsArray = hllsView.value();
      VELOX_USER_CHECK_EQ(
          hllsArray.size(),
          intervalCount_,
          "HLL array size {} does not match endpoints size {}",
          hllsArray.size(),
          intervalCount_);

      for (const auto& entry : hllsArray) {
        if (entry.has_value()) {
          maybeSetIndexBitLengthFromSerialized(entry.value());
          break;
        }
      }

      auto* group = groups[row];
      auto tracker = trackRowSize(group);
      auto* accumulator = value<Accumulator>(group);
      accumulator->ensureSize(allocator_, intervalCount_, indexBitLength_);

      for (size_t i = 0; i < hllsArray.size(); ++i) {
        const auto& entry = hllsArray[i];
        if (!entry.has_value()) {
          continue;
        }
        accumulator->hlls[i].mergeWith(entry.value(), allocator_);
      }
      clearNull(group);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);
    exec::VectorReader<Row<Array<double>, Array<Varbinary>>> reader(
        &decodedIntermediate_);
    auto tracker = trackRowSize(group);
    auto* accumulator = value<Accumulator>(group);

    rows.applyToSelected([&](auto row) {
      if (!reader.isSet(row)) {
        return;
      }
      auto rowView = reader[row];
      auto endpointsView = rowView.template at<0>();
      auto hllsView = rowView.template at<1>();
      if (!endpointsView.has_value() || !hllsView.has_value()) {
        return;
      }
      ensureEndpointsFromIntermediate(endpointsView.value());
      if (!endpointsSet_) {
        return;
      }
      const auto& hllsArray = hllsView.value();
      VELOX_USER_CHECK_EQ(
          hllsArray.size(),
          intervalCount_,
          "HLL array size {} does not match endpoints size {}",
          hllsArray.size(),
          intervalCount_);

      for (const auto& entry : hllsArray) {
        if (entry.has_value()) {
          maybeSetIndexBitLengthFromSerialized(entry.value());
          break;
        }
      }

      accumulator->ensureSize(allocator_, intervalCount_, indexBitLength_);
      for (size_t i = 0; i < hllsArray.size(); ++i) {
        const auto& entry = hllsArray[i];
        if (!entry.has_value()) {
          continue;
        }
        accumulator->hlls[i].mergeWith(entry.value(), allocator_);
      }
      clearNull(group);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    auto arrayVector = (*result)->asUnchecked<ArrayVector>();
    arrayVector->resize(numGroups);

    exec::VectorWriter<Array<int64_t>> writer;
    writer.init(*arrayVector);

    for (auto i = 0; i < numGroups; ++i) {
      writer.setOffset(i);
      if (!endpointsSet_ || indexBitLength_ < 0) {
        writer.commitNull();
        continue;
      }
      auto& arrayWriter = writer.current();
      auto* accumulator = value<Accumulator>(groups[i]);

      for (int32_t interval = 0; interval < intervalCount_; ++interval) {
        int64_t count = 0;
        if (accumulator->hlls.size() == intervalCount_) {
          count = accumulator->hlls[interval].cardinality();
        }
        if (duplicateIntervals_[interval]) {
          count = 1;
        }
        arrayWriter.add_item() = count;
      }
      writer.commit();
    }

    writer.finish();
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    auto rowVector = (*result)->asUnchecked<RowVector>();
    rowVector->resize(numGroups);

    exec::VectorWriter<Row<Array<double>, Array<Varbinary>>> writer;
    writer.init(*rowVector);

    for (auto i = 0; i < numGroups; ++i) {
      writer.setOffset(i);
      if (!endpointsSet_) {
        writer.commitNull();
        continue;
      }
      auto& rowWriter = writer.current();
      auto& endpointsWriter = rowWriter.template get_writer_at<0>();
      for (double endpoint : endpoints_) {
        endpointsWriter.add_item() = endpoint;
      }

      auto& hllsWriter = rowWriter.template get_writer_at<1>();
      auto* accumulator = value<Accumulator>(groups[i]);
      ensureEmptyHll();
      for (int32_t interval = 0; interval < intervalCount_; ++interval) {
        auto& itemWriter = hllsWriter.add_item();
        if (accumulator->hlls.size() == intervalCount_) {
          auto& hll = accumulator->hlls[interval];
          const auto size = hll.serializedSize();
          std::string buffer(size, '\0');
          hll.serialize(buffer.data());
          itemWriter.append(std::string_view(buffer));
        } else {
          itemWriter.append(std::string_view(emptyHll_));
        }
      }
      writer.commit();
    }

    writer.finish();
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) Accumulator();
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<Accumulator>(groups);
  }

 private:
  struct Accumulator {
    std::vector<
        common::hll::HllAccumulator<int64_t, false, HashStringAllocator>>
        hlls;

    void ensureSize(
        HashStringAllocator* allocator,
        int32_t targetSize,
        int8_t indexBitLength) {
      if (hlls.empty()) {
        hlls.reserve(targetSize);
        for (int32_t i = 0; i < targetSize; ++i) {
          if (indexBitLength >= 0) {
            hlls.emplace_back(indexBitLength, allocator);
          } else {
            hlls.emplace_back(allocator);
          }
        }
        return;
      }
      VELOX_USER_CHECK_EQ(
          hlls.size(),
          targetSize,
          "HLL size {} does not match endpoints size {}",
          hlls.size(),
          targetSize);
    }
  };

  void ensureRelativeSd(const std::vector<VectorPtr>& args) {
    if (indexBitLength_ >= 0) {
      return;
    }
    double relativeSD = kDefaultRelativeSD;
    if (args.size() >= 3) {
      SelectivityVector rows(args[2]->size());
      rows.setAll();
      DecodedVector decoded;
      decoded.decode(*args[2], rows);
      VELOX_USER_CHECK(
          decoded.isConstantMapping(),
          "relativeSD must be constant for "
          "approx_count_distinct_for_intervals");
      VELOX_USER_CHECK(
          !decoded.isNullAt(decoded.index(0)),
          "relativeSD must not be null for approx_count_distinct_for_intervals");
      relativeSD = decoded.valueAt<double>(decoded.index(0));
    }
    indexBitLength_ = computeIndexBitLength(relativeSD);
    ensureEmptyHll();
  }

  void ensureEndpointsFromRaw(const std::vector<VectorPtr>& args) {
    if (endpointsSet_ || args.size() < 2) {
      return;
    }

    SelectivityVector rows(args[1]->size());
    rows.setAll();
    decodedEndpoints_.decode(*args[1], rows);

    VELOX_USER_CHECK(
        decodedEndpoints_.isConstantMapping(),
        "Endpoints must be constant for approx_count_distinct_for_intervals");
    VELOX_USER_CHECK(
        !decodedEndpoints_.isNullAt(0),
        "Endpoints must not be null for approx_count_distinct_for_intervals");

    const auto index = decodedEndpoints_.index(0);
    auto* arrayVector = decodedEndpoints_.base()->as<ArrayVector>();
    const auto offset = arrayVector->offsetAt(index);
    const auto size = arrayVector->sizeAt(index);
    VELOX_USER_CHECK_GE(
        size,
        2,
        "approx_count_distinct_for_intervals requires at least 2 endpoints");

    auto elements = arrayVector->elements();
    SelectivityVector elementRows(elements->size());
    elementRows.setAll();
    decodedEndpointElements_.decode(*elements, elementRows);

    std::vector<double> endpoints;
    endpoints.reserve(size);
    for (vector_size_t i = 0; i < size; ++i) {
      const auto elementIndex = offset + i;
      VELOX_USER_CHECK(
          !decodedEndpointElements_.isNullAt(elementIndex),
          "Endpoints must not contain null values");
      endpoints.push_back(toDouble(
          decodedEndpointElements_, elementIndex, endpointsElementType_));
    }
    setEndpoints(endpoints);
  }

  void ensureEndpointsFromIntermediate(
      const exec::ArrayView<true, double>& endpointsView) {
    if (endpointsSet_) {
      return;
    }
    VELOX_USER_CHECK_GE(
        endpointsView.size(),
        2,
        "approx_count_distinct_for_intervals requires at least 2 endpoints");
    std::vector<double> endpoints;
    endpoints.reserve(endpointsView.size());
    for (const auto& entry : endpointsView) {
      VELOX_USER_CHECK(
          entry.has_value(), "Endpoints must not contain null values");
      endpoints.push_back(entry.value());
    }
    setEndpoints(endpoints);
  }

  void setEndpoints(const std::vector<double>& endpoints) {
    if (endpointsSet_) {
      VELOX_USER_CHECK_EQ(
          endpoints_.size(),
          endpoints.size(),
          "Endpoints size {} does not match existing size {}",
          endpoints.size(),
          endpoints_.size());
      for (size_t i = 0; i < endpoints.size(); ++i) {
        VELOX_USER_CHECK(
            endpoints_[i] == endpoints[i], "Endpoints mismatch at index {}", i);
      }
      return;
    }

    endpoints_ = endpoints;
    endpointsMin_ = endpoints_.front();
    endpointsMax_ = endpoints_.back();
    intervalCount_ = static_cast<int32_t>(endpoints_.size() - 1);
    duplicateIntervals_.resize(intervalCount_);
    for (int32_t i = 0; i < intervalCount_; ++i) {
      duplicateIntervals_[i] = (endpoints_[i] == endpoints_[i + 1]);
    }
    endpointsSet_ = true;
  }

  void ensureEmptyHll() {
    if (!emptyHllInitialized_ && indexBitLength_ >= 0) {
      emptyHll_ = common::hll::SparseHlls::serializeEmpty(indexBitLength_);
      emptyHllInitialized_ = true;
    }
  }

  void maybeSetIndexBitLengthFromSerialized(const StringView& serialized) {
    if (indexBitLength_ >= 0) {
      return;
    }
    const char* data = serialized.data();
    if (common::hll::SparseHlls::canDeserialize(data)) {
      indexBitLength_ =
          common::hll::SparseHlls::deserializeIndexBitLength(data);
    } else if (common::hll::DenseHlls::canDeserialize(data)) {
      indexBitLength_ = common::hll::DenseHlls::deserializeIndexBitLength(data);
    }
    ensureEmptyHll();
  }

  static double toDouble(
      const DecodedVector& decoded,
      vector_size_t row,
      const TypePtr& type) {
    if (type->isDate()) {
      return static_cast<double>(decoded.valueAt<int32_t>(row));
    }
    if (type->isIntervalYearMonth()) {
      return static_cast<double>(decoded.valueAt<int32_t>(row));
    }
    if (type->isIntervalDayTime()) {
      return static_cast<double>(decoded.valueAt<int64_t>(row));
    }
    if (type->isShortDecimal()) {
      auto value = decoded.valueAt<int64_t>(row);
      auto scale = type->asShortDecimal().scale();
      return decimalToDouble(static_cast<int128_t>(value), scale);
    }
    if (type->isLongDecimal()) {
      auto value = decoded.valueAt<int128_t>(row);
      auto scale = type->asLongDecimal().scale();
      return decimalToDouble(value, scale);
    }

    switch (type->kind()) {
      case TypeKind::TINYINT:
        return static_cast<double>(decoded.valueAt<int8_t>(row));
      case TypeKind::SMALLINT:
        return static_cast<double>(decoded.valueAt<int16_t>(row));
      case TypeKind::INTEGER:
        return static_cast<double>(decoded.valueAt<int32_t>(row));
      case TypeKind::BIGINT:
        return static_cast<double>(decoded.valueAt<int64_t>(row));
      case TypeKind::REAL:
        return static_cast<double>(decoded.valueAt<float>(row));
      case TypeKind::DOUBLE:
        return decoded.valueAt<double>(row);
      case TypeKind::TIMESTAMP:
        return static_cast<double>(decoded.valueAt<Timestamp>(row).toMicros());
      case TypeKind::HUGEINT:
        return static_cast<double>(decoded.valueAt<int128_t>(row));
      default:
        VELOX_UNSUPPORTED(
            "Unsupported type for approx_count_distinct_for_intervals: {}",
            type->toString());
    }
  }

  static uint64_t hashValue(
      const DecodedVector& decoded,
      vector_size_t row,
      const TypePtr& type) {
    if (type->isDate()) {
      return SparkXxHash64::hashInt32(
          decoded.valueAt<int32_t>(row), kXxHash64Seed);
    }
    if (type->isIntervalYearMonth()) {
      return SparkXxHash64::hashInt32(
          decoded.valueAt<int32_t>(row), kXxHash64Seed);
    }
    if (type->isIntervalDayTime()) {
      return SparkXxHash64::hashInt64(
          decoded.valueAt<int64_t>(row), kXxHash64Seed);
    }
    if (type->isShortDecimal()) {
      auto value = decoded.valueAt<int64_t>(row);
      return SparkXxHash64::hashDecimal(
          static_cast<int128_t>(value), kXxHash64Seed);
    }
    if (type->isLongDecimal()) {
      auto value = decoded.valueAt<int128_t>(row);
      return SparkXxHash64::hashDecimal(value, kXxHash64Seed);
    }

    switch (type->kind()) {
      case TypeKind::TINYINT:
        return SparkXxHash64::hashInt32(
            static_cast<int32_t>(decoded.valueAt<int8_t>(row)), kXxHash64Seed);
      case TypeKind::SMALLINT:
        return SparkXxHash64::hashInt32(
            static_cast<int32_t>(decoded.valueAt<int16_t>(row)), kXxHash64Seed);
      case TypeKind::INTEGER:
        return SparkXxHash64::hashInt32(
            decoded.valueAt<int32_t>(row), kXxHash64Seed);
      case TypeKind::BIGINT:
        return SparkXxHash64::hashInt64(
            decoded.valueAt<int64_t>(row), kXxHash64Seed);
      case TypeKind::REAL:
        return SparkXxHash64::hashFloat(
            decoded.valueAt<float>(row), kXxHash64Seed);
      case TypeKind::DOUBLE:
        return SparkXxHash64::hashDouble(
            decoded.valueAt<double>(row), kXxHash64Seed);
      case TypeKind::TIMESTAMP:
        return SparkXxHash64::hashInt64(
            decoded.valueAt<Timestamp>(row).toMicros(), kXxHash64Seed);
      case TypeKind::HUGEINT:
        return SparkXxHash64::hashDecimal(
            decoded.valueAt<int128_t>(row), kXxHash64Seed);
      default:
        VELOX_UNSUPPORTED(
            "Unsupported type for approx_count_distinct_for_intervals: {}",
            type->toString());
    }
  }

  int32_t findIntervalIndex(double value) const {
    auto it = std::lower_bound(endpoints_.begin(), endpoints_.end(), value);
    if (it != endpoints_.end() && *it == value) {
      auto index = static_cast<int32_t>(it - endpoints_.begin());
      while (index > 0 && endpoints_[index - 1] == value) {
        --index;
      }
      return index == 0 ? 0 : index - 1;
    }
    auto insertionPoint = static_cast<int32_t>(it - endpoints_.begin());
    return insertionPoint == 0 ? 0 : insertionPoint - 1;
  }

  TypePtr inputType_;
  TypePtr endpointsElementType_;
  bool endpointsSet_{false};
  std::vector<double> endpoints_;
  std::vector<char> duplicateIntervals_;
  int32_t intervalCount_{0};
  double endpointsMin_{0};
  double endpointsMax_{0};
  int8_t indexBitLength_{-1};
  bool emptyHllInitialized_{false};
  std::string emptyHll_;

  DecodedVector decodedValues_;
  DecodedVector decodedEndpoints_;
  DecodedVector decodedEndpointElements_;
  DecodedVector decodedIntermediate_;
};

exec::AggregateRegistrationResult registerApproxCountDistinctForIntervals(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  const auto returnType = "array(bigint)";
  const auto intermediateType = "row(array(double), array(varbinary))";

  const std::vector<std::string> valueTypes = {
      "tinyint",
      "smallint",
      "integer",
      "bigint",
      "real",
      "double",
      "date",
      "timestamp",
      "interval day to second",
      "interval year to month"};

  const std::vector<std::string> endpointTypes = valueTypes;

  auto addSignature = [&](exec::AggregateFunctionSignatureBuilder builder) {
    auto withRelative = builder;
    signatures.push_back(builder.build());
    withRelative.argumentType("double");
    signatures.push_back(withRelative.build());
  };

  for (const auto& valueType : valueTypes) {
    for (const auto& endpointType : endpointTypes) {
      addSignature(
          exec::AggregateFunctionSignatureBuilder()
              .returnType(returnType)
              .intermediateType(intermediateType)
              .argumentType(valueType)
              .argumentType(fmt::format("array({})", endpointType)));
    }
    addSignature(
        exec::AggregateFunctionSignatureBuilder()
            .integerVariable("b_precision")
            .integerVariable("b_scale")
            .returnType(returnType)
            .intermediateType(intermediateType)
            .argumentType(valueType)
            .argumentType("array(DECIMAL(b_precision, b_scale))"));
  }

  for (const auto& endpointType : endpointTypes) {
    addSignature(
        exec::AggregateFunctionSignatureBuilder()
            .integerVariable("a_precision")
            .integerVariable("a_scale")
            .returnType(returnType)
            .intermediateType(intermediateType)
            .argumentType("DECIMAL(a_precision, a_scale)")
            .argumentType(fmt::format("array({})", endpointType)));
  }

  addSignature(
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("b_precision")
          .integerVariable("b_scale")
          .returnType(returnType)
          .intermediateType(intermediateType)
          .argumentType("DECIMAL(a_precision, a_scale)")
          .argumentType("array(DECIMAL(b_precision, b_scale))"));

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK(
            argTypes.size() == 1 || argTypes.size() == 2 ||
                argTypes.size() == 3,
            "{} takes 2 or 3 arguments",
            name);
        return std::make_unique<ApproxCountDistinctForIntervalsAggregate>(
            resultType, argTypes);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerApproxCountDistinctForIntervalsAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerApproxCountDistinctForIntervals(
      prefix + "approx_count_distinct_for_intervals",
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
