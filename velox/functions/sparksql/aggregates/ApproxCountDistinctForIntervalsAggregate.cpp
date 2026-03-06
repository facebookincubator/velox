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
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/VectorReaders.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/lib/HllAccumulator.h"
#include "velox/functions/sparksql/XxHash64.h"
#include "velox/type/Conversions.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

constexpr uint64_t kXxHash64Seed = 42;

using HllAccumulator =
    common::hll::HllAccumulator<int64_t, false, HashStringAllocator>;

int8_t computeIndexBitLength(double relativeSD) {
  VELOX_USER_CHECK(
      std::isfinite(relativeSD) && relativeSD > 0.0,
      "relativeSD must be a positive finite value");
  const int32_t p = static_cast<int32_t>(
      std::ceil(2.0 * std::log(1.106 / relativeSD) / std::log(2.0)));
  VELOX_USER_CHECK_GE(
      p,
      4,
      "HLL++ requires at least 4 bits for addressing. Use a lower error, "
      "at most 39%.");
  VELOX_USER_CHECK_LE(
      p,
      16,
      "HLL++ requires at most 16 bits for addressing. Use a higher error.");
  return static_cast<int8_t>(p);
}

template <TypeKind kind>
double toDoubleDispatch(
    const DecodedVector& decoded,
    vector_size_t row,
    const TypePtr& type) {
  if constexpr (kind == TypeKind::TIMESTAMP) {
    return static_cast<double>(decoded.valueAt<Timestamp>(row).toMicros());
  } else if constexpr (
      kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT ||
      kind == TypeKind::INTEGER || kind == TypeKind::BIGINT ||
      kind == TypeKind::REAL || kind == TypeKind::DOUBLE) {
    auto converted = util::Converter<TypeKind::DOUBLE>::tryCast(
        decoded.valueAt<typename TypeTraits<kind>::NativeType>(row));
    VELOX_USER_CHECK(converted.hasValue(), "Failed to convert value to DOUBLE");
    return converted.value();
  } else {
    VELOX_UNSUPPORTED(
        "Unsupported type for approx_count_distinct_for_intervals: {}",
        type->toString());
  }
}

template <TypeKind kind>
uint64_t hashValueDispatch(
    const DecodedVector& decoded,
    vector_size_t row,
    const TypePtr& type) {
  if constexpr (kind == TypeKind::TINYINT) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashInt32(
        static_cast<int32_t>(decoded.valueAt<int8_t>(row)), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::SMALLINT) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashInt32(
        static_cast<int32_t>(decoded.valueAt<int16_t>(row)), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::INTEGER) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashInt32(
        decoded.valueAt<int32_t>(row), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::BIGINT) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashInt64(
        decoded.valueAt<int64_t>(row), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::REAL) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashFloat(
        decoded.valueAt<float>(row), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::DOUBLE) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashDouble(
        decoded.valueAt<double>(row), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::TIMESTAMP) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashTimestamp(
        decoded.valueAt<Timestamp>(row), kXxHash64Seed);
  } else {
    VELOX_UNSUPPORTED(
        "Unsupported type for approx_count_distinct_for_intervals: {}",
        type->toString());
  }
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
    auto updateRow = [&](auto row, char* group) {
      const double inputValue = toDouble(decodedValues_, row, inputType_);
      if (inputValue < endpointsMin_ || inputValue > endpointsMax_) {
        return;
      }

      const auto intervalIndex = std::isnan(inputValue)
          ? (intervalCount_ - 1)
          : findIntervalIndex(inputValue);
      auto tracker = trackRowSize(group);
      auto* accumulator = value<Accumulator>(group);
      accumulator->ensureSize(allocator_, intervalCount_, indexBitLength_);
      const uint64_t hash = hashValue(decodedValues_, row, inputType_);
      accumulator->hlls[intervalIndex].insertHash(hash);
      clearNull(group);
    };

    if (mayHaveNulls) {
      rows.applyToSelected([&](auto row) {
        if (decodedValues_.isNullAt(row)) {
          return;
        }
        updateRow(row, groups[row]);
      });
    } else {
      rows.applyToSelected([&](auto row) { updateRow(row, groups[row]); });
    }
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
    auto updateRow = [&](auto row) {
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
    };

    if (mayHaveNulls) {
      rows.applyToSelected([&](auto row) {
        if (decodedValues_.isNullAt(row)) {
          return;
        }
        updateRow(row);
      });
    } else {
      rows.applyToSelected([&](auto row) { updateRow(row); });
    }
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
    std::vector<HllAccumulator> hlls;

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
      VELOX_USER_CHECK_EQ(hlls.size(), targetSize);
    }
  };

  void ensureRelativeSd(const std::vector<VectorPtr>& args) {
    if (indexBitLength_ >= 0) {
      return;
    }
    VELOX_USER_CHECK_EQ(
        args.size(),
        3,
        "approx_count_distinct_for_intervals requires relativeSD");
    SelectivityVector rows(args[2]->size());
    rows.setAll();
    DecodedVector decoded;
    decoded.decode(*args[2], rows);
    VELOX_USER_CHECK(
        decoded.isConstantMapping(),
        "relativeSD must be constant for approx_count_distinct_for_intervals");
    VELOX_USER_CHECK(
        !decoded.isNullAt(decoded.index(0)),
        "relativeSD must not be null for approx_count_distinct_for_intervals");
    double relativeSD = decoded.valueAt<double>(decoded.index(0));
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
      VELOX_USER_CHECK_EQ(endpoints_.size(), endpoints.size());
      for (size_t i = 0; i < endpoints.size(); ++i) {
        VELOX_USER_CHECK_EQ(endpoints_[i], endpoints[i]);
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
    if (emptyHll_.empty() && indexBitLength_ >= 0) {
      emptyHll_ = common::hll::SparseHlls::serializeEmpty(indexBitLength_);
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
    } else {
      VELOX_USER_FAIL("Unexpected type of HLL");
    }
    ensureEmptyHll();
  }

  static double toDouble(
      const DecodedVector& decoded,
      vector_size_t row,
      const TypePtr& type) {
    if (type->isShortDecimal()) {
      auto value = decoded.valueAt<int64_t>(row);
      auto scale = type->asShortDecimal().scale();
      return DecimalUtil::toDouble(static_cast<int128_t>(value), scale);
    }
    if (type->isLongDecimal()) {
      auto value = decoded.valueAt<int128_t>(row);
      auto scale = type->asLongDecimal().scale();
      return DecimalUtil::toDouble(value, scale);
    }
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        toDoubleDispatch, type->kind(), decoded, row, type);
  }

  static uint64_t hashValue(
      const DecodedVector& decoded,
      vector_size_t row,
      const TypePtr& type) {
    if (type->isShortDecimal()) {
      auto value = decoded.valueAt<int64_t>(row);
      return ::facebook::velox::functions::sparksql::XxHash64::hashLongDecimal(
          static_cast<int128_t>(value), kXxHash64Seed);
    }
    if (type->isLongDecimal()) {
      auto value = decoded.valueAt<int128_t>(row);
      return ::facebook::velox::functions::sparksql::XxHash64::hashLongDecimal(
          value, kXxHash64Seed);
    }
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        hashValueDispatch, type->kind(), decoded, row, type);
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
    builder.constantArgumentType("double");
    signatures.push_back(builder.build());
  };

  for (const auto& valueType : valueTypes) {
    for (const auto& endpointType : endpointTypes) {
      addSignature(
          exec::AggregateFunctionSignatureBuilder()
              .returnType(returnType)
              .intermediateType(intermediateType)
              .argumentType(valueType)
              .constantArgumentType(fmt::format("array({})", endpointType)));
    }
    addSignature(
        exec::AggregateFunctionSignatureBuilder()
            .integerVariable("b_precision")
            .integerVariable("b_scale")
            .returnType(returnType)
            .intermediateType(intermediateType)
            .argumentType(valueType)
            .constantArgumentType("array(DECIMAL(b_precision, b_scale))"));
  }

  for (const auto& endpointType : endpointTypes) {
    addSignature(
        exec::AggregateFunctionSignatureBuilder()
            .integerVariable("a_precision")
            .integerVariable("a_scale")
            .returnType(returnType)
            .intermediateType(intermediateType)
            .argumentType("DECIMAL(a_precision, a_scale)")
            .constantArgumentType(fmt::format("array({})", endpointType)));
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
          .constantArgumentType("array(DECIMAL(b_precision, b_scale))"));

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
            argTypes.size() == 1 || argTypes.size() == 3,
            "{} takes 3 arguments",
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
