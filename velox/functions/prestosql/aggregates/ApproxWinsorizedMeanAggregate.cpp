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
#include <cstring>

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/functions/lib/TDigest.h"
#include "velox/vector/FlatVector.h"

using namespace facebook::velox::exec;

namespace facebook::velox::aggregate::prestosql {

namespace {

constexpr double kMinCompression{10.0};
constexpr double kMaxCompression{1'000.0};

// The intermediate type is varbinary containing:
//   [8 bytes: lowerQuantile double][8 bytes: upperQuantile double][TDigest
//   bytes]
// This ensures quantile bounds survive partial-to-final distributed
// aggregation.
constexpr size_t kQuantileBoundsHeaderSize{2 * sizeof(double)};

// Accumulates a TDigest sketch for computing the winsorized mean of a set of
// values. Stores the TDigest and the optional compression parameter.
struct WinsorizedMeanAccumulator {
  explicit WinsorizedMeanAccumulator(HashStringAllocator* allocator)
      : digest(StlAllocator<double>(allocator)) {}

  double compression{0.0};
  facebook::velox::functions::TDigest<StlAllocator<double>> digest;
};

// Velox aggregate function implementing approx_winsorized_mean using
// TDigest-based quantile estimation. Builds a TDigest in a single pass,
// then computes the Winsorized mean at extractValues time.
//
// The intermediate type is varbinary with a 16-byte header containing
// the lower and upper quantile bounds, followed by the serialized TDigest.
// This ensures quantile bounds survive partial-to-final distributed
// aggregation.
class ApproxWinsorizedMeanAggregate : public exec::Aggregate {
 private:
  // Whether the optional compression parameter was provided.
  bool hasCompression_{false};
  // Validated compression factor, shared across all rows. Zero means unset.
  double compression_{0.0};
  // Lower quantile bound for winsorization [0.0, 1.0].
  double lowerQuantile_{0.0};
  // Upper quantile bound for winsorization [0.0, 1.0].
  double upperQuantile_{1.0};
  // True once quantile bounds have been set from the first non-null row.
  bool quantileBoundsSet_{false};
  // Decoded input columns for raw input processing.
  DecodedVector decodedValue_;
  DecodedVector decodedLower_;
  DecodedVector decodedUpper_;
  DecodedVector decodedCompression_;
  // Decoded intermediate input for partial-to-final aggregation.
  DecodedVector decodedIntermediate_;

 public:
  ApproxWinsorizedMeanAggregate(const TypePtr& resultType, bool hasCompression)
      : exec::Aggregate(resultType), hasCompression_{hasCompression} {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(WinsorizedMeanAccumulator);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(WinsorizedMeanAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  WinsorizedMeanAccumulator* getAccumulator(char* group) {
    return value<WinsorizedMeanAccumulator>(group);
  }

  // Decodes raw input columns: args[0]=value, args[1]=lower, args[2]=upper,
  // and optionally args[3]=compression.
  void decodeArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    decodedValue_.decode(*args[0], rows, true);
    decodedLower_.decode(*args[1], rows, true);
    decodedUpper_.decode(*args[2], rows, true);
    if (hasCompression_) {
      decodedCompression_.decode(*args[3], rows, true);
    }
  }

  // Validates that quantile bounds are in [0,1] with lower <= upper, and
  // ensures all rows use the same bounds. Throws on inconsistency.
  void checkAndSetQuantileBounds(vector_size_t row) {
    double lower = decodedLower_.valueAt<double>(row);
    double upper = decodedUpper_.valueAt<double>(row);
    VELOX_USER_CHECK(
        !std::isnan(lower) && !std::isnan(upper),
        "Quantile bounds must not be NaN.");
    VELOX_USER_CHECK_GE(lower, 0.0, "Lower quantile bound must be >= 0");
    VELOX_USER_CHECK_LE(lower, 1.0, "Lower quantile bound must be <= 1");
    VELOX_USER_CHECK_GE(upper, 0.0, "Upper quantile bound must be >= 0");
    VELOX_USER_CHECK_LE(upper, 1.0, "Upper quantile bound must be <= 1");
    VELOX_USER_CHECK_LE(
        lower,
        upper,
        "Lower quantile bound must be less than or equal to upper quantile bound");
    if (!quantileBoundsSet_) {
      lowerQuantile_ = lower;
      upperQuantile_ = upper;
      quantileBoundsSet_ = true;
    } else if (lowerQuantile_ != lower || upperQuantile_ != upper) {
      VELOX_USER_FAIL("Quantile bounds must be the same for all rows");
    }
  }

  // Validates compression is positive, at most kMaxCompression, and consistent
  // across rows. Clamps to kMinCompression and applies to the accumulator.
  void checkAndSetCompression(
      WinsorizedMeanAccumulator* accumulator,
      vector_size_t row) {
    double compressionValue = decodedCompression_.valueAt<double>(row);
    VELOX_USER_CHECK(
        !std::isnan(compressionValue), "Compression factor must not be NaN.");
    VELOX_USER_CHECK_GT(
        compressionValue, 0, "Compression factor must be positive.");
    VELOX_USER_CHECK_LE(
        compressionValue,
        kMaxCompression,
        "Compression must be at most {}",
        kMaxCompression);
    compressionValue = std::max(compressionValue, kMinCompression);
    if (compression_ == 0.0) {
      compression_ = compressionValue;
    } else if (compression_ != compressionValue) {
      VELOX_USER_FAIL("Compression factor must be the same for all rows");
    }
    if (accumulator->compression == 0.0) {
      accumulator->compression = compression_;
    }
    if (accumulator->digest.compression() != compression_) {
      accumulator->digest.setCompression(compression_);
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    std::vector<int16_t> positions;
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }
      double inputValue = decodedValue_.valueAt<double>(row);
      if (std::isnan(inputValue)) {
        VELOX_USER_FAIL("Cannot add NaN to approx_winsorized_mean");
      }
      checkAndSetQuantileBounds(row);
      auto* accumulator = getAccumulator(groups[row]);
      if (hasCompression_) {
        checkAndSetCompression(accumulator, row);
      }
      accumulator->digest.add(positions, inputValue);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    auto* accumulator = getAccumulator(group);
    std::vector<int16_t> positions;
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }
      double inputValue = decodedValue_.valueAt<double>(row);
      if (std::isnan(inputValue)) {
        VELOX_USER_FAIL("Cannot add NaN to approx_winsorized_mean");
      }
      checkAndSetQuantileBounds(row);
      if (hasCompression_) {
        checkAndSetCompression(accumulator, row);
      }
      accumulator->digest.add(positions, inputValue);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    addIntermediate<false>(groups, rows, args);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    addIntermediate<true>(group, rows, args);
  }

  template <bool kSingleGroup>
  void addIntermediate(
      std::conditional_t<kSingleGroup, char*, char**> group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    decodedIntermediate_.decode(*args[0], rows);
    std::vector<int16_t> positions;
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      WinsorizedMeanAccumulator* accumulator;
      if constexpr (kSingleGroup) {
        accumulator = getAccumulator(group);
      } else {
        accumulator = getAccumulator(group[row]);
      }

      auto serialized = decodedIntermediate_.valueAt<StringView>(row);
      VELOX_CHECK_GE(
          serialized.size(),
          kQuantileBoundsHeaderSize,
          "Intermediate data too small to contain quantile bounds header");
      auto* data = serialized.data();

      // Restore quantile bounds from the varbinary header.
      if (!quantileBoundsSet_) {
        memcpy(&lowerQuantile_, data, sizeof(double));
        memcpy(&upperQuantile_, data + sizeof(double), sizeof(double));
        quantileBoundsSet_ = true;
      }

      // Merge the TDigest bytes after the header.
      accumulator->digest.mergeDeserialized(
          positions, data + kQuantileBoundsHeaderSize);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    if (numGroups == 0) {
      (*result)->resize(0);
      return;
    }
    auto flatResult = (*result)->asFlatVector<double>();
    flatResult->resize(numGroups);
    std::vector<int16_t> positions;
    for (int32_t i = 0; i < numGroups; ++i) {
      auto group = groups[i];
      if (!group) {
        flatResult->setNull(i, true);
        continue;
      }
      auto* accumulator = getAccumulator(group);
      if (!isInitialized(group) || accumulator->digest.size() == 0) {
        flatResult->setNull(i, true);
        continue;
      }
      accumulator->digest.compress(positions);
      double winsorizedMeanValue =
          accumulator->digest.winsorizedMean(lowerQuantile_, upperQuantile_);
      if (std::isnan(winsorizedMeanValue)) {
        flatResult->setNull(i, true);
      } else {
        flatResult->set(i, winsorizedMeanValue);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    // Serialize as varbinary: [lowerQuantile][upperQuantile][TDigest bytes]
    if (numGroups == 0) {
      (*result)->resize(0);
      return;
    }
    auto flatResult = (*result)->asFlatVector<StringView>();
    flatResult->resize(numGroups);
    std::vector<int16_t> positions;
    for (int32_t i = 0; i < numGroups; ++i) {
      auto group = groups[i];
      if (!group) {
        flatResult->setNull(i, true);
        continue;
      }
      auto* accumulator = getAccumulator(group);
      if (!isInitialized(group) || accumulator->digest.size() == 0) {
        flatResult->setNull(i, true);
        continue;
      }

      accumulator->digest.compress(positions);
      auto digestSize = accumulator->digest.serializedByteSize();
      int32_t totalSize =
          static_cast<int32_t>(kQuantileBoundsHeaderSize + digestSize);

      char* rawBuffer = flatResult->getRawStringBufferWithSpace(totalSize);
      // Write quantile bounds header.
      memcpy(rawBuffer, &lowerQuantile_, sizeof(double));
      memcpy(rawBuffer + sizeof(double), &upperQuantile_, sizeof(double));
      // Write TDigest after header.
      accumulator->digest.serialize(rawBuffer + kQuantileBoundsHeaderSize);

      flatResult->setNoCopy(i, StringView(rawBuffer, totalSize));
    }
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (group + offset_) WinsorizedMeanAccumulator(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        value<WinsorizedMeanAccumulator>(group)->~WinsorizedMeanAccumulator();
      }
    }
  }
};
} // namespace

void registerApproxWinsorizedMeanAggregate(
    const std::vector<std::string>& names,
    bool overwrite) {
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures;
  // (value DOUBLE, lower DOUBLE, upper DOUBLE) -> DOUBLE
  signatures.push_back(
      AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("varbinary")
          .argumentType("double")
          .argumentType("double")
          .argumentType("double")
          .build());
  // (value DOUBLE, lower DOUBLE, upper DOUBLE, compression DOUBLE) -> DOUBLE
  signatures.push_back(
      AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("varbinary")
          .argumentType("double")
          .argumentType("double")
          .argumentType("double")
          .argumentType("double")
          .build());
  exec::registerAggregateFunction(
      names,
      signatures,
      [](core::AggregationNode::Step /*step*/,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        bool hasCompression = argTypes.size() == 4;
        return std::make_unique<ApproxWinsorizedMeanAggregate>(
            resultType, hasCompression);
      },
      {},
      false /*registerCompanionFunctions*/,
      overwrite);
}
} // namespace facebook::velox::aggregate::prestosql
