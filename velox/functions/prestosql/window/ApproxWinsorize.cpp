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
#include "velox/exec/WindowFunction.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/TDigest.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::window::prestosql {

namespace {

constexpr double kMinCompression = 10.0;
constexpr double kMaxCompression = 1'000.0;
constexpr double kDefaultCompression = 100.0;

// Returns per-row winsorized values using a TDigest
// sketch built over the entire partition. Values below the lower quantile
// boundary are replaced with that boundary value; values above the upper
// quantile boundary are replaced with that boundary value.
//
// This eliminates the common two-pass pattern:
//   Pass 1: APPROX_PERCENTILE(col, p) -> threshold
//   Pass 2: LEAST(col, threshold) per row
// by building the sketch once in resetPartition() and clamping per row
// in apply().
//
// Signatures:
//   approx_winsorize(value DOUBLE, lower DOUBLE, upper DOUBLE) OVER (...)
//   approx_winsorize(value DOUBLE, lower DOUBLE, upper DOUBLE,
//                    compression DOUBLE) OVER (...)
class ApproxWinsorizeFunction : public exec::WindowFunction {
 public:
  ApproxWinsorizeFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      memory::MemoryPool* pool,
      HashStringAllocator* stringAllocator)
      : WindowFunction(resultType, pool, stringAllocator) {
    // Argument 0: value column (not constant).
    VELOX_USER_CHECK(
        args[0].index.has_value(),
        "First argument (value) must be a column, not a constant");
    valueIndex_ = args[0].index.value();

    // Arguments 1-2: lower and upper quantile bounds (must be constants).
    VELOX_USER_CHECK(
        args[1].constantValue != nullptr,
        "Lower quantile bound must be a constant");
    VELOX_USER_CHECK(
        args[2].constantValue != nullptr,
        "Upper quantile bound must be a constant");

    lowerQuantile_ =
        args[1].constantValue->as<ConstantVector<double>>()->valueAt(0);
    upperQuantile_ =
        args[2].constantValue->as<ConstantVector<double>>()->valueAt(0);

    VELOX_USER_CHECK(
        !std::isnan(lowerQuantile_) && !std::isnan(upperQuantile_),
        "Quantile bounds must not be NaN");
    VELOX_USER_CHECK_GE(
        lowerQuantile_, 0.0, "Lower quantile bound must be >= 0");
    VELOX_USER_CHECK_LE(
        lowerQuantile_, 1.0, "Lower quantile bound must be <= 1");
    VELOX_USER_CHECK_GE(
        upperQuantile_, 0.0, "Upper quantile bound must be >= 0");
    VELOX_USER_CHECK_LE(
        upperQuantile_, 1.0, "Upper quantile bound must be <= 1");
    VELOX_USER_CHECK_LE(
        lowerQuantile_,
        upperQuantile_,
        "Lower quantile bound must be <= upper quantile bound");

    // Argument 3 (optional): compression factor.
    if (args.size() > 3) {
      VELOX_USER_CHECK(
          args[3].constantValue != nullptr,
          "Compression factor must be a constant");
      compression_ =
          args[3].constantValue->as<ConstantVector<double>>()->valueAt(0);
      VELOX_USER_CHECK(
          !std::isnan(compression_), "Compression factor must not be NaN");
      VELOX_USER_CHECK_GT(
          compression_, 0.0, "Compression factor must be positive");
      VELOX_USER_CHECK_LE(
          compression_,
          kMaxCompression,
          "Compression must be at most {}",
          kMaxCompression);
      compression_ = std::max(compression_, kMinCompression);
    }
  }

  void resetPartition(const exec::WindowPartition* partition) override {
    partition_ = partition;
    numPartitionRows_ = partition->numRows();
    rowOffset_ = 0;

    if (numPartitionRows_ == 0) {
      lowerBound_ = 0.0;
      upperBound_ = 0.0;
      return;
    }

    // Extract all values from the partition.
    auto values = BaseVector::create(DOUBLE(), numPartitionRows_, pool_);
    partition_->extractColumn(
        static_cast<int32_t>(valueIndex_), 0, numPartitionRows_, 0, values);
    auto* flatValues = values->asFlatVector<double>();

    // Build a TDigest from all non-null values.
    functions::TDigest<> digest;
    digest.setCompression(compression_);
    std::vector<int16_t> positions;
    for (vector_size_t i = 0; i < numPartitionRows_; ++i) {
      if (!flatValues->isNullAt(i)) {
        double inputValue{flatValues->valueAt(i)};
        VELOX_USER_CHECK(
            !std::isnan(inputValue), "Cannot add NaN to approx_winsorize");
        digest.add(positions, inputValue);
      }
    }

    if (digest.size() == 0) {
      // All values are null; bounds don't matter.
      lowerBound_ = 0.0;
      upperBound_ = 0.0;
      return;
    }

    digest.compress(positions);

    // Compute quantile boundaries once for the entire partition.
    lowerBound_ = digest.estimateQuantile(lowerQuantile_);
    upperBound_ = digest.estimateQuantile(upperQuantile_);
  }

  void apply(
      const BufferPtr& peerGroupStarts,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& /*frameStarts*/,
      const BufferPtr& /*frameEnds*/,
      const SelectivityVector& validRows,
      vector_size_t resultOffset,
      const VectorPtr& result) override {
    // Derive batch size from the peerGroupStarts buffer, matching the pattern
    // used by CumeDist and other window functions.
    auto numRows = static_cast<vector_size_t>(
        peerGroupStarts->size() / sizeof(vector_size_t));
    auto* flatResult = result->asFlatVector<double>();

    // Extract the value column for this batch.
    auto values = BaseVector::create(DOUBLE(), numRows, pool_);
    partition_->extractColumn(
        static_cast<int32_t>(valueIndex_), rowOffset_, numRows, 0, values);
    auto* flatValues = values->asFlatVector<double>();

    for (vector_size_t i = 0; i < numRows; ++i) {
      if (!validRows.isValid(i) || flatValues->isNullAt(i)) {
        flatResult->setNull(resultOffset + i, true);
      } else {
        double inputValue{flatValues->valueAt(i)};
        flatResult->set(
            resultOffset + i,
            std::max(lowerBound_, std::min(upperBound_, inputValue)));
      }
    }

    rowOffset_ += numRows;
  }

 private:
  column_index_t valueIndex_;
  double lowerQuantile_;
  double upperQuantile_;
  double compression_{kDefaultCompression};

  const exec::WindowPartition* partition_{nullptr};
  vector_size_t numPartitionRows_{0};
  vector_size_t rowOffset_{0};

  // Quantile boundary values computed once per partition.
  double lowerBound_{0.0};
  double upperBound_{0.0};
};

} // namespace

void registerApproxWinsorize(const std::string& name) {
  std::vector<exec::FunctionSignaturePtr> signatures;
  // (value DOUBLE, lower DOUBLE, upper DOUBLE) -> DOUBLE
  signatures.push_back(
      exec::FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .argumentType("double")
          .argumentType("double")
          .build());
  // (value DOUBLE, lower DOUBLE, upper DOUBLE, compression DOUBLE) -> DOUBLE
  signatures.push_back(
      exec::FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .argumentType("double")
          .argumentType("double")
          .argumentType("double")
          .build());

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      exec::WindowFunction::Metadata::defaultMetadata(),
      [name](
          const std::vector<exec::WindowFunctionArg>& args,
          const TypePtr& resultType,
          bool /*ignoreNulls*/,
          memory::MemoryPool* pool,
          HashStringAllocator* stringAllocator,
          const core::QueryConfig& /*queryConfig*/)
          -> std::unique_ptr<exec::WindowFunction> {
        return std::make_unique<ApproxWinsorizeFunction>(
            args, resultType, pool, stringAllocator);
      });
}

} // namespace facebook::velox::window::prestosql
