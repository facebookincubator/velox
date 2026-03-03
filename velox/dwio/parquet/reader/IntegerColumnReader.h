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

#include "velox/dwio/common/SelectiveIntegerColumnReader.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::parquet {

class IntegerColumnReader : public dwio::common::SelectiveIntegerColumnReader {
 public:
  IntegerColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveIntegerColumnReader(
            requestedType,
            params,
            scanSpec,
            std::move(fileType)) {}

  bool hasBulkPath() const override {
    return !formatData_->as<ParquetData>().isDeltaBinaryPacked() &&
        !this->fileType().type()->isLongDecimal() &&
        ((this->fileType().type()->isShortDecimal())
             ? formatData_->as<ParquetData>().hasDictionary()
             : true);
  }

  void seekToRowGroup(int64_t index) override {
    SelectiveIntegerColumnReader::seekToRowGroup(index);
    scanState().clear();
    readOffset_ = 0;
    formatData_->as<ParquetData>().seekToRowGroup(index);
  }

  uint64_t skip(uint64_t numValues) override {
    formatData_->as<ParquetData>().skip(numValues);
    return numValues;
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    auto& fileType = static_cast<const ParquetTypeWithId&>(*fileType_);
    auto logicalType = fileType.logicalType_;
    if (logicalType.has_value() && logicalType.value().__isset.INTEGER &&
        !logicalType.value().INTEGER.isSigned) {
      getUnsignedIntValues(rows, requestedType_, result);
    } else {
      getIntValues(rows, requestedType_, result);
      // For INT->Decimal widening in Parquet, apply scale adjustment.
      // Integer values stored in Parquet need to be multiplied by 10^scale
      // when read as DecimalType. This is Parquet-specific because ORC decimal
      // data is already properly scaled.
      if (requestedType_->isDecimal() && *result) {
        auto [precision, scale] = getDecimalPrecisionScale(*requestedType_);
        scaleDecimalValues(*result, static_cast<int32_t>(scale));
      }
    }
  }

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    VELOX_WIDTH_DISPATCH(
        parquetSizeOfIntKind(fileType_->type()->kind()),
        prepareRead,
        offset,
        rows,
        nullptr);
    readCommon<IntegerColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

  template <typename ColumnVisitor>
  void readWithVisitor(const RowSet& rows, ColumnVisitor visitor) {
    formatData_->as<ParquetData>().readWithVisitor(visitor);
  }

 private:
  // Multiplies all non-null decimal values by 10^scaleExp. No-op when
  // scaleExp == 0. Dispatches to the appropriate integer width based on
  // whether requestedType_ is short (int64_t) or long (int128_t) decimal.
  void scaleDecimalValues(const VectorPtr& result, int32_t scaleExp) const {
    VELOX_DCHECK_GE(
        scaleExp, 0, "Expected non-negative scale exponent: {}", scaleExp);
    VELOX_DCHECK_LE(
        scaleExp,
        LongDecimalType::kMaxPrecision,
        "Scale exponent exceeds max decimal precision: {}",
        scaleExp);
    if (scaleExp == 0) {
      return;
    }
    if (requestedType_->isShortDecimal()) {
      // Safe to cast: for short decimal, scaleExp <= maxPrecision(18) - 10 = 8,
      // so kPowersOfTen[scaleExp] <= 10^8 which fits in int64_t.
      applyDecimalScaleMultiplier<int64_t>(
          result, static_cast<int64_t>(DecimalUtil::kPowersOfTen[scaleExp]));
    } else {
      applyDecimalScaleMultiplier<int128_t>(
          result, DecimalUtil::kPowersOfTen[scaleExp]);
    }
  }

  // Multiplies all non-null values in result by multiplier.
  template <typename T>
  void applyDecimalScaleMultiplier(const VectorPtr& result, T multiplier)
      const {
    auto* flat = result->asUnchecked<FlatVector<T>>();
    auto* rawValues = flat->mutableRawValues();
    const auto* rawNulls = flat->rawNulls();
    const auto size = flat->size();
    if (!rawNulls) {
      for (vector_size_t i = 0; i < size; ++i) {
        rawValues[i] *= multiplier;
      }
    } else {
      for (vector_size_t i = 0; i < size; ++i) {
        if (bits::isBitSet(rawNulls, i)) {
          rawValues[i] *= multiplier;
        }
      }
    }
  }
};

} // namespace facebook::velox::parquet
