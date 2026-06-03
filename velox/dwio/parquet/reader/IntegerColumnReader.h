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
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetData.h"
#include "velox/dwio/parquet/reader/ParquetTypeWithId.h"
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
      if (requestedType_->isDecimal() && !allNull_) {
        rescaleDecimalValues(fileType, *result);
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
  // Rescales integer or decimal values to match the requested decimal type.
  // For INT->Decimal, fileScale is 0; for Decimal->Decimal, fileScale comes
  // from the file's decimal type.
  void rescaleDecimalValues(
      const ParquetTypeWithId& fileType,
      VectorPtr& result) {
    int32_t requestedScale = getDecimalPrecisionScale(*requestedType_).second;
    int32_t fileScale = fileType.type()->isDecimal()
        ? getDecimalPrecisionScale(*fileType.type()).second
        : 0;
    int32_t scaleAdjust = requestedScale - fileScale;
    VELOX_USER_CHECK_GE(
        scaleAdjust,
        0,
        "Parquet does not support scale narrowing: {}",
        scaleAdjust);
    VELOX_USER_CHECK_LE(
        scaleAdjust,
        LongDecimalType::kMaxPrecision,
        "Scale adjustment exceeds max decimal precision: {}",
        scaleAdjust);

    if (scaleAdjust > 0) {
      if (requestedType_->isShortDecimal()) {
        // Safe to cast: kPowersOfTen[scaleAdjust] fits in int64_t because
        // scaleAdjust <= maxPrecision(18) and 10^18 < 2^63.
        applyDecimalScaleMultiplier<int64_t>(
            result,
            static_cast<int64_t>(DecimalUtil::kPowersOfTen[scaleAdjust]));
      } else {
        applyDecimalScaleMultiplier<int128_t>(
            result, DecimalUtil::kPowersOfTen[scaleAdjust]);
      }
    }
  }

  /// Multiplies all non-null values in result by multiplier.
  /// Overflow is impossible because convertType validates precInc >= scaleInc,
  /// guaranteeing that originalValue * 10^scaleAdjust fits within the target
  /// precision.
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
