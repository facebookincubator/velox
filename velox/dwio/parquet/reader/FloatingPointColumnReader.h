/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/common/SelectiveFloatingPointColumnReader.h"

namespace facebook::velox::parquet {

template <typename TData, typename TRequested>
class FloatingPointColumnReader
    : public dwio::common::
          SelectiveFloatingPointColumnReader<TData, TRequested> {
 public:
  using ValueType = TRequested;

  using base =
      dwio::common::SelectiveFloatingPointColumnReader<TData, TRequested>;

  FloatingPointColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec);

  // Parquet floating point reader always supports a bulk path
  static constexpr bool kHasBulkPath = true;

  bool hasBulkPath() const override {
    return kHasBulkPath;
  }

  void seekToRowGroup(int64_t index) override {
    base::seekToRowGroup(index);
    this->scanState().clear();
    this->readOffset_ = 0;
    this->formatData_->template as<ParquetData>().seekToRowGroup(index);
  }

  uint64_t skip(uint64_t numValues) override;

  void read(int64_t offset, const RowSet& rows, const uint64_t* incomingNulls)
      override {
    using T = FloatingPointColumnReader<TData, TRequested>;
    this->template readCommon<T, true>(offset, rows, incomingNulls);
    this->readOffset_ += rows.back() + 1;
  }

  template <typename TVisitor>
  void readWithVisitor(const RowSet& rows, TVisitor visitor);
};

template <typename TData, typename TRequested>
FloatingPointColumnReader<TData, TRequested>::FloatingPointColumnReader(
    const TypePtr& requestedType,
    std::shared_ptr<const dwio::common::TypeWithId> fileType,
    ParquetParams& params,
    common::ScanSpec& scanSpec)
    : dwio::common::SelectiveFloatingPointColumnReader<TData, TRequested>(
          requestedType,
          std::move(fileType),
          params,
          scanSpec) {
  VELOX_DCHECK(
      (this->requestedType_->kind() == TypeKind::REAL &&
       std::is_same_v<TRequested, float>) ||
          (this->requestedType_->kind() == TypeKind::DOUBLE &&
           std::is_same_v<TRequested, double>),
      "TRequested type mismatch: template parameter is {}, but requestedType is {}",
      folly::demangle(typeid(TRequested)),
      this->requestedType_->toString());
}

template <typename TData, typename TRequested>
uint64_t FloatingPointColumnReader<TData, TRequested>::skip(
    uint64_t numValues) {
  return this->formatData_->skip(numValues);
}

template <typename TData, typename TRequested>
template <typename TVisitor>
void FloatingPointColumnReader<TData, TRequested>::readWithVisitor(
    const RowSet& rows,
    TVisitor visitor) {
  this->formatData_->template as<ParquetData>().readWithVisitor(visitor);
}

} // namespace facebook::velox::parquet
