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

#include "velox/dwio/parquet/common/BloomFilter.h"
#include "velox/type/Filter.h"

namespace facebook::velox::parquet {

class ParquetBloomFilter final : public common::AbstractBloomFilter {
 public:
  ParquetBloomFilter(
      std::shared_ptr<facebook::velox::parquet::BloomFilter> bloomFilter)
      : bloomFilter_(bloomFilter) {}

  bool mightContainInt32(int32_t value) const override {
    return bloomFilter_->findHash(bloomFilter_->hashInt32(value));
  }

  bool mightContainInt64(int64_t value) const override {
    return bloomFilter_->findHash(bloomFilter_->hashInt64(value));
  }

  bool mightContainString(const std::string& value) const override {
    ByteArray byteArray{value};
    return bloomFilter_->findHash(bloomFilter_->hashByteArray(&byteArray));
  }

 private:
  std::shared_ptr<facebook::velox::parquet::BloomFilter> bloomFilter_;
};

} // namespace facebook::velox::parquet
