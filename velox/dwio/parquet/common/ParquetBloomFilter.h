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
  ParquetBloomFilter(std::shared_ptr<BloomFilter> bloomFilter)
      : bloomFilter_(bloomFilter) {}

  bool mightContainInt32(int32_t value) const override;
  bool mightContainInt64(int64_t value) const override;
  bool mightContainString(const std::string& value) const override;
  bool mightContainInt32Range(int32_t low, int32_t high) const override;
  bool mightContainInt64Range(int64_t low, int64_t high) const override;
  bool mightContainInt32Values(
      const std::vector<int64_t>& values) const override;
  bool mightContainInt64Values(
      const std::vector<int64_t>& values) const override;
  bool mightContainStringValues(
      const folly::F14FastSet<std::string>& values) const override;

 private:
  std::shared_ptr<BloomFilter> bloomFilter_;
};

} // namespace facebook::velox::parquet
