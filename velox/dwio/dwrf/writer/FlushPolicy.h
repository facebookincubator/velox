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

#include <cstdint>
#include "velox/dwio/dwrf/writer/WriterContext.h"

namespace facebook::velox::dwrf {

class DefaultFlushPolicy {
 public:
  explicit DefaultFlushPolicy(
      uint64_t stripeSizeThreshold,
      uint64_t dictionarySizeThreshold);
  // Actually, should probably take the three values respectively for ease of
  // testing.
  bool operator()(bool overMemoryBudget, const WriterContext& context) const;

  bool operator()(uint64_t estimatedStripeSize, uint64_t dictionarySize) const;

 private:
  const uint64_t stripeSizeThreshold_;
  const uint64_t dictionarySizeThreshold_;
};

class RowsPerStripeFlushPolicy {
 public:
  explicit RowsPerStripeFlushPolicy(std::vector<uint64_t> rowsPerStripe);
  bool operator()(bool overMemoryBudget, const WriterContext& context) const;

 private:
  std::vector<uint64_t> rowsPerStripe_;
};

} // namespace facebook::velox::dwrf
