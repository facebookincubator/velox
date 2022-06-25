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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::dwrf {

class AbstractColumnReader {
 public:
  explicit AbstractColumnReader(
      memory::MemoryPool& memoryPool,
      const std::shared_ptr<const dwio::common::TypeWithId>& type)
      : nodeType_{type}, memoryPool_{memoryPool} {}

  virtual ~AbstractColumnReader() = default;

  /**
   * Skip number of specified rows.
   * @param numValues the number of values to skip
   * @return the number of non-null values skipped
   */
  virtual uint64_t skip(uint64_t numValues) = 0;

  /**
   * Read the next group of values into a RowVector.
   * @param numValues the number of values to read
   * @param vector to read into
   */
  virtual void
  next(uint64_t numValues, VectorPtr& result, const uint64_t* nulls = nullptr) {
    VELOX_UNSUPPORTED("next() is only defined in SelectiveStructColumnReader");
  };

  // Note: for DWRF/ORC this filters the row groups within a stripe.
  // Return list of strides/rowgroups that can be skipped (based on statistics).
  // Stride indices are monotonically increasing.
  virtual std::vector<uint32_t> filterRowGroups(
      uint64_t /*rowGroupSize*/,
      const dwio::common::StatsContext& /* context */) const {
    static const std::vector<uint32_t> kEmpty;
    return kEmpty;
  }

  // Sets the streams of this and child readers to the first row of
  // the row group at 'index'. This advances readers and touches the
  // actual data, unlike setRowGroup().
  virtual void seekToRowGroup(uint32_t /*index*/) {
    VELOX_NYI();
  }

 protected:
  const std::shared_ptr<const dwio::common::TypeWithId> nodeType_;
  memory::MemoryPool& memoryPool_;
};

} // namespace facebook::velox::dwrf
