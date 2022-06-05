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

#include "velox/dwio/common/reader/ColumnReader.h"

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/dwrf/common/ByteRLE.h"
#include "velox/dwio/dwrf/common/Compression.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/reader/EncodingContext.h"
#include "velox/dwio/dwrf/reader/StripeStream.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::dwrf {

/**
 * The interface for reading ORC data types.
 */
class DwrfColumnReader : public dwio::common::reader::ColumnReader {
 protected:
  explicit DwrfColumnReader(
      memory::MemoryPool& memoryPool,
      const std::shared_ptr<const dwio::common::TypeWithId>& type)
      : dwio::common::reader::ColumnReader(type, memoryPool),
        flatMapContext_{FlatMapContext::nonFlatMapContext()} {}

  FlatMapContext flatMapContext_;

 public:
  DwrfColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> nodeId,
      StripeStreams& stripe,
      FlatMapContext flatMapContext = FlatMapContext::nonFlatMapContext());

  virtual ~DwrfColumnReader() = default;

  /**
   * Read the next group of values into a RowVector.
   * @param numValues the number of values to read
   * @param vector to read into
   */
  virtual void next(
      uint64_t numValues,
      VectorPtr& result,
      const uint64_t* nulls = nullptr) = 0;

  // Return list of strides/rowgroups that can be skipped (based on statistics).
  // Stride indices are monotonically increasing.
  virtual std::vector<uint32_t> filterRowGroups(
      uint64_t /*rowGroupSize*/,
      const StatsContext& /* context */) const {
    static const std::vector<uint32_t> kEmpty;
    return kEmpty;
  }

  // Sets the streams of this and child readers to the first row of
  // the row group at 'index'. This advances readers and touches the
  // actual data, unlike setRowGroup().
  virtual void seekToRowGroup(uint32_t /*index*/) {
    VELOX_NYI();
  }

  /**
   * Create a reader for the given stripe.
   */
  static std::unique_ptr<DwrfColumnReader> build(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      StripeStreams& stripe,
      FlatMapContext flatMapContext = FlatMapContext::nonFlatMapContext());
};

class DwrfColumnReaderFactory {
 public:
  virtual ~DwrfColumnReaderFactory() = default;
  virtual std::unique_ptr<DwrfColumnReader> build(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      StripeStreams& stripe,
      FlatMapContext flatMapContext = FlatMapContext::nonFlatMapContext()) {
    return DwrfColumnReader::build(
        requestedType, dataType, stripe, std::move(flatMapContext));
  }

  static DwrfColumnReaderFactory* baseFactory();
};

} // namespace facebook::velox::dwrf
