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

#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/vector/FlatMapVector.h"

namespace facebook::velox::dwio::common {

class SelectiveFlatMapColumnReader : public SelectiveStructColumnReaderBase {
 protected:
  SelectiveFlatMapColumnReader(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec)
      : SelectiveStructColumnReaderBase(
            columnReaderOptions,
            requestedType,
            fileType,
            params,
            scanSpec,
            false,
            false) {}

  void getValues(const RowSet& rows, VectorPtr* result) override;

  void seekTo(int64_t offset, bool /*readsNullsOnly*/) override {
    seekToPropagateNullsToChildren(offset);
  }

  virtual const BufferPtr& inMapBuffer(column_index_t childIndex) const = 0;

  /// Sets 'selected' to the in-map bits of 'rows', copied from 'inMap', which
  /// has a bit for every row read rather than every row selected. Sets it to
  /// nullptr if 'inMap' is null. The copy also keeps the result from sharing a
  /// buffer the reader reuses for the next batch. Writes into 'selected' or
  /// 'spare' when nothing else holds them rather than allocating.
  static void selectInMap(
      const BufferPtr& inMap,
      const RowSet& rows,
      memory::MemoryPool* pool,
      BufferPtr& selected,
      BufferPtr& spare);

  VectorPtr keysVector_;

 private:
  FlatMapVector* prepareResult(
      VectorPtr& result,
      const VectorPtr& distinctKeys,
      vector_size_t size);

  // Map values of the previous result that nothing else holds, by key, for
  // the child readers to reuse.
  std::vector<VectorPtr> reusableValues_;

  // Spare in-map buffers, by key, for selectInMap().
  std::vector<BufferPtr> spareInMaps_;
};

} // namespace facebook::velox::dwio::common
