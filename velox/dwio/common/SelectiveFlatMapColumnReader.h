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
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec)
      : SelectiveStructColumnReaderBase(
            requestedType,
            fileType,
            params,
            scanSpec,
            false,
            false) {}

  void getValues(const RowSet& rows, VectorPtr* result) override {
    VELOX_UNSUPPORTED(
        "This overload of getValues is not supported. Please pass in the inMapProvider.");
  }

  void getValues(
      const RowSet& rows,
      VectorPtr* result,
      const std::function<const velox::BufferPtr&(const size_t)>&
          inMapProvider) {
    VELOX_CHECK(!scanSpec_->children().empty());
    VELOX_CHECK_NOT_NULL(
        *result, "SelectiveFlatMapColumnReader expects a non-null result");
    VELOX_CHECK(
        result->get()->type()->isMap(),
        "Struct reader expects a result of type MAP.");

    if (rows.empty()) {
      return;
    }

    auto* resultFlatMap = prepareResult(*result, keysVector_, rows.size());
    setComplexNulls(rows, *result);

    for (const auto& childSpec : scanSpec_->children()) {
      VELOX_TRACE_HISTORY_PUSH("getValues %s", childSpec->fieldName().c_str());
      if (!childSpec->keepValues()) {
        continue;
      }

      VELOX_CHECK(
          childSpec->readFromFile(),
          "Flatmap children must always be read from file.");

      if (childSpec->subscript() == kConstantChildSpecSubscript) {
        continue;
      }

      const auto channel = childSpec->channel();
      const auto index = childSpec->subscript();
      auto& childResult = resultFlatMap->mapValuesAt(channel);

      VELOX_CHECK(
          !childSpec->deltaUpdate(),
          "Delta update not supported in flat map yet");
      VELOX_CHECK(
          !childSpec->isConstant(),
          "Flat map values cannot be constant in scanSpec.");
      VELOX_CHECK_EQ(
          childSpec->columnType(),
          velox::common::ScanSpec::ColumnType::kRegular,
          "Flat map only supports regular column types in scan spec.");

      children_[index]->getValues(rows, &childResult);

      for (size_t i = 0; i < children_.size(); ++i) {
        const auto& inMapBuffer = inMapProvider(i);
        if (inMapBuffer) {
          resultFlatMap->inMapsAt(i, true) = inMapBuffer;
        }
      }
    }
  }

  void seekTo(int64_t offset, bool /*readsNullsOnly*/) override {
    if (offset == readOffset_) {
      return;
    }
    if (readOffset_ < offset) {
      if (numParentNulls_) {
        VELOX_CHECK_LE(
            parentNullsRecordedTo_,
            offset,
            "Must not seek to before parentNullsRecordedTo_");
      }
      auto distance = offset - readOffset_ - numParentNulls_;
      auto numNonNulls = formatData_->skipNulls(distance);
      // We inform children how many nulls there were between original position
      // and destination. The children will seek this many less. The
      // nulls include the nulls found here as well as the enclosing
      // level nulls reported to this by parents.
      for (auto& child : children_) {
        if (child) {
          child->addSkippedParentNulls(
              readOffset_, offset, numParentNulls_ + distance - numNonNulls);
        }
      }
      numParentNulls_ = 0;
      parentNullsRecordedTo_ = 0;
      readOffset_ = offset;
    } else {
      VELOX_FAIL("Seeking backward on a ColumnReader");
    }
  }

  VectorPtr keysVector_;

 private:
  FlatMapVector* prepareResult(
      VectorPtr& result,
      const VectorPtr& distinctKeys,
      vector_size_t size) const;
};

} // namespace facebook::velox::dwio::common
