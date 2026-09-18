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

#include "velox/dwio/common/ColumnLoader.h"
#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/dwio/nimble/velox/selective/ReaderBase.h"
#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"

namespace facebook::nimble {

class TrackedColumnLoader : public velox::dwio::common::ColumnLoader {
 public:
  TrackedColumnLoader(
      velox::dwio::common::SelectiveStructColumnReaderBase* structReader,
      velox::dwio::common::SelectiveColumnReader* fieldReader,
      uint64_t version,
      RowSizeTracker* rowSizeTracker = nullptr)
      : velox::dwio::common::ColumnLoader{structReader, fieldReader, version},
        typeWithId_{fieldReader->fileType()},
        rowSizeTracker_{rowSizeTracker} {}

 private:
  void loadInternal(
      velox::RowSet rows,
      velox::ValueHook* hook,
      velox::vector_size_t resultSize,
      velox::VectorPtr* result) override {
    velox::dwio::common::ColumnLoader::loadInternal(
        rows, hook, resultSize, result);
    if (result && rowSizeTracker_) {
      updateRowSize(typeWithId_, resultSize, *result);
    }
  }

  bool isFullyLoaded(const velox::VectorPtr& vector) const {
    if (!vector || isLazyNotLoaded(*vector)) {
      return false;
    }

    const auto& vectorType = vector->type();
    switch (vectorType->kind()) {
      case velox::TypeKind::BOOLEAN:
      case velox::TypeKind::TINYINT:
      case velox::TypeKind::SMALLINT:
      case velox::TypeKind::INTEGER:
      case velox::TypeKind::BIGINT:
      case velox::TypeKind::HUGEINT:
      case velox::TypeKind::REAL:
      case velox::TypeKind::DOUBLE:
      case velox::TypeKind::VARCHAR:
      case velox::TypeKind::VARBINARY:
      case velox::TypeKind::TIMESTAMP: {
        return true;
      }
      case velox::TypeKind::ARRAY: {
        if (vector->encoding() != velox::VectorEncoding::Simple::ARRAY) {
          return true;
        }
        velox::DecodedVector decodedVector(*vector);
        auto arrayVector = decodedVector.base()->as<velox::ArrayVector>();
        return isFullyLoaded(arrayVector->elements());
      }
      case velox::TypeKind::MAP: {
        if (vector->encoding() != velox::VectorEncoding::Simple::MAP) {
          return true;
        }
        velox::DecodedVector decodedVector(*vector);
        auto mapVector = decodedVector.base()->as<velox::MapVector>();
        return isFullyLoaded(mapVector->mapKeys()) &&
            isFullyLoaded(mapVector->mapValues());
      }
      case velox::TypeKind::ROW: {
        if (vector->encoding() != velox::VectorEncoding::Simple::ROW) {
          return true;
        }
        velox::DecodedVector decodedVector(*vector);
        auto rowVector = decodedVector.base()->as<velox::RowVector>();
        for (const auto& child : rowVector->children()) {
          if (!isFullyLoaded(child)) {
            return false;
          }
        }
        return true;
      }
      default: {
        VELOX_FAIL("Unsupported lazy field type " + vectorType->toString());
      }
    }
  }

  void updateRowSize(
      const velox::dwio::common::TypeWithId& localTypeWithId,
      velox::vector_size_t resultSize,
      const velox::VectorPtr& vector) {
    if (!vector || isLazyNotLoaded(*vector)) {
      return;
    }

    const auto& vectorType = vector->type();

    switch (vectorType->kind()) {
      case velox::TypeKind::BOOLEAN:
      case velox::TypeKind::TINYINT:
      case velox::TypeKind::SMALLINT:
      case velox::TypeKind::INTEGER:
      case velox::TypeKind::BIGINT:
      case velox::TypeKind::HUGEINT:
      case velox::TypeKind::REAL:
      case velox::TypeKind::DOUBLE:
      case velox::TypeKind::VARCHAR:
      case velox::TypeKind::VARBINARY:
      case velox::TypeKind::TIMESTAMP: {
        VLOG(1) << fmt::format(
            "updating primitive type, node id {}, vector size {}, row count {}",
            localTypeWithId.id(),
            vector->retainedSize(),
            resultSize);
        rowSizeTracker_->update(
            localTypeWithId.id(), vector->retainedSize(), resultSize);
        break;
      }
      case velox::TypeKind::ARRAY: {
        // Or we can subtract the retained sizes from the children retained
        // sizes.
        rowSizeTracker_->update(localTypeWithId.id(), 0, resultSize);
        if (vector->encoding() != velox::VectorEncoding::Simple::ARRAY) {
          if (isFullyLoaded(vector)) {
            rowSizeTracker_->update(
                localTypeWithId.id(), vector->retainedSize(), resultSize);
          }
          break;
        }

        auto arrayVector = vector->as<velox::ArrayVector>();
        updateRowSize(
            *localTypeWithId.childAt(0), resultSize, arrayVector->elements());
        break;
      }
      case velox::TypeKind::MAP: {
        rowSizeTracker_->update(localTypeWithId.id(), 0, resultSize);
        if (vector->encoding() != velox::VectorEncoding::Simple::MAP) {
          if (isFullyLoaded(vector)) {
            rowSizeTracker_->update(
                localTypeWithId.id(), vector->retainedSize(), resultSize);
          }
          break;
        }

        auto mapVector = vector->as<velox::MapVector>();
        updateRowSize(
            *localTypeWithId.childAt(0), resultSize, mapVector->mapKeys());
        updateRowSize(
            *localTypeWithId.childAt(1), resultSize, mapVector->mapValues());
        break;
      }
      case velox::TypeKind::ROW: {
        if (vector->encoding() != velox::VectorEncoding::Simple::ROW ||
            localTypeWithId.type()->kind() == velox::TypeKind::MAP) {
          if (isFullyLoaded(vector)) {
            rowSizeTracker_->update(
                localTypeWithId.id(), vector->retainedSize(), resultSize);
          }
          break;
        }

        rowSizeTracker_->update(localTypeWithId.id(), 0, resultSize);
        auto rowVector = vector->as<velox::RowVector>();

        // Deal with schema evolution..
        const auto childrenCount = std::min(
            rowVector->childrenSize(),
            static_cast<uint64_t>(localTypeWithId.size()));
        for (auto i = 0; i < childrenCount; i++) {
          updateRowSize(
              *localTypeWithId.childAt(i), resultSize, rowVector->childAt(i));
        }
        break;
      }
      default: {
        VELOX_FAIL("Unsupported lazy field type " + vectorType->toString());
      }
    }
  }

  const velox::dwio::common::TypeWithId& typeWithId_;
  RowSizeTracker* const rowSizeTracker_;
};

} // namespace facebook::nimble
