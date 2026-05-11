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

#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/vector/LazyVector.h"

namespace facebook::velox::dwio::common {

class ColumnLoader : public VectorLoader {
 public:
  ColumnLoader(
      SelectiveStructColumnReaderBase* structReader,
      SelectiveColumnReader* fieldReader,
      uint64_t version)
      : structReader_(structReader),
        fieldReader_(fieldReader),
        version_(version) {}

  virtual ~ColumnLoader() = default;

  bool supportsHook() const override {
    return true;
  }

 protected:
  void loadInternal(
      RowSet rows,
      ValueHook* hook,
      vector_size_t resultSize,
      VectorPtr* result) override;

  SelectiveStructColumnReaderBase* const structReader_;
  SelectiveColumnReader* const fieldReader_;
  // This is checked against the version of 'structReader' on load. If
  // these differ, 'structReader' has been advanced since the creation
  // of 'this' and 'this' is no longer loadable.
  const uint64_t version_;
};

/// Wraps a ColumnLoader and applies a post-read transform when the lazy
/// vector is loaded.  Used for mixed extraction transforms where the reader
/// produces the file type and the transform converts to the extraction
/// output type (e.g., MAP → ROW<keys, sz> for MapKeys + Size extractions).
class TransformColumnLoader : public ColumnLoader {
 public:
  TransformColumnLoader(
      SelectiveStructColumnReaderBase* structReader,
      SelectiveColumnReader* fieldReader,
      uint64_t version,
      common::ScanSpec::VectorTransform transform)
      : ColumnLoader(structReader, fieldReader, version),
        transform_(std::move(transform)) {}

  bool supportsHook() const override {
    return false;
  }

 private:
  void loadInternal(
      RowSet rows,
      ValueHook* hook,
      vector_size_t resultSize,
      VectorPtr* result) override;

  common::ScanSpec::VectorTransform transform_;
};

class DeltaUpdateColumnLoader : public VectorLoader {
 public:
  DeltaUpdateColumnLoader(
      SelectiveStructColumnReaderBase* structReader,
      SelectiveColumnReader* fieldReader,
      uint64_t version)
      : structReader_(structReader),
        fieldReader_(fieldReader),
        version_(version) {}

 private:
  void loadInternal(
      RowSet rows,
      ValueHook* hook,
      vector_size_t resultSize,
      VectorPtr* result) override;

  SelectiveStructColumnReaderBase* const structReader_;
  SelectiveColumnReader* const fieldReader_;
  const uint64_t version_;
};

} // namespace facebook::velox::dwio::common
