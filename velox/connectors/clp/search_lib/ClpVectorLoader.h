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

#include "clp_s/ColumnReader.hpp"
#include "clp_s/SchemaTree.hpp"

#include "velox/connectors/clp/search_lib/ClpCursor.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/LazyVector.h"

namespace clp_s {
class BaseColumnReader;
} // namespace clp_s

namespace facebook::velox::connector::clp::search_lib {

/// A custom Velox VectorLoader that populates Velox vectors from a CLP-based
/// column reader. It supports various column types including integers, floats,
/// booleans, strings, and arrays of strings.
class ClpVectorLoader : public VectorLoader {
 public:
  ClpVectorLoader(
      clp_s::BaseColumnReader* columnReader,
      ColumnType nodeType,
      std::shared_ptr<std::vector<uint64_t>> filteredRowIndices);

 private:
  void loadInternal(
      RowSet rows,
      ValueHook* hook,
      vector_size_t resultSize,
      VectorPtr* result) override;

  template <typename T, typename VectorPtr>
  void populateData(RowSet rows, VectorPtr vector);

  template <clp_s::NodeType Type>
  void populateTimestampData(
      RowSet rows,
      FlatVector<facebook::velox::Timestamp>* vector);

  clp_s::BaseColumnReader* columnReader_;
  ColumnType nodeType_;
  std::shared_ptr<std::vector<uint64_t>> filteredRowIndices_;

  inline static thread_local std::unique_ptr<simdjson::ondemand::parser>
      arrayParser_ = std::make_unique<simdjson::ondemand::parser>();
};

} // namespace facebook::velox::connector::clp::search_lib
