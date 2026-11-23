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

#include <memory>
#include <string>
#include <vector>

#include "velox/dwio/nimble/index/DenseIndexFactory.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble::index {

/// Unified registry for built-in and registered dense indices on a Nimble file.
///
/// Loads index payloads referenced by the common index manifest.
class DenseIndexRegistry {
 public:
  struct Entry {
    std::string name;
    Section section;
  };

  /// Creates readers by dispatching each named root section to its registered
  /// factory. Returns nullptr if entries is empty.
  static std::unique_ptr<DenseIndexRegistry> create(
      std::vector<Entry> entries,
      const IndexLookup::Options& options,
      velox::memory::MemoryPool* pool);

  /// Finds any dense index matching the given columns.
  const IndexLookup* findIndex(
      const std::vector<std::string>& queryColumns) const;

  /// Finds the dense index with the given implementation name and columns.
  const IndexLookup* findIndex(
      std::string_view name,
      const std::vector<std::string>& queryColumns) const;

 private:
  DenseIndexRegistry() = default;

  struct DenseIndex {
    std::string name;
    std::unique_ptr<DenseIndexReader> reader;
  };
  std::vector<DenseIndex> denseIndices_;
};

} // namespace facebook::nimble::index
