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

#include <span>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/index/IndexWriter.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/type/Type.h"

namespace facebook::nimble::index {

class DenseIndexReader {
 public:
  virtual ~DenseIndexReader() = default;

  /// Returns a non-owning pointer valid while this reader remains alive.
  /// Concurrent calls must be safe.
  virtual const IndexLookup* findIndex(
      const std::vector<std::string>& queryColumns) const = 0;

  /// Returns the complete set of non-empty, unique, ordered column sets
  /// accepted by findIndex(). The result must remain unchanged for the
  /// lifetime of this reader.
  virtual std::vector<std::vector<std::string>> indexColumns() const = 0;
};

/// Creates readers for one named dense index implementation.
class DenseIndexFactory {
 public:
  virtual ~DenseIndexFactory() = default;

  virtual std::string_view name() const = 0;

  /// Creates a non-null writer for one or more index configurations. The
  /// implementation must consume or copy configs during this call and must
  /// not retain the span storage.
  virtual std::unique_ptr<IndexWriter> createWriter(
      std::span<const IndexConfig*> configs,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool) const = 0;

  /// Creates a non-null reader. The reader's index column topology must remain
  /// immutable for its lifetime.
  virtual std::unique_ptr<DenseIndexReader> createReader(
      Section rootSection,
      const IndexLookup::Options& options,
      velox::memory::MemoryPool* pool) const = 0;
};

/// Registers a process-wide factory. Call during process initialization,
/// before concurrent factory lookups begin.
void registerDenseIndexFactory(
    std::shared_ptr<const DenseIndexFactory> factory);

/// Returns a process-lifetime pointer, or nullptr for an unknown
/// implementation. Concurrent access after registration is supported.
const DenseIndexFactory* denseIndexFactory(std::string_view name);

} // namespace facebook::nimble::index
