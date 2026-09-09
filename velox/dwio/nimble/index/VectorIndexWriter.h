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

#include <functional>
#include <memory>
#include <span>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/index/VectorIndexConfig.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble::index {

/// Builds a vector similarity search index during file writes.
///
/// Accumulates vectors from each configured column during write() calls, then
/// trains and serializes the indexes during close().
class VectorIndexWriter {
 public:
  virtual ~VectorIndexWriter() = default;

  /// Extracts vectors from the configured column and buffers them.
  virtual void write(const velox::VectorPtr& input) = 0;

  /// Writes each index blob as a metadata section and their directory as an
  /// optional section.
  virtual void close(
      const CreateMetadataSectionFn& createMetadataFn,
      const WriteOptionalSectionFn& writeMetadataFn) = 0;
};

/// Creates a non-null writer for one or more index configurations. The
/// implementation must consume or copy configs during the call and must not
/// retain the span storage.
///
/// Injected through WriterOptions rather than resolved from a global so that a
/// writer which never configures a vector index does not have to link an index
/// implementation, and the heavyweight similarity-search libraries it depends
/// on.
using VectorIndexWriterFactory =
    std::function<std::unique_ptr<VectorIndexWriter>(
        std::span<const VectorIndexConfig> configs,
        const velox::RowTypePtr& inputType,
        velox::memory::MemoryPool* pool)>;

} // namespace facebook::nimble::index
