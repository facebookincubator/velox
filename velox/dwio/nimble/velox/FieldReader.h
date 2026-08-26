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

#include <folly/Executor.h>
#include <folly/container/F14Map.h>
#include <folly/coro/Task.h>

#include "velox/common/memory/MemoryPool.h"
#include "velox/dwio/common/FlatMapHelper.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/Decoder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble {

enum class SelectionMode {
  Include = 0,
  Exclude = 1,
};

struct FeatureSelection {
  std::vector<std::string> features;
  /// When mode == Include, only features appearing in 'features' will be
  /// included in returned map, otherwise,
  /// all features from the file will be returned in the map, excluding
  /// the features appearing in 'features'.
  SelectionMode mode{SelectionMode::Include};
};

struct FieldReaderParams {
  /// Allow selecting subset of features to be included/excluded in flat maps.
  /// The key in the map is the flat map (top-level) column name.
  folly::F14FastMap<std::string, FeatureSelection> flatMapFeatureSelector;

  /// Contains flatmap field name which we want to return as Struct
  folly::F14FastSet<std::string> readFlatMapFieldAsStruct;

  /// Callback to populate feature projection stats when needed
  std::function<void(velox::dwio::common::flatmap::FlatMapKeySelectionStats)>
      keySelectionCallback{nullptr};

  bool optimizeStringBufferHandling{false};

  /// Executor for parallel decoding of child fields.
  folly::Executor* decodeExecutor{nullptr};

  /// Maximum number of parallel coroutine tasks for child field decoding.
  /// Children are grouped into this many batches, each decoded sequentially
  /// within a single coroutine task. 0 disables parallel decoding.
  uint32_t maxDecodeParallelism{0};

  /// Minimum number of child streams per parallel decode task. Ensures each
  /// coroutine task has enough work to amortize threading overhead.
  uint32_t minStreamsPerDecodeUnit{1};
};

class FieldReader {
 public:
  struct Options {
    folly::Executor* decodeExecutor{nullptr};
    uint32_t maxDecodeParallelism{0};
    uint32_t minStreamsPerDecodeUnit{1};
  };

  FieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder);

  FieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder,
      const Options& options);

  virtual ~FieldReader() = default;

  /// Estimation of the per row size of the field on current reading stripe in
  /// bytes. Returns a pair containing the number of rows of the field and
  /// average row size in bytes of the field. This method will return nullopt if
  /// the field or encoding is not supported for estimation.
  ///
  /// NOTE: This is not the estimation based on the remaining rows, but the
  /// entire stripe's rows.
  virtual std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize()
      const = 0;

  /// Place the next 'count' rows of data into the passed in output vector.
  ///
  /// NOTE: scatterBitmap is not for external selectivity. External callers must
  /// leave scatterBitmap nullptr
  virtual void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap = nullptr) = 0;

  /// Coroutine version of next(). Used for parallel decoding: parent readers
  /// co_await children's co_next() via collectAllRange, yielding their thread
  /// back to the executor. This prevents deadlock from nested parallelism.
  /// Default implementation wraps the synchronous next() call.
  virtual folly::coro::Task<void> co_next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap = nullptr) {
    next(count, output, scatterBitmap);
    co_return;
  }

  virtual void skip(uint32_t count) = 0;

  /// Called at the end of stripe
  virtual void reset();

  const velox::TypePtr& type() const {
    return type_;
  }

 protected:
  void ensureNullConstant(
      const std::shared_ptr<const velox::Type>& type,
      uint32_t count,
      velox::VectorPtr& output) const;

  // Computes the number of parallel decode tasks based on max parallelism and
  // minimum streams per task. The result is clamped to [1, numStreamChildren].
  uint32_t computeParallelDecodeTaskCount(uint32_t numStreamChildren) const;

  // Returns true if child fields should be decoded in parallel.
  // Requires at least 2 parallel tasks to justify coroutine/executor overhead.
  bool parallelDecodeEnabled(uint32_t numChildren) const {
    return decodeExecutor_ != nullptr &&
        computeParallelDecodeTaskCount(numChildren) > 1;
  }

  velox::memory::MemoryPool* const pool_;
  const velox::TypePtr type_;
  Decoder* const decoder_;
  folly::Executor* const decodeExecutor_;
  const uint32_t maxDecodeParallelism_;
  const uint32_t minStreamsPerDecodeUnit_;
};

class FieldReaderFactory {
 public:
  FieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* nimbleType,
      velox::memory::MemoryPool* pool)
      : pool_{pool},
        veloxType_{std::move(veloxType)},
        nimbleType_{nimbleType} {}

  virtual ~FieldReaderFactory() = default;

  virtual std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>&
          decoders) = 0;

  const velox::TypePtr& veloxType() const {
    return veloxType_;
  }

  /// Build a field reader factory tree. Will traverse the passed in types and
  /// create matching field readers.
  static std::unique_ptr<FieldReaderFactory> create(
      const FieldReaderParams& parameters,
      const std::shared_ptr<const nimble::Type>& nimbleType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& veloxType,
      std::vector<uint32_t>& offsets,
      const std::function<bool(uint32_t)>& isSelected,
      velox::memory::MemoryPool* pool);

 protected:
  std::unique_ptr<FieldReader> createNullColumnReader() const;

  Decoder* getDecoder(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders,
      const StreamDescriptor& streamDescriptor) const;

  template <typename T, typename... Args>
  std::unique_ptr<FieldReader> createReaderImpl(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders,
      const StreamDescriptor& nullsDecriptor,
      Args&&... args) const;

  velox::memory::MemoryPool* const pool_;
  const velox::TypePtr veloxType_;
  const Type* nimbleType_;
};

} // namespace facebook::nimble
