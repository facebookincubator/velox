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

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include "velox/common/memory/MappedMemory.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/Context.h"
#include "velox/core/QueryConfig.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::core {

class QueryCtx : public Context {
 public:
  // Returns QueryCtx with new executor created if not supplied. For testing
  // purpose.
  static std::shared_ptr<QueryCtx> createForTest(
      std::shared_ptr<Config> config = std::make_shared<MemConfig>(),
      std::shared_ptr<folly::Executor> executor =
          std::make_shared<folly::CPUThreadPoolExecutor>(
              std::thread::hardware_concurrency())) {
    return std::make_shared<QueryCtx>(std::move(executor), std::move(config));
  }

  // QueryCtx is used in different places. When used with `Task`, it's required
  // that callers supply executors. In contrast, when used in expression
  // evaluation through `ExecCtx`, executor is not needed. Hence, we don't
  // require executor to always be passed in here, but instead, ensure that
  // executor exists when actually being used.
  //
  // This constructor keeps the ownership of `executor`.
  QueryCtx(
      std::shared_ptr<folly::Executor> executor = nullptr,
      std::shared_ptr<Config> config = std::make_shared<MemConfig>(),
      std::unordered_map<std::string, std::shared_ptr<Config>>
          connectorConfigs = {},
      memory::MappedMemory* mappedMemory = memory::MappedMemory::getInstance(),
      std::unique_ptr<memory::MemoryPool> pool = nullptr)
      : Context{ContextScope::QUERY},
        pool_(std::move(pool)),
        mappedMemory_(mappedMemory),
        connectorConfigs_(connectorConfigs),
        executor_{std::move(executor)},
        config_{this} {
    setConfigOverrides(config);
    if (!pool_) {
      initPool();
    }
  }

  // Constructor to block the destruction of executor while this
  // object is alive.
  //
  // This constructor does not keep the ownership of executor.
  explicit QueryCtx(
      folly::Executor::KeepAlive<> executorKeepalive,
      std::shared_ptr<Config> config = std::make_shared<MemConfig>(),
      std::unordered_map<std::string, std::shared_ptr<Config>>
          connectorConfigs = {},
      memory::MappedMemory* mappedMemory = memory::MappedMemory::getInstance(),
      std::unique_ptr<memory::MemoryPool> pool = nullptr)
      : Context{ContextScope::QUERY},
        pool_(std::move(pool)),
        mappedMemory_(mappedMemory),
        connectorConfigs_(connectorConfigs),
        executorKeepalive_(std::move(executorKeepalive)),
        config_{this} {
    setConfigOverrides(config);
    if (!pool_) {
      initPool();
    }
  }

  memory::MemoryPool* pool() const {
    return pool_.get();
  }

  memory::MappedMemory* mappedMemory() const {
    return mappedMemory_;
  }

  folly::Executor* executor() const {
    if (executor_) {
      return executor_.get();
    }
    auto executor = executorKeepalive_.get();
    VELOX_CHECK(executor, "Executor was not supplied.");
    return executor;
  }

  const QueryConfig& config() const {
    return config_;
  }

  Config* getConnectorConfig(const std::string& connectorId) const {
    auto it = connectorConfigs_.find(connectorId);
    if (it == connectorConfigs_.end()) {
      return getEmptyConfig();
    }
    return it->second.get();
  }

  // Multiple logical servers (hosts) can be colocated in one
  // process. This returns the logical host on behalf of which the
  // Task referencing this is running. This is used as a key to select
  // the appropriate host-level singleton resource for different
  // purposes, e.g. memory or outgoing exchange buffers.
  std::string host() const {
    static std::string local = "local";
    return get<std::string>("host", local);
  }

  // Overrides the previous configuration. Note that this function is NOT
  // thread-safe and should probably only be used in tests.
  void setConfigOverridesUnsafe(
      std::unordered_map<std::string, std::string>&& configOverrides) {
    setConfigOverrides(
        std::make_shared<const MemConfig>(std::move(configOverrides)));
  }

 private:
  static Config* getEmptyConfig() {
    static const std::unique_ptr<Config> kEmptyConfig =
        std::make_unique<MemConfig>();
    return kEmptyConfig.get();
  }

  void initPool() {
    pool_ = memory::getProcessDefaultMemoryManager().getRoot().addScopedChild(
        kQueryRootMemoryPool);
    static const auto kUnlimited = std::numeric_limits<int64_t>::max();
    pool_->setMemoryUsageTracker(
        memory::MemoryUsageTracker::create(kUnlimited, kUnlimited, kUnlimited));
  }

  static constexpr const char* kQueryRootMemoryPool = "query_root";

  std::unique_ptr<memory::MemoryPool> pool_;
  memory::MappedMemory* mappedMemory_;
  std::unordered_map<std::string, std::shared_ptr<Config>> connectorConfigs_;
  std::shared_ptr<folly::Executor> executor_;
  folly::Executor::KeepAlive<> executorKeepalive_;
  QueryConfig config_;
};

// A thread-level cache of preallocated flat vectors of different types.
class VectorPool {
 public:
  VectorPtr
  get(const TypePtr& type, vector_size_t size, memory::MemoryPool& pool) {
    auto kind = static_cast<int32_t>(type->kind());
    if (kind < kNumCachedVectorTypes) {
      return vectors_[kind].pop(type, size, pool);
    }
    return BaseVector::create(type, size, &pool);
  }

  // Moves vector into 'this' if it is flat, recursively singly referenced and
  // there is space.
  void release(VectorPtr& vector) {
    auto kind = static_cast<int32_t>(vector->typeKind());
    if (kind < kNumCachedVectorTypes) {
      vectors_[kind].maybe_push_back(vector);
    }
  }

 private:
  static constexpr int32_t kNumCachedVectorTypes =
      static_cast<int32_t>(TypeKind::ARRAY);
  static constexpr int32_t kNumPerType = 10;
  struct TypePool {
    int32_t size{0};
    std::array<VectorPtr, kNumPerType> vectors;

    void maybe_push_back(VectorPtr& vector) {
      if (!vector->isRecyclable()) {
        return;
      }
      if (size < kNumPerType) {
        if (vector->typeKind() == TypeKind::VARCHAR ||
            vector->typeKind() == TypeKind::VARBINARY) {
          vector->asUnchecked<FlatVector<StringView>>()
              ->stringBuffers()
              .clear();
        }
        vectors[size++] = std::move(vector);
      }
    }

    VectorPtr pop(
        const TypePtr& type,
        vector_size_t vectorSize,
        memory::MemoryPool& pool) {
      if (size) {
        auto result = std::move(vectors[--size]);
        if (UNLIKELY(result->rawNulls() != nullptr)) {
          // This is a recyclable vector, no need to check uniqueness.
          simd::memset(
              const_cast<uint64_t*>(result->rawNulls()),
              bits::kNotNull,
              bits::roundUp(std::min<int32_t>(vectorSize, result->size()), 64) /
                  8);
        }
        if (UNLIKELY(
                result->typeKind() == TypeKind::VARCHAR ||
                result->typeKind() == TypeKind::VARBINARY)) {
          simd::memset(
              const_cast<void*>(result->valuesAsVoid()),
              0,
              std::min<int32_t>(vectorSize, result->size()) *
                  sizeof(StringView));
          result->asUnchecked<FlatVector<StringView>>()
              ->stringBuffers()
              .clear();
        }
        if (result->size() != vectorSize) {
          result->resize(vectorSize);
        }
        return result;
      }
      return BaseVector::create(type, vectorSize, &pool);
    }
  };

  // Caches of preallocated vectors indexed by typeKind.
  std::array<TypePool, kNumCachedVectorTypes> vectors_;
};

// Represents the state of one thread of query execution.
class ExecCtx : public Context {
 public:
  ExecCtx(memory::MemoryPool* pool, QueryCtx* queryCtx)
      : Context{ContextScope::QUERY}, pool_(pool), queryCtx_(queryCtx) {}

  velox::memory::MemoryPool* pool() const {
    return pool_;
  }

  QueryCtx* queryCtx() const {
    return queryCtx_;
  }

  /// Returns a SelectivityVector from a pool. Allocates new one if none is
  /// available. Make sure to call 'releaseSelectivityVector' when done using
  /// the vector to allow for reuse.
  ///
  /// Prefer using LocalSelectivityVector which takes care of returning the
  /// vector to the pool on destruction.
  std::unique_ptr<SelectivityVector> getSelectivityVector(int32_t size) {
    if (selectivityVectorPool_.empty()) {
      return std::make_unique<SelectivityVector>(size);
    }
    auto vector = std::move(selectivityVectorPool_.back());
    selectivityVectorPool_.pop_back();
    vector->resize(size);
    return vector;
  }

  void releaseSelectivityVector(std::unique_ptr<SelectivityVector>&& vector) {
    selectivityVectorPool_.push_back(std::move(vector));
  }

  std::unique_ptr<DecodedVector> getDecodedVector() {
    if (decodedVectorPool_.empty()) {
      return std::make_unique<DecodedVector>();
    }
    auto vector = std::move(decodedVectorPool_.back());
    decodedVectorPool_.pop_back();
    return vector;
  }

  void releaseDecodedVector(std::unique_ptr<DecodedVector>&& vector) {
    decodedVectorPool_.push_back(std::move(vector));
  }

  VectorPtr getVector(const TypePtr& type, vector_size_t size) {
    return vectorPool_.get(type, size, *pool_);
  }

  void releaseVector(VectorPtr& vector) {
    vectorPool_.release(vector);
  }

  void releaseVectors(std::vector<VectorPtr>& vectors) {
    for (auto& vector : vectors) {
      vectorPool_.release(vector);
    }
  }

 private:
  // Pool for all Buffers for this thread
  memory::MemoryPool* pool_;
  QueryCtx* queryCtx_;
  // A pool of preallocated DecodedVectors for use by expressions and operators.
  std::vector<std::unique_ptr<DecodedVector>> decodedVectorPool_;
  // A pool of preallocated SelectivityVectors for use by expressions
  // and operators.
  std::vector<std::unique_ptr<SelectivityVector>> selectivityVectorPool_;

  VectorPool vectorPool_;
};

} // namespace facebook::velox::core
