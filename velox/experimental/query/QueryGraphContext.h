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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/core/PlanNode.h"
#include "velox/experimental/query/ArenaCache.h"

// #define QG_USE_MALLOC
#define QG_CACHE_ARENA

/// Thread local context and utilities for query planning.

namespace facebook::verax {

/// Pointer to an arena allocated interned copy of a null terminated string.
/// Used for identifiers. Allows comparing strings by comparing pointers.
using Name = const char*;

/// Shorthand for a view on an array of T*
template <typename T>
using PtrSpan = folly::Range<const T* const*>;

class PlanObject;

using PlanObjectPtr = PlanObject*;
using PlanObjectConstPtr = const PlanObject*;

struct PlanObjectPtrHasher {
  size_t operator()(const PlanObjectConstPtr& object) const;
};

struct PlanObjectPtrComparer {
  bool operator()(const PlanObjectConstPtr& lhs, const PlanObjectConstPtr& rhs)
      const;
};

struct Plan;
using PlanPtr = Plan*;
class Optimization;

/// Context for making a query plan. Owns all memory associated to
/// planning, except for the input PlanNode tree. The result of
/// planning is also owned by 'this', so the planning result must be
/// copied into a non-owned target specific representation before
/// destroying 'this'. QueryGraphContext is not thread safe and may
/// be accessed from one thread at a time. Memory allocation
/// references this via a thread local through queryCtx().
class QueryGraphContext {
 public:
  explicit QueryGraphContext(velox::HashStringAllocator& allocator)
      : allocator_(allocator), cache_(allocator_) {}

  /// Returns the interned representation of 'str', i.e. Returns a
  /// pointer to a canonical null terminated const char* with the same
  /// characters as 'str'. Allows comparing names by comparing
  /// pointers.
  Name toName(std::string_view str);

  /// Returns a new unique id to use for 'object' and associates 'object' to
  /// this id. Tagging objects with integere ids is useful for efficiently
  /// representing sets of objects as bitmaps.
  int32_t newId(PlanObject* FOLLY_NONNULL object) {
    objects_.push_back(object);
    return objects_.size() - 1;
  }

  /// Allocates 'size' bytes from the arena of 'this'. The allocation lives
  /// until free() is called on it or the arena is destroyed.
  void* allocate(size_t size) {
#ifdef QG_USE_MALLOC
    return ::malloc(size);
#elif defined(QG_CACHE_ARENA)
    return cache_.allocate(size);
#else
    return allocator_.allocate(size)->begin();
#endif
  }

  /// Frees ptr, which must have been allocated with allocate() above. Calling
  /// this is not mandatory since objects from the arena get freed at latest
  /// when the arena is destroyed.
  void free(void* ptr) {
#ifdef QG_USE_MALLOC
    ::free(ptr);
#elif defined(QG_CACHE_ARENA)
    cache_.free(ptr);
#else
    allocator_.free(velox::HashStringAllocator::headerOf(ptr));
#endif
  }

  /// Returns a canonical instance for all logically equal values of 'object'.
  /// Returns 'object' on first call with object, thereafter the same physical
  /// object if the argument is equal.
  PlanObjectPtr dedup(PlanObjectPtr object);

  /// Returns the object associated to 'id'. See newId()
  PlanObjectConstPtr objectAt(int32_t id) {
    return objects_[id];
  }

  PlanObjectPtr mutableObjectAt(int32_t id) {
    return objects_[id];
  }

  /// Returns the top level plan being processed when printing operator trees.
  /// If non-null, allows showing percentages.
  Plan*& contextPlan() {
    return contextPlan_;
  }

  /// The top level Optimization instance.
  Optimization*& optimization() {
    return optimization_;
  }

 private:
  velox::HashStringAllocator& allocator_;
  ArenaCache cache_;

  // PlanObjects are stored at the index given by their id.
  std::vector<PlanObjectPtr> objects_;

  // Set of interned copies of identifiers. insert() into this returns the
  // canonical interned copy of any string. Lifetime is limited to 'allocator_'.
  std::unordered_set<std::string_view> names_;

  // Set for deduplicating planObject trees.
  std::unordered_set<PlanObjectPtr, PlanObjectPtrHasher, PlanObjectPtrComparer>
      deduppedObjects_;
  Plan* FOLLY_NULLABLE contextPlan_{nullptr};
  Optimization* FOLLY_NULLABLE optimization_{nullptr};
};

/// Returns a mutable reference to the calling thread's QueryGraphContext.
QueryGraphContext*& queryCtx();

/// Declares 'destination' as a pointer to T, allocated from the thread's
/// QueryGraphContext arena. The remaining arguments are passed to the
/// constructor of T.
#define Declare(T, destination, ...)                                      \
  T* destination = reinterpret_cast<T*>(queryCtx()->allocate(sizeof(T))); \
  new (destination) T(__VA_ARGS__);

/// Converts std::string to name used in query graph objects. raw pointer to
/// arena allocated const chars.
Name toName(const std::string& string);

/// STL compatible allocator that manages std:: containers allocated in the
/// QueryGraphContext arena.
template <class T>
struct QGAllocator {
  using value_type = T;
  QGAllocator() = default;

  template <typename U>
  explicit QGAllocator(QGAllocator<U>) {}

  T* FOLLY_NONNULL allocate(std::size_t n) {
    return reinterpret_cast<T*>(
        queryCtx()->allocate(velox::checkedMultiply(n, sizeof(T)))); // NOLINT
  }

  void deallocate(T* FOLLY_NONNULL p, std::size_t /*n*/) noexcept {
    queryCtx()->free(p);
  }

  friend bool operator==(
      const QGAllocator& /*lhs*/,
      const QGAllocator& /*rhs*/) {
    return true;
  }

  friend bool operator!=(const QGAllocator& lhs, const QGAllocator& rhs) {
    return !(lhs == rhs);
  }
};

// Forward declarations of common types and collections.
class Expr;
using ExprPtr = const Expr*;
class Column;
using ColumnPtr = const Column*;
using ExprVector = std::vector<ExprPtr, QGAllocator<ExprPtr>>;
using ColumnVector = std::vector<ColumnPtr, QGAllocator<ColumnPtr>>;

} // namespace facebook::verax
