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
#include "velox/optimizer/ArenaCache.h"

// #define QG_USE_MALLOC
#define QG_CACHE_ARENA

/// Thread local context and utilities for query planning.

namespace facebook::velox::optimizer {

/// Pointer to an arena allocated interned copy of a null terminated string.
/// Used for identifiers. Allows comparing strings by comparing pointers.
using Name = const char*;

/// Shorthand for a view on an array of T*
template <typename T>
using CPSpan = folly::Range<const T* const*>;

class PlanObject;

using PlanObjectP = PlanObject*;
using PlanObjectCP = const PlanObject*;

struct PlanObjectPHasher {
  size_t operator()(const PlanObjectCP& object) const;
};

struct PlanObjectPComparer {
  bool operator()(const PlanObjectCP& lhs, const PlanObjectCP& rhs) const;
};

struct TypeHasher {
  size_t operator()(const TypePtr& type) const {
    // hash on recursive TypeKind. Structs that differ in field names
    // only or decimals with different precisions will collide, no
    // other collisions expected.
    return type->hashKind();
  }
};

struct TypeComparer {
  bool operator()(const TypePtr& lhs, const TypePtr& rhs) const {
    return *lhs == *rhs;
  }
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
  int32_t newId(PlanObject* object) {
    objects_.push_back(object);
    return objects_.size() - 1;
  }

  /// Allocates 'size' bytes from the arena of 'this'. The allocation lives
  /// until free() is called on it or the arena is destroyed.
  void* allocate(size_t size) {
#ifdef QG_TEST_USE_MALLOC
    // Benchmark-only. Dropping the arena will not free un-free'd allocs.
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
#ifdef QG_TEST_USE_MALLOC
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
  PlanObjectP dedup(PlanObjectP object);

  /// Returns the object associated to 'id'. See newId()
  PlanObjectCP objectAt(int32_t id) {
    return objects_[id];
  }

  PlanObjectP mutableObjectAt(int32_t id) {
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

  // Records the use of a TypePtr in optimization. Returns a canonical
  // representative of the type, allowing pointer equality for exact match.
  // Allows mapping from the Type* back to TypePtr.
  const Type* toType(const velox::TypePtr& type);

  /// Returns the canonical TypePtr corresponding to 'type'. 'type' must have
  /// been previously returned by toType().
  const TypePtr& toTypePtr(const Type* type);

 private:
  TypePtr dedupType(const TypePtr& type);

  velox::HashStringAllocator& allocator_;
  ArenaCache cache_;

  // PlanObjects are stored at the index given by their id.
  std::vector<PlanObjectP> objects_;

  // Set of interned copies of identifiers. insert() into this returns the
  // canonical interned copy of any string. Lifetime is limited to 'allocator_'.
  std::unordered_set<std::string_view> names_;

  // Set for deduplicating planObject trees.
  std::unordered_set<PlanObjectP, PlanObjectPHasher, PlanObjectPComparer>
      deduppedObjects_;

  std::unordered_set<TypePtr, TypeHasher, TypeComparer> deduppedTypes_;
  // Maps raw Type* back to shared TypePtr. Used in toType()() and toTypePtr().
  std::unordered_map<const velox::Type*, velox::TypePtr> toTypePtr_;

  Plan* contextPlan_{nullptr};
  Optimization* optimization_{nullptr};
};

/// Returns a mutable reference to the calling thread's QueryGraphContext.
QueryGraphContext*& queryCtx();

template <class _Tp, class... _Args>
inline _Tp* make(_Args&&... __args) {
  return new (queryCtx()->allocate(sizeof(_Tp)))
      _Tp(std::forward<_Args>(__args)...);
}

/// Macro to use instead of make() when make() errors out from too
/// many arguments.
#define QGC_MAKE_IN_ARENA(_Tp) new (queryCtx()->allocate(sizeof(_Tp))) _Tp

/// Converts std::string to name used in query graph objects. raw pointer to
/// arena allocated const chars.
// Name toName(const std::string& string);
Name toName(std::string_view string);

/// Shorthand for toType() in thread's QueryGraphContext.
const Type* toType(const TypePtr& type);
/// Shorthand for toTypePtr() in thread's QueryGraphContext.
const TypePtr& toTypePtr(const Type* type);

/// STL compatible allocator that manages std:: containers allocated in the
/// QueryGraphContext arena.
template <class T>
struct QGAllocator {
  using value_type = T;
  QGAllocator() = default;

  template <typename U>
  explicit QGAllocator(QGAllocator<U>) {}

  T* allocate(std::size_t n) {
    return reinterpret_cast<T*>(
        queryCtx()->allocate(velox::checkedMultiply(n, sizeof(T)))); // NOLINT
  }

  void deallocate(T* p, std::size_t /*n*/) noexcept {
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
using ExprCP = const Expr*;
class Column;
using ColumnCP = const Column*;
using ExprVector = std::vector<ExprCP, QGAllocator<ExprCP>>;
using ColumnVector = std::vector<ColumnCP, QGAllocator<ColumnCP>>;

} // namespace facebook::velox::optimizer
