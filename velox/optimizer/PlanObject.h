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

#include "velox/optimizer/QueryGraphContext.h"

namespace facebook::velox::optimizer {

/// Enum for types of query graph nodes. Used when making a tree into
/// a query graph and later to differentiate between tables, derived
/// tables and different expressions.
enum class PlanType {
  kTable,
  kDerivedTable,
  kColumn,
  kLiteral,
  kCall,
  kAggregate,
  kAggregation,
  kProject,
  kFilter,
  kJoin,
  kOrderBy,
  kLimit
};

/// True if 'type' is an expression with a value.
inline bool isExprType(PlanType type) {
  return type == PlanType::kColumn || type == PlanType::kCall ||
      type == PlanType::kLiteral;
}

/// Common superclass of all vertices of a query graph. This
/// includes tables, columns, expressions, derived tables etc. These
/// all have a unique small integer id. Ids are often to track
/// membership in PlanObjectSets. These stand for e.g. the columns
/// assigned by an operator or the tables in a partial plan.  Joins
/// are edges of the graph but do not have ids, rather, they are
/// identified by the ids of their end points. PlanObjects are created at the
/// start of planning and are arena allocated to be all dropped when the
/// planning is complete.
class PlanObject {
 public:
  explicit PlanObject(PlanType _type)
      : type_(_type), id_(queryCtx()->newId(this)) {}

  virtual ~PlanObject() = default;

  void operator delete(void* /*ptr*/) {
    LOG(FATAL) << "Plan objects are not deletable";
  }

  int32_t id() const {
    return id_;
  }

  PlanType type() const {
    return type_;
  }

  template <typename T>
  T* as() {
    return reinterpret_cast<T*>(this);
  }

  /// Returns 'this' as const T.
  template <typename T>
  const T* as() const {
    return reinterpret_cast<const T*>(this);
  }

  /// Returns a view on children, e.g. arguments of a function call.
  virtual CPSpan<PlanObject> children() const {
    return CPSpan<PlanObject>(nullptr, nullptr);
  }

  /// Returns true if 'this' is an expression with a value.
  virtual bool isExpr() const {
    return false;
  }

  /// Returns true if this is a table or derived table.
  virtual bool isTable() const {
    return false;
  }

  /// Returns true if 'this' is a function call or aggregate function call.
  virtual bool isFunction() const {
    return false;
  }

  size_t hash() const;

  virtual std::string toString() const {
    return fmt::format("#{}", id_);
  }

 private:
  const PlanType type_;
  const int32_t id_;
};

/// Set of PlanObjects. Uses the objects id() as an index into a bitmap.
class PlanObjectSet {
 public:
  /// True if id of 'object' is in 'this'.
  bool contains(PlanObjectCP object) const {
    return object->id() < bits_.size() * 64 &&
        velox::bits::isBitSet(bits_.data(), object->id());
  }

  bool operator==(const PlanObjectSet& other) const;

  size_t hash() const;

  // True if no members.
  bool empty() const {
    for (auto word : bits_) {
      if (word) {
        return false;
      }
    }
    return true;
  }

  /// Inserts id of 'object'.
  void add(PlanObjectCP object) {
    auto id = object->id();
    ensureSize(id);
    velox::bits::setBit(bits_.data(), id);
  }

  /// Returns true if 'this' is a subset of 'super'.
  bool isSubset(const PlanObjectSet& super) const;

  /// Erases id of 'object'.
  void erase(PlanObjectCP object) {
    if (object->id() < bits_.size() * 64) {
      velox::bits::clearBit(bits_.data(), object->id());
    }
  }

  void except(const PlanObjectSet& other) {
    velox::bits::forEachSetBit(
        other.bits_.data(), 0, other.bits_.size() * 64, [&](auto id) {
          if (id < bits_.size() * 64) {
            velox::bits::clearBit(bits_.data(), id);
          }
        });
  }

  /// Adds ids of all columns 'expr' depends on.
  void unionColumns(ExprCP expr);

  /// Adds ids of all columns 'exprs' depend on.
  void unionColumns(const ExprVector& exprs);
  void unionColumns(const ColumnVector& exprs);

  /// Adds all ids in 'other'.
  void unionSet(const PlanObjectSet& other);

  /// Erases all ids not in 'other'.
  void intersect(const PlanObjectSet& other);

  /// Adds ids of all objects in 'objects'.
  template <typename V>
  void unionObjects(const V& objects) {
    for (auto& object : objects) {
      add(object);
    }
  }

  /// Applies 'func' to each object in 'this'.
  template <typename Func>
  void forEach(Func func) const {
    auto ctx = queryCtx();
    velox::bits::forEachSetBit(bits_.data(), 0, bits_.size() * 64, [&](auto i) {
      func(ctx->objectAt(i));
    });
  }

  template <typename Func>
  void forEachMutable(Func func) const {
    auto ctx = queryCtx();
    velox::bits::forEachSetBit(bits_.data(), 0, bits_.size() * 64, [&](auto i) {
      func(ctx->mutableObjectAt(i));
    });
  }

  /// Returns the objects corresponding to ids in 'this' as a vector of T.
  template <typename T = PlanObjectP>
  std::vector<T> objects() const {
    std::vector<T> result;
    forEach(
        [&](auto object) { result.push_back(reinterpret_cast<T>(object)); });
    return result;
  }

  /// Prnts the contents with ids and the string representation of the objects
  /// if 'names' is true.
  std::string toString(bool names) const;

 private:
  void ensureSize(int32_t id) {
    ensureWords(velox::bits::nwords(id + 1));
  }

  void ensureWords(int32_t size) {
    if (bits_.size() < size) {
      bits_.resize(size);
    }
  }

  // A one bit corresponds to the id of each member.
  std::vector<uint64_t, QGAllocator<uint64_t>> bits_;
};

using PlanObjectVector = std::vector<PlanObjectCP, QGAllocator<PlanObjectCP>>;

} // namespace facebook::velox::optimizer

namespace std {
template <>
struct hash<::facebook::velox::optimizer::PlanObjectSet> {
  size_t operator()(
      const ::facebook::velox::optimizer::PlanObjectSet& set) const {
    return set.hash();
  }
};
} // namespace std
