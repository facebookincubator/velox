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

#include "velox/common/base/SimdUtil.h"
#include "velox/optimizer/QueryGraph.h"

namespace facebook::velox::optimizer {

size_t PlanObjectPHasher::operator()(const PlanObjectCP& object) const {
  return object->hash();
}

bool PlanObjectPComparer::operator()(
    const PlanObjectCP& lhs,
    const PlanObjectCP& rhs) const {
  if (rhs == lhs) {
    return true;
  }
  return rhs && lhs && lhs->isExpr() && rhs->isExpr() &&
      reinterpret_cast<const Expr*>(lhs)->sameOrEqual(
          *reinterpret_cast<const Expr*>(rhs));
}

size_t PlanObject::hash() const {
  size_t h = static_cast<size_t>(id_);
  for (auto& child : children()) {
    h = velox::bits::hashMix(h, child->hash());
  }
  return h;
}

namespace {
template <typename V>
bool isZero(const V& bits, size_t begin, size_t end) {
  for (size_t i = begin; i < end; ++i) {
    if (bits[i]) {
      return false;
    }
  }
  return true;
}
} // namespace

bool PlanObjectSet::operator==(const PlanObjectSet& other) const {
  // The sets are equal if they have the same bits set. Trailing words of zeros
  // do not count.
  auto l1 = bits_.size();
  auto l2 = other.bits_.size();
  for (unsigned i = 0; i < l1 && i < l2; ++i) {
    if (bits_[i] != other.bits_[i]) {
      return false;
    }
  }
  if (l1 < l2) {
    return isZero(other.bits_, l1, l2);
  }
  if (l2 < l1) {
    return isZero(bits_, l2, l1);
  }
  return true;
}

bool PlanObjectSet::isSubset(const PlanObjectSet& super) const {
  auto l1 = bits_.size();
  auto l2 = super.bits_.size();
  for (unsigned i = 0; i < l1 && i < l2; ++i) {
    if (bits_[i] & ~super.bits_[i]) {
      return false;
    }
  }
  if (l2 < l1) {
    return isZero(bits_, l2, l1);
  }
  return true;
}

size_t PlanObjectSet::hash() const {
  // The hash is a mix of the hashes of all non-zero words.
  size_t hash = 123;
  for (unsigned i = 0; i < bits_.size(); ++i) {
    hash = velox::simd::crc32U64(hash, bits_[i]);
  }
  return hash * hash;
}

void PlanObjectSet::unionColumns(ExprCP expr) {
  switch (expr->type()) {
    case PlanType::kLiteral:
      return;
    case PlanType::kColumn:
      add(expr);
      return;
    case PlanType::kAggregate: {
      auto condition = expr->as<Aggregate>()->condition();
      if (condition) {
        unionColumns(condition);
      }
    }
      // Fall through.
      FMT_FALLTHROUGH;
    case PlanType::kCall: {
      auto call = reinterpret_cast<const Call*>(expr);
      unionSet(call->columns());
      return;
    }
    default:
      VELOX_UNREACHABLE();
  }
}

void PlanObjectSet::unionColumns(const ExprVector& exprs) {
  for (auto& expr : exprs) {
    unionColumns(expr);
  }
}

void PlanObjectSet::unionColumns(const ColumnVector& exprs) {
  for (auto& expr : exprs) {
    unionColumns(expr);
  }
}

void PlanObjectSet::unionSet(const PlanObjectSet& other) {
  ensureWords(other.bits_.size());
  for (auto i = 0; i < other.bits_.size(); ++i) {
    bits_[i] |= other.bits_[i];
  }
}

void PlanObjectSet::intersect(const PlanObjectSet& other) {
  bits_.resize(std::min(bits_.size(), other.bits_.size()));
  for (auto i = 0; i < bits_.size(); ++i) {
    assert(!other.bits_.empty());
    bits_[i] &= other.bits_[i];
  }
}

std::string PlanObjectSet::toString(bool names) const {
  std::stringstream out;
  forEach([&](auto object) {
    out << object->id();
    if (names) {
      out << ": " << object->toString() << std::endl;
    } else {
      out << " ";
    }
  });
  return out.str();
}

} // namespace facebook::velox::optimizer
