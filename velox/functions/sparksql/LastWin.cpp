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
#include "velox/functions/sparksql/LastWin.h"

#include <bit>
#include <cmath>
#include <limits>

#include <folly/hash/Hash.h>

#include "velox/vector/SimpleVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Returns the bits Java equality compares, collapsing every NaN to one pattern.
template <typename T>
auto javaBits(T value) {
  using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
  return std::bit_cast<Bits>(
      std::isnan(value) ? std::numeric_limits<T>::quiet_NaN() : value);
}

} // namespace

uint64_t SparkMapKeyEquality::hash(vector_size_t position) const {
  const auto* base = keys_->base();
  const auto index = keys_->index(offset_ + position);
  switch (base->typeKind()) {
    case TypeKind::REAL:
      return folly::hasher<uint32_t>{}(
          javaBits(base->asUnchecked<SimpleVector<float>>()->valueAt(index)));
    case TypeKind::DOUBLE:
      return folly::hasher<uint64_t>{}(
          javaBits(base->asUnchecked<SimpleVector<double>>()->valueAt(index)));
    default:
      return base->hashValueAt(index);
  }
}

bool SparkMapKeyEquality::equals(vector_size_t lhs, vector_size_t rhs) const {
  const auto* base = keys_->base();
  const auto lhsIndex = keys_->index(offset_ + lhs);
  const auto rhsIndex = keys_->index(offset_ + rhs);
  switch (base->typeKind()) {
    case TypeKind::REAL: {
      const auto* floats = base->asUnchecked<SimpleVector<float>>();
      return javaBits(floats->valueAt(lhsIndex)) ==
          javaBits(floats->valueAt(rhsIndex));
    }
    case TypeKind::DOUBLE: {
      const auto* doubles = base->asUnchecked<SimpleVector<double>>();
      return javaBits(doubles->valueAt(lhsIndex)) ==
          javaBits(doubles->valueAt(rhsIndex));
    }
    default:
      return base->equalValueAt(base, lhsIndex, rhsIndex);
  }
}

std::string SparkMapKeyEquality::describe(vector_size_t position) const {
  return keys_->base()->toString(keys_->index(offset_ + position));
}

} // namespace facebook::velox::functions::sparksql
