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

#include "velox/functions/sparksql/JavaFloatingPoint.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::functions::sparksql {

uint64_t SparkMapKeyEquality::hash(vector_size_t position) const {
  const auto* base = keys_->base();
  const auto index = keys_->index(offset_ + position);
  switch (base->typeKind()) {
    case TypeKind::REAL:
      return JavaFloatingPoint::Hash<float>{}(
          base->asUnchecked<SimpleVector<float>>()->valueAt(index));
    case TypeKind::DOUBLE:
      return JavaFloatingPoint::Hash<double>{}(
          base->asUnchecked<SimpleVector<double>>()->valueAt(index));
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
      return JavaFloatingPoint::Equals<float>{}(
          floats->valueAt(lhsIndex), floats->valueAt(rhsIndex));
    }
    case TypeKind::DOUBLE: {
      const auto* doubles = base->asUnchecked<SimpleVector<double>>();
      return JavaFloatingPoint::Equals<double>{}(
          doubles->valueAt(lhsIndex), doubles->valueAt(rhsIndex));
    }
    default:
      return base->equalValueAt(base, lhsIndex, rhsIndex);
  }
}

std::string SparkMapKeyEquality::describe(vector_size_t position) const {
  return keys_->base()->toString(keys_->index(offset_ + position));
}

} // namespace facebook::velox::functions::sparksql
