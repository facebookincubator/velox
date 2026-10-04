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
#include "velox/functions/sparksql/tests/CanonicalFloatingPoint.h"

#include <cstring>

#include <gtest/gtest.h>

#include "velox/functions/lib/NormalizeFloatingPoint.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

template <typename T>
void expectCanonical(T value) {
  const T canonical = normalizeFloatingPoint(value);
  EXPECT_EQ(std::memcmp(&value, &canonical, sizeof(T)), 0)
      << "Not in canonical form: " << value;
}

void expectCanonicalAt(const BaseVector& vector, vector_size_t row) {
  DecodedVector decoded(vector);
  if (decoded.isNullAt(row)) {
    return;
  }
  const auto* base = decoded.base();
  const auto index = decoded.index(row);
  switch (base->typeKind()) {
    case TypeKind::REAL:
      expectCanonical(base->as<SimpleVector<float>>()->valueAt(index));
      break;
    case TypeKind::DOUBLE:
      expectCanonical(base->as<SimpleVector<double>>()->valueAt(index));
      break;
    case TypeKind::ARRAY: {
      const auto* array = base->as<ArrayVector>();
      for (auto i = 0; i < array->sizeAt(index); ++i) {
        expectCanonicalAt(*array->elements(), array->offsetAt(index) + i);
      }
      break;
    }
    case TypeKind::MAP: {
      const auto* map = base->as<MapVector>();
      for (auto i = 0; i < map->sizeAt(index); ++i) {
        expectCanonicalAt(*map->mapKeys(), map->offsetAt(index) + i);
        expectCanonicalAt(*map->mapValues(), map->offsetAt(index) + i);
      }
      break;
    }
    case TypeKind::ROW:
      for (const auto& child : base->as<RowVector>()->children()) {
        expectCanonicalAt(*child, index);
      }
      break;
    default:
      break;
  }
}

} // namespace

void expectCanonicalFloatingPoint(const BaseVector& vector) {
  for (vector_size_t row = 0; row < vector.size(); ++row) {
    expectCanonicalAt(vector, row);
  }
}

} // namespace facebook::velox::functions::sparksql::test
