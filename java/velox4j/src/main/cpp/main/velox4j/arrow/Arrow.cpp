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
#include "velox4j/arrow/Arrow.h"

#include <velox/vector/ComplexVector.h>
#include <velox/vector/arrow/Bridge.h>
#include <memory>
#include <vector>

namespace facebook::velox4j {
using namespace facebook::velox;

namespace {

void sliceChildren(VectorPtr& in) {
  auto* rowBase = in->as<RowVector>();
  if (!rowBase) {
    return;
  }
  for (auto& child : rowBase->children()) {
    if (child->size() > rowBase->size()) {
      // Some Velox operations (E.g., Limit) could result in a
      // RowVector whose children have larger size than itself.
      // So we perform a slice to keep only the data that is
      // in real use.
      child = child->slice(0, rowBase->size());
    }
  }
}

void flatten(VectorPtr& in) {
  BaseVector::flattenVector(in);
}

ArrowOptions makeOptions() {
  ArrowOptions options;
  options.timestampUnit = static_cast<TimestampUnit>(6);
  return options;
}
} // namespace

void fromBaseVectorToArrow(
    VectorPtr vector,
    ArrowSchema* cSchema,
    ArrowArray* cArray) {
  flatten(vector);
  sliceChildren(vector);
  auto options = makeOptions();
  exportToArrow(vector, *cSchema, options);
  exportToArrow(vector, *cArray, vector->pool(), options);
}

VectorPtr fromArrowToBaseVector(
    memory::MemoryPool* pool,
    ArrowSchema* cSchema,
    ArrowArray* cArray) {
  return importFromArrowAsOwner(*cSchema, *cArray, pool);
}
} // namespace facebook::velox4j
