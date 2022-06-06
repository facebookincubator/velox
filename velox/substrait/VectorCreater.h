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

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::substrait {

/// Create different types of vector from velox variants.
template <TypeKind KIND>
VectorPtr setVectorFromVariantsByKind(
    const std::vector<velox::variant>& value,
    memory::MemoryPool* pool);

/// Create Vector from velox variants.
VectorPtr setVectorFromVariants(
    const TypePtr& type,
    const std::vector<velox::variant>& value,
    velox::memory::MemoryPool* pool);

/// Create Array Vector from Vector.
ArrayVectorPtr
toArrayVector(TypePtr type, VectorPtr vector, memory::MemoryPool* pool);

} // namespace facebook::velox::substrait
