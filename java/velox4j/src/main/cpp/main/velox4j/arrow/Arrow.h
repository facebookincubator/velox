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

#include <velox/vector/BaseVector.h>
#include <velox/vector/ComplexVector.h>

struct ArrowArray;
struct ArrowSchema;

namespace facebook::velox4j {

// Exports the input base vector to Arrow ABI structs.
void fromBaseVectorToArrow(
    facebook::velox::VectorPtr vector,
    ArrowSchema* cSchema,
    ArrowArray* cArray);

// Imports the given Arrow ABI structs into a base vector, then returns it.
facebook::velox::VectorPtr fromArrowToBaseVector(
    facebook::velox::memory::MemoryPool* pool,
    ArrowSchema* cSchema,
    ArrowArray* cArray);
} // namespace facebook::velox4j
