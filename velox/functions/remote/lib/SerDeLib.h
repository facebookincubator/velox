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

#include <folly/io/IOBuf.h>

#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::functions {

folly::IOBuf serializeIOBuf(
    const RowVectorPtr rowVector,
    vector_size_t rangeEnd,
    memory::MemoryPool& pool);

RowVectorPtr deserializeIOBuf(
    const folly::IOBuf& ioBuf,
    const RowTypePtr& outputType,
    memory::MemoryPool& pool);
} // namespace facebook::velox::functions
