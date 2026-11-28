/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/encodings/BlockBitPackingV2Encoding.h"

namespace facebook::nimble::deprecated {

/// The first BlockBitPacking layout (EncodingType::BlockBitPacking). Its header
/// carries the row count of the first block, which may be shorter than the
/// block size. BlockBitPackingV2Encoding drops that field.
template <typename T>
using BlockBitPackingEncoding =
    BlockBitPackingEncodingImpl<T, BlockBitPackingLayout::kV1>;

} // namespace facebook::nimble::deprecated
