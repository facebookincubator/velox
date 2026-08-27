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

#include "velox/dwio/nimble/encodings/common/BufferedEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

// Legacy encoding headers use qualified names like detail::readWithVisitorSlow.
// Since facebook::nimble::legacy::detail exists (in EncodingUtils.h), qualified
// lookup for detail:: resolves to legacy::detail:: and does NOT fall back to
// nimble::detail::. These using-declarations import the needed names from
// nimble::detail into legacy::detail so that qualified lookup succeeds.

namespace facebook::nimble::legacy {

// Forward declare legacy::callReadWithVisitor. The definition lives in
// legacy/EncodingUtils.h. This forward declaration is needed because
// DictionaryEncoding.h and NullableEncoding.h call legacy::callReadWithVisitor
// in their template methods, and those headers are included before the full
// definition is available.
template <typename V>
void callReadWithVisitor(
    Encoding& encoding,
    V& visitor,
    ReadWithVisitorParams& params);

namespace detail {

using ::facebook::nimble::detail::BufferedDictEncoding;
using ::facebook::nimble::detail::BufferedEncoding;
using ::facebook::nimble::detail::castFromPhysicalType;
using ::facebook::nimble::detail::dataToValue;
using ::facebook::nimble::detail::mutableValues;
using ::facebook::nimble::detail::readWithVisitorFast;
using ::facebook::nimble::detail::readWithVisitorSlow;
using ::facebook::nimble::detail::ValueType;

} // namespace detail

} // namespace facebook::nimble::legacy
