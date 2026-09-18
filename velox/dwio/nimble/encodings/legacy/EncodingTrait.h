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

#include "velox/dwio/nimble/encodings/common/Encoding.h"

namespace facebook::nimble::legacy {

/// Forward declaration of the legacy dispatch function.
/// The full definition (which needs all legacy encoding headers) lives in
/// EncodingUtils.h and is only included by .cpp files that instantiate the
/// template.
template <typename DecoderVisitor>
void callReadWithVisitor(
    Encoding& encoding,
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params);

/// Encoding trait for legacy encodings. Dispatches to
/// legacy::callReadWithVisitor which casts to legacy concrete encoding types.
struct LegacyEncodingTrait {
  template <typename V>
  static void callReadWithVisitor(
      Encoding& encoding,
      V& visitor,
      ReadWithVisitorParams& params) {
    legacy::callReadWithVisitor(encoding, visitor, params);
  }
};

} // namespace facebook::nimble::legacy
