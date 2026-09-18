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

#include "velox/dwio/nimble/encodings/views/BlockBitPackingEncodingView.h"
#include "velox/dwio/nimble/encodings/views/ConstantEncodingView.h"
#include "velox/dwio/nimble/encodings/views/DeltaBlockEncodingView.h"
#include "velox/dwio/nimble/encodings/views/DictionaryEncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/FOREncodingView.h"
#include "velox/dwio/nimble/encodings/views/FixedBitWidthEncodingView.h"
#include "velox/dwio/nimble/encodings/views/HuffmanEncodingView.h"
#include "velox/dwio/nimble/encodings/views/MainlyConstantEncodingView.h"
#include "velox/dwio/nimble/encodings/views/PFOREncodingView.h"
#include "velox/dwio/nimble/encodings/views/RLEEncodingView.h"
#include "velox/dwio/nimble/encodings/views/SimdForBitpackEncodingView.h"
#include "velox/dwio/nimble/encodings/views/TrivialEncodingView.h"

namespace facebook::nimble {

// X-macro list of (EncodingType, EncodingView class) pairs eligible for the
// devirtualized fast-path dispatch. All listed view classes are `final` on
// both the class and the `readTypedAt` override, so the compiler fully inlines
// the concrete body after the static_cast.
//
// Ordered by expected hotness for the current cluster-index workload; the
// switch's jump-table order does not affect correctness, but keeping hot
// types first helps readability.
//
// ALPEncodingView is intentionally omitted: it static_asserts T is floating-
// point, so instantiating ALPEncodingView<uint32_t> is a hard compile error
// rather than SFINAE-recoverable. A separate float-only dispatch macro can
// be added if a caller ever needs it.
//
// SparseBoolEncodingView is also omitted: it is fixed to bool (not templated
// on T). A bool-only dispatch macro can add it if needed.
#define NIMBLE_FAST_VIEW_TYPES(X)                 \
  X(Trivial, TrivialEncodingView)                 \
  X(FixedBitWidth, FixedBitWidthEncodingView)     \
  X(MainlyConstant, MainlyConstantEncodingView)   \
  X(BlockBitPacking, BlockBitPackingEncodingView) \
  X(RLE, RLEEncodingView)                         \
  X(Dictionary, DictionaryEncodingView)           \
  X(Constant, ConstantEncodingView)               \
  X(PFOR, PFOREncodingView)                       \
  X(SimdForBitpack, SimdForBitpackEncodingView)   \
  X(FOR, FOREncodingView)                         \
  X(Huffman, HuffmanEncodingView)                 \
  X(DeltaBlock, DeltaBlockEncodingView)

/// Reads a single value at `index` from `view` without a vtable dispatch when
/// `view`'s concrete encoding is one of the types in NIMBLE_FAST_VIEW_TYPES.
/// Other encoding types fall through to the base-class `readAt(index, void*)`
/// vtable path with unchanged cost.
///
/// Microbench (50 EncodingView instances per iteration, uint32 readAt, opt
/// build): ~33% faster than the base vtable path (~4.9ns -> ~3.3ns per call).
template <typename T>
inline T readEncodingViewAt(EncodingView* view, uint32_t index) {
  switch (view->encodingType()) {
#define NIMBLE_CASE_READAT(TYPE, VIEW) \
  case EncodingType::TYPE:             \
    return static_cast<VIEW<T>*>(view)->readAt(index);
    NIMBLE_FAST_VIEW_TYPES(NIMBLE_CASE_READAT)
#undef NIMBLE_CASE_READAT
    default: {
      T out;
      view->readAt(index, &out);
      return out;
    }
  }
}

} // namespace facebook::nimble
