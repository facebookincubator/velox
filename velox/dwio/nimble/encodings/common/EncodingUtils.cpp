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
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble::detail {

SortedPositionSlots findSortedPositionSlots(
    std::string_view encodedPositions,
    uint32_t positionCount,
    uint32_t valueStartOffset,
    uint32_t valueEndOffset,
    velox::memory::MemoryPool& pool,
    const Encoding::Options& options) {
  const auto encodingType = EncodingPrefix::encodingType(encodedPositions);
  if (supportsEncodingView(encodingType)) {
    auto view = detail::createTypedEncodingView<uint32_t>(
        encodedPositions, &pool, options);
    NIMBLE_CHECK_NOT_NULL(
        view, "createTypedEncodingView returned null for a viewable encoding.");
    const auto lowerBound =
        [&](uint32_t begin, uint32_t end, uint32_t target) -> uint32_t {
      while (begin < end) {
        const auto middle = begin + ((end - begin) >> 1);
        // @lint-ignore NULLSAFECLANG nullable-dereference
        if (view->readAt(middle) < target) {
          begin = middle + 1;
        } else {
          end = middle;
        }
      }
      return begin;
    };
    const uint32_t slotStart = lowerBound(0, positionCount, valueStartOffset);
    const uint32_t slotEnd =
        lowerBound(slotStart, positionCount, valueEndOffset);
    return {.slotStart = slotStart, .slotEnd = slotEnd};
  }

  // Non-viewable position encoding (Varint, DeltaEncoding, ...).
  // Materialize every position once and binary search the decoded values.
  // The buffer is scoped to this function; the positional slice that follows
  // decodes from the encoded bytes again through EncodingFactory::slice.
  ScopedVector<uint32_t> positions{positionCount, &pool, options.bufferPool};
  EncodingFactory{options}
      .create(pool, encodedPositions, nullptr)
      ->materialize(positionCount, positions.data());
  auto* const values = positions.data();
  NIMBLE_CHECK_NOT_NULL(
      values, "ScopedVector must own storage after positionCount reserve.");
  const auto lowerBoundValue =
      [&](uint32_t begin, uint32_t end, uint32_t target) -> uint32_t {
    while (begin < end) {
      const auto middle = begin + ((end - begin) >> 1);
      // @lint-ignore NULLSAFECLANG nullable-dereference
      if (values[middle] < target) {
        begin = middle + 1;
      } else {
        end = middle;
      }
    }
    return begin;
  };
  const uint32_t slotStart =
      lowerBoundValue(0, positionCount, valueStartOffset);
  const uint32_t slotEnd =
      lowerBoundValue(slotStart, positionCount, valueEndOffset);
  return {.slotStart = slotStart, .slotEnd = slotEnd};
}

} // namespace facebook::nimble::detail
