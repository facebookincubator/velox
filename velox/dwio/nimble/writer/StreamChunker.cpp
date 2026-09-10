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

#include "velox/dwio/nimble/writer/StreamChunker.h"

namespace facebook::nimble {
template <typename T>
std::unique_ptr<StreamChunker> getStreamChunkerTyped(
    StreamData& streamData,
    const StreamChunkerOptions& options) {
  if (auto* contentStreamChunker =
          dynamic_cast<ContentStreamData<T>*>(&streamData)) {
    return std::make_unique<ContentStreamChunker<T>>(
        *contentStreamChunker, options);
  } else if (
      auto* nullableContentStreamData =
          dynamic_cast<NullableContentStreamData<T>*>(&streamData)) {
    // When there are no nulls in the NullableContentStreamData stream, we treat
    // it as a regular ContentStreamData stream.
    if (!streamData.hasNulls()) {
      return std::make_unique<
          ContentStreamChunker<T, NullableContentStreamData<T>>>(
          *nullableContentStreamData, options);
    }
    return std::make_unique<NullableContentStreamChunker<T>>(
        *nullableContentStreamData, options);
  } else if (
      auto* nullableContentStringStreamData =
          dynamic_cast<NullableContentStringStreamData*>(&streamData)) {
    return std::make_unique<NullableContentStringStreamChunker>(
        *nullableContentStringStreamData, options);
  } else if (
      auto* nullsStreamData = dynamic_cast<NullsStreamData*>(&streamData)) {
    return std::make_unique<NullsStreamChunker>(*nullsStreamData, options);
  }
  NIMBLE_UNREACHABLE("Unsupported streamData type")
}

std::unique_ptr<StreamChunker> getStreamChunker(
    StreamData& streamData,
    const StreamChunkerOptions& options) {
  const auto scalarKind = streamData.descriptor().scalarKind();
  switch (scalarKind) {
#define HANDLE_SCALAR_KIND(kind, type) \
  case ScalarKind::kind:               \
    return getStreamChunkerTyped<type>(streamData, options);
    HANDLE_SCALAR_KIND(Bool, bool);
    HANDLE_SCALAR_KIND(Int8, int8_t);
    HANDLE_SCALAR_KIND(Int16, int16_t);
    HANDLE_SCALAR_KIND(UInt16, uint16_t);
    HANDLE_SCALAR_KIND(Int32, int32_t);
    HANDLE_SCALAR_KIND(UInt32, uint32_t);
    HANDLE_SCALAR_KIND(Int64, int64_t);
    HANDLE_SCALAR_KIND(Float, float);
    HANDLE_SCALAR_KIND(Double, double);
    HANDLE_SCALAR_KIND(String, std::string_view);
    HANDLE_SCALAR_KIND(Binary, std::string_view);
    default:
      NIMBLE_UNREACHABLE("Unsupported scalar kind {}", toString(scalarKind));
  }
}
} // namespace facebook::nimble
