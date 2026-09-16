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

#include <memory>

#include <zstd.h>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble::serde::detail {

/// Returns the calling thread's ZSTD decompression context, creating it on
/// first use. Shared by StreamData (stream-level decompression) and
/// StreamDataParser (chunk-level decompression) so a thread that does both
/// still allocates exactly one context. Being `inline` is what guarantees a
/// single `thread_local` across translation units — do not copy this into
/// individual .cpp files.
inline ZSTD_DCtx* getThreadLocalDCtx() {
  struct DCtxDeleter {
    void operator()(ZSTD_DCtx* ctx) const {
      ZSTD_freeDCtx(ctx);
    }
  };
  static thread_local std::unique_ptr<ZSTD_DCtx, DCtxDeleter> ctx{
      ZSTD_createDCtx()};
  NIMBLE_CHECK(ctx != nullptr, "Failed to create ZSTD decompression context");
  return ctx.get();
}

} // namespace facebook::nimble::serde::detail
