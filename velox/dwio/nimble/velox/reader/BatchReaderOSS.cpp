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
#include "velox/dwio/nimble/velox/reader/NimbleReaderFactory.h"

namespace facebook::velox::nimble::detail {

// The batch reader reaches into internal-only code, so it is not part of the
// open-source build. Returning nullptr makes the factory report a clear error
// rather than fail to link.
std::unique_ptr<dwio::common::Reader> createBatchReader(
    const dwio::common::ReaderOptions& /*options*/,
    const std::shared_ptr<ReadFile>& /*readFile*/) {
  return nullptr;
}

} // namespace facebook::velox::nimble::detail
