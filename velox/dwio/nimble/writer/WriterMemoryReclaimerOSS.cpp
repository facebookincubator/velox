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
#include "velox/dwio/nimble/writer/Writer.h"

namespace facebook::nimble {

// The internal counterpart returns a reclaimer derived from
// velox::exec::MemoryReclaimer, which additionally suspends the Velox Driver
// for the duration of an arbitration request. velox/exec is not part of
// VELOX_BUILD_MINIMAL_WITH_DWIO, which is what Nimble's open source build
// configures, so no reclaimer is installed here and the writer does not
// participate in memory arbitration.
std::unique_ptr<velox::memory::MemoryReclaimer> Writer::makeMemoryReclaimer() {
  return nullptr;
}

} // namespace facebook::nimble
