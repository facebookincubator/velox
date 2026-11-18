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

#include <cstdint>
#include <vector>

#include "velox/dwio/nimble/tablet/StripeGroup.h"

namespace facebook::nimble {

/// Which stripe of the file a stripe is, where it lies in the file, its rows,
/// and where some of its streams lie in it, as TabletReader reports them for
/// the stripes of a stripe group.
struct StripeMetadata {
  /// Index of the stripe in the file.
  uint32_t stripe{0};
  /// File offset of the stripe, which stream offsets are relative to.
  uint64_t offset{0};
  /// Rows of the stripe.
  uint32_t numRows{0};
  /// Location of each requested stream in the stripe, in request order.
  std::vector<StripeGroup::StreamMetadata> streams;
};

} // namespace facebook::nimble
