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
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/dwio/nimble/tablet/DataInput.h"

namespace facebook::nimble {

// Identifies a stripe and the rows selected within it.
struct StripeSelection {
  // Absolute stripe index in the source tablet.
  uint32_t stripeIndex{0};
  // File-level first row of the source stripe.
  uint32_t startRow{0};
  // Number of rows in the full source stripe, including rows outside rowRange.
  uint32_t numRows{0};
  // Selected stripe-relative [startRow, endRow) rows.
  RowRange rowRange{};

  /// Returns the selected file-level row range, validating stripe bounds.
  RowRange selectedRowRange() const;

  /// Returns whether the file-level row is selected within this stripe.
  bool contains(uint32_t row) const;

  /// Compares the source stripe metadata and selected row range.
  bool operator==(const StripeSelection& other) const = default;
};

// Holds a stripe's projected encoded streams, selected rows, and backing
// storage.
struct ProjectedStripe {
  // Loading selects the full stripe, and the cursor clips it to the active
  // scan.
  StripeSelection selection{};
  // Stream views in projected-schema traversal order.
  std::vector<std::string_view> streams;
  // Keeps the storage referenced by streams alive. Stripes loaded together
  // share the same handle.
  DataInput::Handle dataHandle;
};

} // namespace facebook::nimble

/// Formats a stripe selection for validation errors.
template <>
struct fmt::formatter<facebook::nimble::StripeSelection> {
  constexpr auto parse(fmt::format_parse_context& context) {
    return context.begin();
  }

  template <typename FormatContext>
  auto format(
      const facebook::nimble::StripeSelection& selection,
      FormatContext& context) const {
    return fmt::format_to(
        context.out(),
        "StripeSelection{{stripeIndex={}, startRow={}, numRows={}, "
        "rowRange=[{}, {})}}",
        selection.stripeIndex,
        selection.startRow,
        selection.numRows,
        selection.rowRange.startRow,
        selection.rowRange.endRow);
  }
};
