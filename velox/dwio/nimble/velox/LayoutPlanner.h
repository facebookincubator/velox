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

// Defines the stripe layout planner and its configurable priority orders.
// Physical stream order affects read locality; stream ids preserve semantics.

#pragma once

#include <string>

#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"

namespace facebook::nimble {

/// Configures which columns and FlatMap features lead each stripe.
struct LayoutPlannerOptions {
  LayoutPlannerOptions() = default;

  /// Lists FlatMap column ordinals and their leading feature ids.
  std::vector<std::tuple<size_t, std::vector<int64_t>>> flatMapFeatureOrder;
  /// Lists leading top-level columns by name.
  std::vector<std::string> columnOrder;
};

/// Orders streams according to the logical schema, a configured top-level
/// column order, and configured FlatMap keys.
class DefaultLayoutPlanner : public LayoutPlanner {
 public:
  /// Creates a planner for an evolving writer schema.
  DefaultLayoutPlanner(
      const SchemaBuilder* schemaBuilder,
      LayoutPlannerOptions options);

  /// Preserves the feature-order-only API used by existing callers.
  DefaultLayoutPlanner(
      const SchemaBuilder* schemaBuilder,
      const std::optional<
          std::vector<std::tuple<size_t, std::vector<int64_t>>>>&
          flatMapFeatureOrder);

  /// Returns the streams in physical stripe order.
  virtual std::vector<Stream> getLayout(std::vector<Stream>&& streams) override;

 private:
  // Resolves names again when the evolving root gains a child.
  const std::vector<size_t>& resolvedColumnOrder(const RowTypeBuilder& root);

  // Provides the evolving logical schema and stripe dictionary bindings.
  const SchemaBuilder& schemaBuilder_;
  // Feature and column orders the layout follows.
  const LayoutPlannerOptions options_;
  // `options_.columnOrder` as top-level ordinals, valid while the root is
  // `resolvedRoot_` with `resolvedChildrenCount_` children.
  std::vector<size_t> orderedColumns_;
  // Per top-level ordinal, whether `orderedColumns_` places the column.
  std::vector<bool> isOrderedColumn_;
  // Root row `orderedColumns_` was resolved against.
  const RowTypeBuilder* resolvedRoot_{nullptr};
  // Child count of `resolvedRoot_` at resolution; rows may gain children.
  size_t resolvedChildrenCount_{0};
};
} // namespace facebook::nimble
