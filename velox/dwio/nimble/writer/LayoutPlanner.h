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

#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"

namespace facebook::nimble {

/// Orders streams according to the logical schema and configured FlatMap keys.
class DefaultLayoutPlanner : public LayoutPlanner {
 public:
  /// Creates a planner over a schema that may gain FlatMap streams.
  DefaultLayoutPlanner(
      const SchemaBuilder* schemaBuilder,
      const std::optional<
          std::vector<std::tuple<size_t, std::vector<int64_t>>>>&
          flatMapFeatureOrder);

  /// Returns streams in their physical file order.
  virtual std::vector<Stream> getLayout(std::vector<Stream>&& streams) override;

 private:
  // Provides the evolving logical schema and stripe dictionary bindings.
  const SchemaBuilder* const schemaBuilder_;
  // Preferred key order for configured top-level FlatMap columns.
  std::vector<std::tuple<size_t, std::vector<int64_t>>> flatMapFeatureOrder_;
};
} // namespace facebook::nimble
