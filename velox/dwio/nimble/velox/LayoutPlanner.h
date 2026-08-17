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

#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"

namespace facebook::nimble {

class DefaultLayoutPlanner : public LayoutPlanner {
 public:
  DefaultLayoutPlanner(
      std::function<std::shared_ptr<const TypeBuilder>()> typeResolver,
      const std::optional<
          std::vector<std::tuple<size_t, std::vector<int64_t>>>>&
          flatMapFeatureOrder);

  virtual std::vector<Stream> getLayout(std::vector<Stream>&& streams) override;

 private:
  std::function<std::shared_ptr<const TypeBuilder>()> typeResolver_;
  std::vector<std::tuple<size_t, std::vector<int64_t>>> flatMapFeatureOrder_;
};
} // namespace facebook::nimble
