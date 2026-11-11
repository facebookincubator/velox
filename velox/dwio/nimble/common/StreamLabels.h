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

#include "velox/dwio/nimble/common/SchemaReader.h"

namespace facebook::nimble {

class StreamLabels {
 public:
  explicit StreamLabels(const std::shared_ptr<const Type>& root);

  std::string_view streamLabel(offset_size offset) const;

 private:
  std::vector<std::string> labels_;
  std::vector<size_t> offsetToLabel_;
};

} // namespace facebook::nimble
