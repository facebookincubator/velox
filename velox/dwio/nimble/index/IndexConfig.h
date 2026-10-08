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

#include <string>
#include <type_traits>
#include <utility>

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/IndexTypes.h"

namespace facebook::nimble::index {

/// Common configuration for one named index implementation.
class IndexConfig {
 public:
  virtual ~IndexConfig() = default;

  /// Selects the index factory registry.
  IndexFamily family;
  /// Names the registered index factory.
  std::string name;

 protected:
  IndexConfig(IndexFamily family, std::string name)
      : family{family}, name{std::move(name)} {}
  IndexConfig(const IndexConfig&) = default;
  IndexConfig(IndexConfig&&) = default;
};

template <typename T>
const T& checkedIndexConfig(const IndexConfig& config) {
  static_assert(std::is_base_of_v<IndexConfig, T>);
  return *velox::checkedPointerCast<const T>(&config);
}

} // namespace facebook::nimble::index
