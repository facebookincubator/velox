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
#include <string>
#include <string_view>

#include "folly/Synchronized.h"
#include "folly/container/F14Map.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/IndexConfig.h"

namespace facebook::nimble::index {

template <typename Factory>
class IndexFactoryRegistry {
 public:
  explicit IndexFactoryRegistry(IndexFamily family) : family_{family} {}

  void registerFactory(std::shared_ptr<const Factory> factory) {
    NIMBLE_CHECK_NOT_NULL(factory);
    const auto name = std::string{factory->name()};
    NIMBLE_CHECK(
        !name.empty(),
        "Index factory name cannot be empty for family '{}'",
        familyName(family_));
    auto factories = factories_.wlock();
    auto [_, inserted] = factories->emplace(name, std::move(factory));
    NIMBLE_CHECK(
        inserted,
        "Index factory '{}' is already registered for family '{}'",
        name,
        familyName(family_));
  }

  const Factory& get(std::string_view name) const {
    const auto* factory = tryGet(name);
    NIMBLE_CHECK_NOT_NULL(
        factory,
        "Unknown index factory '{}' for family '{}'",
        name,
        familyName(family_));
    return *factory;
  }

  const Factory* tryGet(std::string_view name) const {
    auto factories = factories_.rlock();
    auto it = factories->find(name);
    return it == factories->end() ? nullptr : it->second.get();
  }

 private:
  static std::string_view familyName(IndexFamily family) {
    switch (family) {
      case IndexFamily::Cluster:
        return "cluster";
      case IndexFamily::Dense:
        return "dense";
    }
    NIMBLE_UNREACHABLE("Unsupported index family");
  }

  const IndexFamily family_;
  folly::Synchronized<
      folly::F14FastMap<std::string, std::shared_ptr<const Factory>>>
      factories_;
};

} // namespace facebook::nimble::index
