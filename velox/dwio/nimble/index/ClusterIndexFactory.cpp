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
#include "velox/dwio/nimble/index/ClusterIndexFactory.h"

#include "velox/dwio/nimble/index/FlatClusterIndexFactory.h"
#include "velox/dwio/nimble/index/HierarchicalClusterIndexFactory.h"
#include "velox/dwio/nimble/index/IndexFactoryRegistry.h"

namespace facebook::nimble::index {

namespace {

IndexFactoryRegistry<ClusterIndexFactory>& factoryRegistry() {
  static auto* registry =
      new IndexFactoryRegistry<ClusterIndexFactory>{IndexFamily::Cluster};
  return *registry;
}

void ensureBuiltInFactoriesRegistered() {
  static const bool registered = [] {
    factoryRegistry().registerFactory(
        std::make_shared<const FlatClusterIndexFactory>());
    factoryRegistry().registerFactory(
        std::make_shared<const HierarchicalClusterIndexFactory>());
    return true;
  }();
  (void)registered;
}

} // namespace

void registerClusterIndexFactory(
    std::shared_ptr<const ClusterIndexFactory> factory) {
  ensureBuiltInFactoriesRegistered();
  factoryRegistry().registerFactory(std::move(factory));
}

const ClusterIndexFactory& clusterIndexFactory(std::string_view name) {
  ensureBuiltInFactoriesRegistered();
  return factoryRegistry().get(name);
}

} // namespace facebook::nimble::index
