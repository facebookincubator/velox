/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/expression/VectorFunctionListener.h"

#include <folly/Synchronized.h>

namespace facebook::velox::exec {

namespace {

folly::Synchronized<
    std::vector<std::shared_ptr<VectorFunctionListenerFactory>>>&
listenerFactories() {
  static folly::Synchronized<
      std::vector<std::shared_ptr<VectorFunctionListenerFactory>>>
      kFactories;
  return kFactories;
}

} // namespace

bool registerVectorFunctionListenerFactory(
    std::shared_ptr<VectorFunctionListenerFactory> factory) {
  return listenerFactories().withWLock([&](auto& factories) {
    for (const auto& existing : factories) {
      if (existing == factory) {
        return false;
      }
    }
    factories.emplace_back(std::move(factory));
    return true;
  });
}

bool unregisterVectorFunctionListenerFactory(
    const std::shared_ptr<VectorFunctionListenerFactory>& factory) {
  return listenerFactories().withWLock([&](auto& factories) {
    for (auto it = factories.begin(); it != factories.end(); ++it) {
      if (*it == factory) {
        factories.erase(it);
        return true;
      }
    }
    return false;
  });
}

std::vector<VectorFunctionListeners> createVectorFunctionListeners(
    std::string_view name,
    const VectorFunctionMetadata& metadata,
    const core::QueryConfig& queryConfig) {
  std::vector<VectorFunctionListeners> listeners;
  listenerFactories().withRLock([&](const auto& factories) {
    for (const auto& factory : factories) {
      if (auto result = factory->create(name, metadata, queryConfig)) {
        listeners.push_back(std::move(*result));
      }
    }
  });
  return listeners;
}

} // namespace facebook::velox::exec
