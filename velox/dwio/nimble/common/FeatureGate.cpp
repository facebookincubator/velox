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
#include "velox/dwio/nimble/common/FeatureGate.h"

#include <folly/Synchronized.h>

namespace facebook::nimble {

namespace {
// Process-wide gate, defaulting to the base no-op. Held by shared_ptr so that
// featureGate() can hand out an owning pointer that outlives a concurrent
// re-registration.
folly::Synchronized<std::shared_ptr<FeatureGate>>& gateStorage() {
  static auto* storage = new folly::Synchronized<std::shared_ptr<FeatureGate>>(
      std::make_shared<FeatureGate>());
  return *storage;
}
} // namespace

void registerFeatureGate(std::shared_ptr<FeatureGate> gate) {
  if (gate == nullptr) {
    gate = std::make_shared<FeatureGate>();
  }
  *gateStorage().wlock() = std::move(gate);
}

std::shared_ptr<const FeatureGate> featureGate() {
  return *gateStorage().rlock();
}

} // namespace facebook::nimble
