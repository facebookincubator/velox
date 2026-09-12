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
#include <string_view>

#include "velox/dwio/nimble/common/FeatureGate.h"

namespace facebook::nimble::test {

/// Force-enables one runtime-gated writer feature for the lifetime of this
/// object, restoring the default no-op gate on destruction. Features that
/// default to off are otherwise unreachable in tests, which register no gate.
/// A writer resolves its gates once at construction, so this must be created
/// before the writer under test.
class ScopedFeatureGate {
 public:
  explicit ScopedFeatureGate(std::string_view feature) {
    registerFeatureGate(std::make_shared<EnablingGate>(feature));
  }

  ScopedFeatureGate(const ScopedFeatureGate&) = delete;
  ScopedFeatureGate& operator=(const ScopedFeatureGate&) = delete;

  ~ScopedFeatureGate() {
    registerFeatureGate(nullptr);
  }

 private:
  // Forces one feature on and leaves every other feature at the value its
  // caller requested, so enabling one does not perturb the rest.
  class EnablingGate : public FeatureGate {
   public:
    explicit EnablingGate(std::string_view feature) : feature_{feature} {}

    bool enabled(std::string_view feature, bool defaultValue) const override {
      return feature == feature_ || defaultValue;
    }

   private:
    const std::string_view feature_;
  };
};

} // namespace facebook::nimble::test
