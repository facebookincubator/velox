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

#include <functional>
#include <memory>
#include <optional>
#include <string_view>

#include "velox/dwio/nimble/writer/FlushPolicy.h"

namespace facebook::nimble {

using FlushPolicyFactory = std::function<std::unique_ptr<FlushPolicy>()>;

/// Factory functor for the test flush policy: owns the shared rng state
/// (TestFlushPolicy::State) and hands each produced policy a raw, non-owning
/// pointer to it, so the policies the writer recreates per write() keep
/// advancing one rng. The state is held via shared_ptr because a std::function
/// target must be copyable; policies only borrow it, so the refcount stays 1. A
/// friend of TestFlushPolicy, giving it access to the private State and ctor.
class TestFlushPolicyFactory {
 public:
  explicit TestFlushPolicyFactory(TestFlushPolicy::Options options);

  std::unique_ptr<FlushPolicy> operator()() const;

 private:
  const std::shared_ptr<TestFlushPolicy::State> state_;
  const TestFlushPolicy::Options options_;
};

/// Builds a flush-policy factory from a nimble.flush_policy_config string of
/// the form "type:<t>,key:value,...". The leading "type" selects which policy
/// to build and each type parses only the keys it accepts, so the handling is
/// fully independent per type. Returns std::nullopt for an absent type,
/// signalling the caller to keep the size-based policy.
///
/// A malformed entry, unknown key, unparseable/out-of-range value, or unknown
/// type is reported as a NimbleUserError.
class FlushPolicyFactoryFactory {
 public:
  static std::optional<FlushPolicyFactory> create(std::string_view configStr);
};

} // namespace facebook::nimble
