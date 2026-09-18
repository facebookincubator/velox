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
#include "velox/dwio/nimble/writer/FlushPolicyFactory.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <folly/Conv.h>
#include <folly/String.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/config/Config.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {
namespace {

// Trims leading/trailing whitespace, returning a view into the input.
std::string_view trimToken(std::string_view token) {
  const auto trimmed = folly::trimWhitespace(folly::StringPiece(token));
  return std::string_view(trimmed.data(), trimmed.size());
}

// Parses a numeric value, surfacing a NimbleUserError (not a
// folly::ConversionError) on bad input.
template <typename T>
T parseValue(std::string_view key, std::string_view value) {
  const auto parsed = folly::tryTo<T>(folly::StringPiece(value));
  NIMBLE_USER_CHECK(
      parsed.hasValue(),
      "Invalid value '{}' for flush policy config key '{}'.",
      value,
      key);
  return parsed.value();
}

using ConfigEntries =
    std::vector<std::pair<std::string_view, std::string_view>>;

// Parses the "test_random" tuning from the config's key:value pairs (the "type"
// key already removed) into TestFlushPolicy::Options. Ranges are validated as
// user errors here so a bad config throws NimbleUserError rather than the
// TestFlushPolicy constructor's NimbleInternalError. This is the shared parser
// the other types build on.
TestFlushPolicy::Options parseTestRandomConfig(const ConfigEntries& entries) {
  TestFlushPolicy::Options options;
  for (const auto& [key, value] : entries) {
    if (key == "seed") {
      options.seed = parseValue<uint64_t>(key, value);
    } else if (key == "flush_chunk_probability") {
      options.flushChunkProbability = parseValue<double>(key, value);
    } else if (key == "flush_stripe_probability") {
      options.flushStripeProbability = parseValue<double>(key, value);
    } else if (key == "max_stripe_physical_size") {
      // velox::config::toCapacity parses a binary byte size with an embedded,
      // case-insensitive KB/MB/GB suffix (e.g. "128MB").
      try {
        options.maxStripePhysicalSize = velox::config::toCapacity(
            std::string(value), velox::config::CapacityUnit::BYTE);
      } catch (const velox::VeloxUserError&) {
        NIMBLE_USER_FAIL(
            "Invalid byte size '{}' for flush policy config key '{}'.",
            value,
            key);
      }
    } else if (key == "estimated_compression_factor") {
      options.estimatedCompressionFactor = parseValue<double>(key, value);
    } else {
      NIMBLE_USER_FAIL("Unknown nimble.flush_policy_config key '{}'.", key);
    }
  }
  NIMBLE_USER_CHECK_GT(
      options.maxStripePhysicalSize,
      0,
      "flush policy max_stripe_physical_size must be > 0.");
  NIMBLE_USER_CHECK_GE(
      options.estimatedCompressionFactor,
      1.0,
      "flush policy estimated_compression_factor must be >= 1.0.");
  NIMBLE_USER_CHECK_GE(
      options.flushChunkProbability,
      0.0,
      "flush policy flush_chunk_probability must be in [0, 1].");
  NIMBLE_USER_CHECK_LE(
      options.flushChunkProbability,
      1.0,
      "flush policy flush_chunk_probability must be in [0, 1].");
  NIMBLE_USER_CHECK_GE(
      options.flushStripeProbability,
      0.0,
      "flush policy flush_stripe_probability must be in [0, 1].");
  NIMBLE_USER_CHECK_LE(
      options.flushStripeProbability,
      1.0,
      "flush policy flush_stripe_probability must be in [0, 1].");
  return options;
}

// Parses the "test_per_batch" tuning: the test_random tuning with
// flush_chunk_probability pinned to 1.0 (cuts a chunk every batch; any value in
// the spec is ignored).
TestFlushPolicy::Options parseTestPerBatchConfig(const ConfigEntries& entries) {
  auto options = parseTestRandomConfig(entries);
  options.flushChunkProbability = 1.0;
  return options;
}

} // namespace

// Out of line because it reaches TestFlushPolicy's private State and ctor via
// friendship: owns the shared rng State and hands each produced policy a raw,
// non-owning pointer to it, so the policies the writer recreates per write()
// keep advancing one rng.
TestFlushPolicyFactory::TestFlushPolicyFactory(TestFlushPolicy::Options options)
    : state_{std::make_shared<TestFlushPolicy::State>(options.seed)},
      options_{std::move(options)} {}

std::unique_ptr<FlushPolicy> TestFlushPolicyFactory::operator()() const {
  // Constructed via new (not make_unique): TestFlushPolicy's constructor is
  // private and make_unique is not the friend -- this factory is.
  return std::unique_ptr<TestFlushPolicy>(
      new TestFlushPolicy(state_.get(), options_));
}

std::optional<FlushPolicyFactory> FlushPolicyFactoryFactory::create(
    std::string_view configStr) {
  if (configStr.empty()) {
    return std::nullopt;
  }
  // Tokenize into key:value pairs and peel off "type", which selects the
  // parser.
  std::string_view type;
  ConfigEntries entries;
  std::vector<std::string_view> rawEntries;
  folly::split(',', configStr, rawEntries);
  for (const auto rawEntry : rawEntries) {
    const auto colonPos = rawEntry.find(':');
    NIMBLE_USER_CHECK(
        colonPos != std::string_view::npos,
        "Malformed nimble.flush_policy_config entry '{}'; want key:value.",
        rawEntry);
    const auto key = trimToken(rawEntry.substr(0, colonPos));
    const auto value = trimToken(rawEntry.substr(colonPos + 1));
    if (key == "type") {
      type = value;
    } else {
      entries.emplace_back(key, value);
    }
  }
  // An absent type keeps the production size-based policy.
  if (type.empty()) {
    return std::nullopt;
  }
  if (type == "test_random") {
    return FlushPolicyFactory{
        TestFlushPolicyFactory{parseTestRandomConfig(entries)}};
  }
  if (type == "test_per_batch") {
    return FlushPolicyFactory{
        TestFlushPolicyFactory{parseTestPerBatchConfig(entries)}};
  }
  NIMBLE_USER_FAIL(
      "Invalid nimble.flush_policy_config type '{}'. Valid: 'test_per_batch', 'test_random'.",
      type);
}

} // namespace facebook::nimble
