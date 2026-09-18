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

#include <cstdint>
#include <functional>
#include <optional>
#include <random>

namespace facebook::nimble {

// TODO: Set default values for these parameters based on DISCO experiments.
// Use abitrary values for now.
struct ChunkFlushPolicyConfig {
  // Threshold to trigger chunking to relieve memory pressure
  const uint64_t writerMemoryHighThresholdBytes{200 * 1024L * 1024L};
  // Threshold below which chunking stops and stripe size optimization resumes
  const uint64_t writerMemoryLowThresholdBytes{100 * 1024L * 1024L};
  // Target size for encoded stripes
  const uint64_t targetStripeSizeBytes{100 * 1024L * 1024L};
  // Expected ratio of raw to encoded data
  const double estimatedCompressionFactor{3.7};
};

struct StripeProgress {
  // Size of the stripe data when it's fully decompressed and decoded
  const uint64_t stripeRawSize;
  // Size of the stripe after buffered data is encoded and optionally compressed
  const uint64_t stripeEncodedSize;
  // Logical size of the now encoded stripe data
  const uint64_t stripeEncodedLogicalSize;
};

class FlushPolicy {
 public:
  virtual ~FlushPolicy() = default;

  /// Post-write flush query consulted at the end of every Writer::write()
  /// call (via evaluateFlushPolicy). Returning true causes the writer
  /// to immediately cut the currently open stripe. This is the size-driven
  /// cut mechanism for policies that decide based on how much data has
  /// accumulated in the current stripe (bytes, encoded size, etc.). Callers
  /// that need content- or count-driven cuts should install a BufferPolicy
  /// instead — see BufferPolicy.h.
  ///
  /// Contract: the answer must be a pure function of stripeProgress at call
  /// time. Policies that need cross-call state (e.g. hysteresis) must own it
  /// in a factory-captured struct — Writer recreates the policy at every
  /// consultation.
  virtual bool shouldFlush(const StripeProgress& stripeProgress) = 0;

  /// Post-write chunk query consulted alongside shouldFlush (via
  /// evaluateFlushPolicy) — but only when
  /// WriterOptions::enableChunking is true. Returning true causes the
  /// writer to soft-chunk stream data currently buffered above the
  /// per-stream chunk threshold, relieving memory pressure without cutting
  /// the stripe. Same purity + state-ownership contract as shouldFlush.
  virtual bool shouldChunk(const StripeProgress& stripeProgress) = 0;
};

class StripeRawSizeFlushPolicy final : public FlushPolicy {
 public:
  explicit StripeRawSizeFlushPolicy(uint64_t stripeRawSize)
      : stripeRawSize_{stripeRawSize} {}

  bool shouldFlush(const StripeProgress& stripeProgress) override {
    return stripeProgress.stripeRawSize >= stripeRawSize_;
  }

  bool shouldChunk(const StripeProgress&) override {
    return false;
  }

 private:
  const uint64_t stripeRawSize_;
};

class LambdaFlushPolicy : public FlushPolicy {
 public:
  explicit LambdaFlushPolicy(
      std::function<bool(const StripeProgress&)> flushLambda =
          [](const StripeProgress&) { return false; },
      std::function<bool(const StripeProgress&)> chunkLambda =
          [](const StripeProgress&) { return false; })
      : flushLambda_{std::move(flushLambda)},
        chunkLambda_{std::move(chunkLambda)} {}

  bool shouldFlush(const StripeProgress& stripeProgress) override {
    return flushLambda_(stripeProgress);
  }

  bool shouldChunk(const StripeProgress& stripeProgress) override {
    return chunkLambda_(stripeProgress);
  }

 private:
  std::function<bool(const StripeProgress&)> flushLambda_;
  std::function<bool(const StripeProgress&)> chunkLambda_;
};

class ChunkFlushPolicy : public FlushPolicy {
 public:
  explicit ChunkFlushPolicy(ChunkFlushPolicyConfig config)
      : config_{std::move(config)}, lastChunkDecision_{false} {}

  // Optimize for expected storage stripe size.
  bool shouldFlush(const StripeProgress& stripeProgress) override;

  // Relieve memory pressure with chunking.
  bool shouldChunk(const StripeProgress& stripeProgress) override;

  const ChunkFlushPolicyConfig& getConfig() const {
    return config_;
  }

 private:
  const ChunkFlushPolicyConfig config_;
  bool lastChunkDecision_;
};

/// Test-only policy driving chunk and stripe boundaries for the fuzzer, seeded
/// so a failing run can be reproduced from the logged seed. It chunks and
/// flushes probabilistically, with an encoded-size flush guard. Per-type tuning
/// (the nimble.flush_policy_config "type") is assembled by
/// FlushPolicyFactoryFactory, not here. Instances are created only by
/// TestFlushPolicyFactory, which owns the shared rng State (see State below) so
/// decisions vary across the per-write() policy instances the writer creates.
class TestFlushPolicy final : public FlushPolicy {
 public:
  /// Default chunk/stripe flush probabilities. Prod-representative values
  /// ported from the (removed) Vader validation-service policy.
  static constexpr double kDefaultFlushChunkProbability = 1.0 / 3;
  static constexpr double kDefaultFlushStripeProbability = 1.0 / 20;

  /// Tuning for TestFlushPolicy; unset fields keep the defaults below.
  struct Options {
    /// Seed for the policy's rng. Log it to reproduce a failing run.
    uint64_t seed{0};
    /// Probability in [0, 1] of cutting a chunk on each write.
    double flushChunkProbability{kDefaultFlushChunkProbability};
    /// Probability in [0, 1] of flushing the stripe on each write, independent
    /// of the size guard below.
    double flushStripeProbability{kDefaultFlushStripeProbability};
    /// Encoded-size flush guard: flush once the predicted encoded stripe size
    /// reaches this many bytes.
    uint64_t maxStripePhysicalSize{128 * 1024L * 1024L};
    /// Raw->encoded ratio used to predict the encoded stripe size before any
    /// data has been encoded (once a stripe has encoded bytes the observed
    /// ratio is used instead). 3.7 is the prod-representative value the Vader
    /// validation service used.
    double estimatedCompressionFactor{3.7};
  };

  /// Flushes the stripe with probability options.flushStripeProbability, or
  /// when the predicted encoded stripe size would exceed
  /// options.maxStripePhysicalSize.
  bool shouldFlush(const StripeProgress& stripeProgress) override;

  /// Cuts a chunk with probability options.flushChunkProbability, independent
  /// of accumulated size.
  bool shouldChunk(const StripeProgress& stripeProgress) override;

 private:
  // TestFlushPolicyFactory owns the shared State and constructs policies that
  // borrow it, so it needs access to the private State and constructor.
  friend class TestFlushPolicyFactory;

  /// Shared rng state, owned by TestFlushPolicyFactory and outliving the
  /// per-write() policy instances the writer recreates. Writer rebuilds
  /// the flush policy from its factory on every write() (see
  /// Writer::evaluateFlushPolicy), so an rng owned by the policy
  /// itself would reseed identically each call and make the same decision every
  /// write(). Holding it here instead lets decisions vary across writes while
  /// staying reproducible from the seed.
  struct State {
    explicit State(uint64_t seed) : rng{seed} {}

    // TODO: rng is not thread-safe; TestFlushPolicy must be driven from a
    // single writer thread.
    std::mt19937_64 rng;
  };

  /// 'state' is the shared rng container (see State), owned by the factory and
  /// outliving this policy; it must be non-null. Every tuning value is
  /// sanity-checked here; an already range-validated Options from
  /// FlushPolicyFactoryFactory always passes.
  TestFlushPolicy(State* state, Options options);

  const Options options_;
  // Borrowed, non-owning: the factory (owner) outlives this policy, which the
  // writer recreates on every write().
  State* const state_;
};

} // namespace facebook::nimble
