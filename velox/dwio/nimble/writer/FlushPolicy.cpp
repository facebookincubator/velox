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
#include "velox/dwio/nimble/writer/FlushPolicy.h"

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {
// Relieve memory pressure with chunking.
bool ChunkFlushPolicy::shouldChunk(const StripeProgress& stripeProgress) {
  const uint64_t inMemoryBytes =
      stripeProgress.stripeRawSize + stripeProgress.stripeEncodedSize;
  const auto writerMemoryThreshold = (lastChunkDecision_ == false)
      ? config_.writerMemoryHighThresholdBytes
      : config_.writerMemoryLowThresholdBytes;
  lastChunkDecision_ = inMemoryBytes > writerMemoryThreshold;
  return lastChunkDecision_;
}

// Optimize for expected storage stripe size.
bool ChunkFlushPolicy::shouldFlush(const StripeProgress& stripeProgress) {
  // When chunking is unable to relieve memory pressure, we flush stripe.
  if (stripeProgress.stripeRawSize + stripeProgress.stripeEncodedSize >
      config_.writerMemoryHighThresholdBytes) {
    return true;
  }

  double compressionFactor = config_.estimatedCompressionFactor;
  // Use historical compression ratio as a heuristic when available.
  if (stripeProgress.stripeEncodedSize > 0) {
    compressionFactor =
        static_cast<double>(stripeProgress.stripeEncodedLogicalSize) /
        stripeProgress.stripeEncodedSize;
  }
  double expectedEncodedStripeSize = stripeProgress.stripeEncodedSize +
      stripeProgress.stripeRawSize / std::max(compressionFactor, 1.0);
  return (expectedEncodedStripeSize >= config_.targetStripeSizeBytes);
}

TestFlushPolicy::TestFlushPolicy(State* state, Options options)
    : options_{std::move(options)}, state_{state} {
  NIMBLE_CHECK_NOT_NULL(state_, "TestFlushPolicy state must not be null.");
  NIMBLE_CHECK_GT(
      options_.maxStripePhysicalSize, 0, "maxStripePhysicalSize must be > 0.");
  NIMBLE_CHECK_GE(
      options_.estimatedCompressionFactor,
      1.0,
      "estimatedCompressionFactor must be >= 1.0.");
  NIMBLE_CHECK_GE(
      options_.flushChunkProbability,
      0.0,
      "flushChunkProbability must be in [0, 1].");
  NIMBLE_CHECK_LE(
      options_.flushChunkProbability,
      1.0,
      "flushChunkProbability must be in [0, 1].");
  NIMBLE_CHECK_GE(
      options_.flushStripeProbability,
      0.0,
      "flushStripeProbability must be in [0, 1].");
  NIMBLE_CHECK_LE(
      options_.flushStripeProbability,
      1.0,
      "flushStripeProbability must be in [0, 1].");
}

bool TestFlushPolicy::shouldChunk(const StripeProgress& /* stripeProgress */) {
  return std::uniform_real_distribution<double>(0.0, 1.0)(state_->rng) <
      options_.flushChunkProbability;
}

bool TestFlushPolicy::shouldFlush(const StripeProgress& stripeProgress) {
  double compressionFactor = options_.estimatedCompressionFactor;
  // Use the observed compression ratio once some data has been encoded.
  if (stripeProgress.stripeEncodedSize > 0) {
    compressionFactor =
        static_cast<double>(stripeProgress.stripeEncodedLogicalSize) /
        stripeProgress.stripeEncodedSize;
  }
  const double expectedEncodedStripeSize = stripeProgress.stripeEncodedSize +
      stripeProgress.stripeRawSize / std::max(compressionFactor, 1.0);
  // The rng draw is always evaluated first (left of the ||), so the rng
  // advances on every call regardless of the size guard -- the determinism
  // tests rely on that fixed consumption order.
  return std::uniform_real_distribution<double>(0.0, 1.0)(state_->rng) <
      options_.flushStripeProbability ||
      expectedEncodedStripeSize >= options_.maxStripePhysicalSize;
}

} // namespace facebook::nimble
