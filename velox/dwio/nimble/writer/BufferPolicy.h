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

#include <deque>
#include <memory>
#include <vector>

#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble {

/// A stripe-ready collection of input vectors and per-input row ranges.
/// The writer, on receiving a non-empty BufferRange from
/// BufferPolicy::flushInput(), ingests each
/// `inputs[i]->slice(rowRanges[i])` into its per-column stream buffers
/// and then flushes exactly one stripe.
///
/// Parallel arrays: `inputs.size() == rowRanges.size()`, and
/// `rowRanges[i]` applies to `inputs[i]`. Empty means "no stripe ready
/// yet".
struct BufferRange {
  std::vector<velox::VectorPtr> inputs;
  std::vector<RowRange> rowRanges;

  bool empty() const {
    return inputs.empty();
  }
};

/// Content-driven cutting hook for Writer. Replaces the earlier
/// per-batch pre-write cut model with a buffering model: the policy
/// accumulates incoming row vectors across successive `write()` calls and
/// emits stripe-ready `BufferRange`s when its internal invariant (group
/// boundary, row-count cap, etc.) says a stripe is complete.
///
/// Lifecycle, from the writer's perspective:
///   1. `bufferInput(input)` — called once per incoming batch in
///      `Writer::write()`. Policy retains a reference to `input` (via
///      VectorPtr's shared ownership) until the corresponding rows are
///      emitted by `flushInput()` or dropped.
///   2. `flushInput()` — called repeatedly by the writer immediately after
///      every `bufferInput()`, draining the policy of any completed
///      BufferRanges. Returns empty when the policy has nothing ready.
///   3. `finalize()` — called once by `Writer::close()` to signal
///      end-of-input. On subsequent `flushInput()` calls the policy must
///      emit any residual open range (e.g. the last user's rows) rather
///      than continuing to accumulate.
///
/// The writer only invokes BufferPolicy when
/// `WriterOptions::bufferPolicyFactory` is set. When unset, the writer
/// falls back to the legacy FlushPolicy path (`shouldFlush` post-write).
/// FlushPolicy and BufferPolicy address different concerns: FlushPolicy
/// cuts on size accumulated in the writer's stream buffers; BufferPolicy
/// cuts on content the policy sees in the incoming vectors. Callers pick
/// one or the other, not both.
///
/// Memory: BufferPolicy retains vector references for as long as it holds
/// unemitted rows from them. The policy is responsible for its own
/// bookkeeping and for bounding memory use (e.g. by emitting periodically
/// or capping accumulated rows). Writer imposes no upper bound.
class BufferPolicy {
 public:
  virtual ~BufferPolicy() = default;

  /// Ingest an incoming batch into the policy's internal buffer. The base
  /// interface accepts any VectorPtr — policies that need a specific shape
  /// (e.g. a plain RowVector for column-content inspection) should
  /// dynamic_cast internally and reject wrapped/mismatched inputs.
  /// Implementations should also reject empty (0-row) inputs if their
  /// bookkeeping (e.g. indexing at `numRows - 1`) is undefined on them.
  /// Writer never passes an empty batch in normal usage, so this only
  /// matters when the policy is driven directly (e.g. tests).
  virtual void bufferInput(velox::VectorPtr input) = 0;

  /// Return the next ready-to-write BufferRange, or an empty BufferRange
  /// if the policy is still accumulating. Writer drains this method
  /// after every `bufferInput()` (and after `finalize()` at close time)
  /// until it returns empty.
  virtual BufferRange flushInput() = 0;

  /// Signal end-of-input. After this call, the next `flushInput()` calls
  /// must emit any residual open buffer content rather than continuing to
  /// wait for a boundary.
  virtual void finalize() = 0;
};

/// Cuts a stripe once its rows reach a raw-size target, slicing an input batch
/// when the target falls inside it. A FlushPolicy ends stripes only between
/// write() calls, so a batch larger than its target becomes one stripe: 1,024
/// rows of 3,072 floats make a 12 MB stripe under a 1 MB
/// StripeRawSizeFlushPolicy. This policy cuts the same batch into 1 MB stripes.
///
/// Raw sizes come from getRawSizeFromVector(), averaged over the rows of each
/// input. Stripes of fixed-width rows end within one row of the target, and
/// stripes of rows of varying size end near it. Batches smaller than the target
/// accumulate across write() calls.
class StripeRawSizeBufferPolicy final : public BufferPolicy {
 public:
  explicit StripeRawSizeBufferPolicy(uint64_t stripeRawSize);

  void bufferInput(velox::VectorPtr input) override;

  BufferRange flushInput() override;

  void finalize() override;

 private:
  // Queues the buffered rows as a stripe for flushInput().
  void flushRange();

  // Raw bytes at which a stripe is cut.
  const uint64_t stripeRawSize_;
  // Stripes ready for flushInput(), oldest first.
  std::deque<BufferRange> flushQueue_;
  // Rows of the stripe being filled.
  BufferRange bufferedRange_;
  // Estimated raw bytes of bufferedRange_.
  double bufferedRawSize_{0};
  // Set by finalize(); flushInput() then also returns a partly filled stripe.
  bool finalized_{false};
};

} // namespace facebook::nimble
