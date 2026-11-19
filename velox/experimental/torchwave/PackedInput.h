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

#pragma once

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <c10/util/ArrayRef.h>

#include "velox/experimental/torchwave/PackFormat.h"

namespace torch::wave {

/// The host side of packed inputs (format in PackFormat.h): the recipe that
/// says how an input divides into slots, and the encoder and decoder.

/// One slot of a packed input.
struct PackSlot {
  /// Position of the slot's length among the graph's user inputs, or -1 when
  /// the length is the constant 'constLength'.
  int32_t lengthInput{-1};
  int64_t constLength{-1};

  /// False if nothing reads the slot's values. The encoder then stores nothing
  /// for it.
  bool used{true};

  /// If > 0, every path from the slot's values ends in aten.remainder with this
  /// divisor, so the encoder may store the values already reduced: the
  /// remainder stays in the graph and leaves reduced values unchanged.
  int64_t modulus{0};

  /// Row of the recipe's lengths input that holds this slot's length per
  /// batch row, or -1. Set together with a window below.
  int32_t lengthsRow{-1};

  /// Window. The host trims every batch row of the slot before encoding it,
  /// and writes the trimmed lengths back to the lengths input and to the
  /// slot's length input, so the graph reads data that is already trimmed. A
  /// rewrite that sets a window removes from the graph the op that did the
  /// trimming there.
  ///
  /// If >= 0, slots of the same guard group are first trimmed to the per-row
  /// minimum of their lengths, keeping the head of each row.
  int32_t guardGroup{-1};

  /// If >= 0, each row is then trimmed to at most this many values: its last
  /// ones if 'keepTail', else its first.
  int64_t maxLength{-1};
  bool keepTail{false};

  bool hasWindow() const {
    return guardGroup >= 0 || maxLength >= 0;
  }
};

/// What the host must do to an input for the rewritten graph: which input,
/// and how its values divide into slots. Produced by rewritePackedInputs() and
/// kept with the compiled graph.
struct PackRecipe {
  /// Name of the packed graph input and its position among the user inputs.
  std::string inputName;
  int32_t inputIndex{-1};
  std::vector<PackSlot> slots;

  /// The user input holding the per-row lengths of the windowed slots, a
  /// [lengthsRows, batch rows] tensor flattened to 1-d, or -1 if no slot has a
  /// window.
  int32_t lengthsInput{-1};
  int64_t lengthsRows{0};

  /// One line per recipe, then one per slot, for logs.
  std::string toString() const;

  /// JSON object with the fields above.
  std::string toJson() const;
};

/// Host-side knobs of encodePacked() and PackEncoder.
struct PackOptions {
  /// Threads that compute the slot ranges and write the encoded values. 1 runs
  /// on the calling thread.
  int32_t numThreads{1};

  /// Threads that write the encoded values, if not -1; else 'numThreads'. A
  /// batch merger that already runs on a pool writes on its own thread (1)
  /// and keeps 'numThreads' for the range passes.
  int32_t writeThreads{-1};

  /// Allocate the encoded tensor in page-locked memory, so that its copy to
  /// the device is a DMA from it.
  bool pinned{false};

  /// Store the values of a slot with a modulus already reduced.
  bool reduce{true};

  /// Values per unit of work. A slot larger than this is split across threads.
  int64_t chunkValues{1 << 16};

  /// Keep each slot's frame across the encode() calls of one PackEncoder. The
  /// first call measures every slot. Later calls write with the kept frames,
  /// checking each block of values against its slot's frame. A slot whose
  /// values do not fit is measured again and re-encoded in a frame widened by
  /// an eighth of its span on each side. Frames are never narrowed.
  bool keepFrames{false};
  /// If > 0, the id hashing a trainer applies to the unpacked values after
  /// H2D (max_hash_size): every decoded value v becomes v % hashSize
  /// (truncated toward zero, as torch.fmod). The decoder applies it, so the
  /// host stores the values unhashed and does no modulo. A slot with a
  /// modulus is then stored unreduced, since its remainder must follow the
  /// hash; the graph takes the remainder as it does for any slot.
  int64_t hashSize{0};

  /// With keepFrames, the encoded buffer is allocated this much larger than
  /// the layout needs, so that a re-encode that widens a slot usually fits.
  double bufferHeadroom{0.15};
};

/// Where the time and the bytes of one encoding went.
struct PackStats {
  int64_t inputBytes{0};
  int64_t packedBytes{0};
  int64_t rangeUs{0};
  int64_t layoutUs{0};
  int64_t writeUs{0};
  int64_t totalUs{0};

  /// Time to trim the windowed slots, and the values the windows dropped.
  int64_t windowUs{0};
  int64_t droppedValues{0};

  /// Slots by stored width: 0 (constant or unused), 8, 16, 32 and 64 bits.
  int32_t slotsByWidth[5]{};

  /// With keepFrames: whether this call measured every slot, the slots
  /// re-encoded because their values left the kept frame, the time that took,
  /// the bytes written again because later slots moved, and whether the
  /// buffer had to be reallocated (and copied) to fit.
  bool measured{false};
  int32_t reencodedSlots{0};
  int64_t reencodeUs{0};
  int64_t rewrittenBytes{0};
  int32_t regrownBuffers{0};

  /// From the effective ranges of this call: the data bytes at the byte
  /// widths the values actually need, and at their exact bit widths (bit
  /// packing), next to the data bytes stored. Slots stored wider than they
  /// need.
  int64_t storedDataBytes{0};
  int64_t neededDataBytes{0};
  int64_t bitPackedDataBytes{0};
  int32_t widerThanNeeded{0};
};

/// A run of one slot's values in host memory. A slot may come in several
/// runs, for example one per batch merged into it. They are encoded in order,
/// as if concatenated.
struct SlotSegment {
  const int64_t* values{nullptr};
  int64_t count{0};
};

/// The frame a slot's values are stored in: unsigned offsets from 'base' of
/// 'bits' bits, 0 when every value is 'base'.
struct SlotFrame {
  int64_t base{0};
  int32_t bits{0};
};

/// The smallest and largest value a slot held in one encoding, after any
/// reduction by its modulus. min > max if it held none.
struct SlotRange {
  int64_t min{std::numeric_limits<int64_t>::max()};
  int64_t max{std::numeric_limits<int64_t>::min()};
};

/// Encodes the packed input of one recipe, call after call. Thread-safe: a
/// batch merger may call encode() from several threads; they share the kept
/// frames.
class PackEncoder {
 public:
  PackEncoder(PackRecipe recipe, PackOptions options);

  struct Result {
    /// The encoded words, on the host (pinned if the options say so).
    at::Tensor packed;
    /// Each slot's length after its window, if any.
    std::vector<int64_t> slotLengths;
    /// The lengths input with the windowed slots' rows trimmed, int64 on the
    /// host; undefined if the recipe has no windows.
    at::Tensor rowLengths;
  };

  /// Encodes one execution's values of the recipe's input: slot k's values
  /// are 'segments[k]'. If the recipe has windows, 'rowLengths' is its lengths
  /// input, [lengthsRows * batch rows], any integer type and device.
  Result encode(
      const std::vector<std::vector<SlotSegment>>& segments,
      const at::Tensor& rowLengths = {},
      PackStats* stats = nullptr);

  const PackRecipe& recipe() const {
    return recipe_;
  }

  /// The kept frames, empty before the first encode() with keepFrames.
  std::vector<SlotFrame> frames() const;

  /// Each slot's effective range in the latest encode().
  std::vector<SlotRange> ranges() const;

 private:
  const PackRecipe recipe_;
  const PackOptions options_;
  mutable std::mutex mutex_;
  std::vector<SlotFrame> frames_;
  std::vector<SlotRange> ranges_;
};

/// The slot lengths for 'recipe' from the user inputs of one execution,
/// 'inputs' being in graph user-input order.
std::vector<int64_t> packSlotLengths(
    const PackRecipe& recipe,
    const std::vector<c10::IValue>& inputs);

/// Encodes 'values' (int64, 1-d, contiguous, on the host) as the packed input
/// of 'recipe', given the length of each slot. The lengths must cover 'values'
/// exactly. Returns the encoded words on the host.
at::Tensor encodePacked(
    const PackRecipe& recipe,
    const at::Tensor& values,
    c10::ArrayRef<int64_t> slotLengths,
    const PackOptions& options = {},
    PackStats* stats = nullptr);

/// Encodes, in place, each packed input of 'recipes' among 'inputs', the user
/// inputs of one execution in graph order. An input on the device is encoded
/// from a host copy. A recipe with windows also has its lengths input and the
/// length inputs of its windowed slots replaced by the trimmed ones, on the
/// device they came from. Appends one PackStats per recipe to 'stats' if
/// given.
void encodePackedInputs(
    const std::vector<PackRecipe>& recipes,
    std::vector<c10::IValue>& inputs,
    const PackOptions& options,
    std::vector<PackStats>* stats);

/// Decodes an encoded host tensor back to its int64 values, the reference for
/// unpack_slot. Unused slots decode to their base, zero.
at::Tensor decodePacked(const at::Tensor& packed);

/// Makes every slot of the encoded host tensor 'packed' decode with
/// PackOptions::hashSize 'hashSize', for a hash known only after the encoding,
/// such as a batch's max_hash_size. Fails if a slot was stored reduced by its
/// modulus and 'hashSize' is not 0: encode with 'reduce' off when a hash may
/// come later.
void setPackedHash(at::Tensor& packed, int64_t hashSize);

} // namespace torch::wave
