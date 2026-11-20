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

#include "velox/experimental/torchwave/PackedInput.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>
#include <unordered_map>

#include <ATen/ATen.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <xsimd/xsimd.hpp>

namespace torch::wave {
namespace {

// --- Encoding ---

int64_t alignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

int64_t microsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

// Runs fn(i) for i in [0, numItems) on up to 'numThreads' threads, the caller
// being one of them. The first exception any fn(i) throws is rethrown on the
// caller once all threads are done; the remaining items are skipped.
template <typename Fn>
void runParallel(int32_t numThreads, int64_t numItems, const Fn& fn) {
  const auto numWorkers =
      static_cast<int32_t>(std::min<int64_t>(numThreads, numItems));
  if (numWorkers <= 1) {
    for (int64_t i = 0; i < numItems; ++i) {
      fn(i);
    }
    return;
  }
  std::atomic<int64_t> next{0};
  std::mutex errorMutex;
  std::exception_ptr error;
  auto work = [&]() {
    try {
      for (int64_t i = next.fetch_add(1); i < numItems; i = next.fetch_add(1)) {
        fn(i);
      }
    } catch (...) {
      next = numItems;
      std::lock_guard<std::mutex> lock(errorMutex);
      if (!error) {
        error = std::current_exception();
      }
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(numWorkers - 1);
  for (int32_t i = 1; i < numWorkers; ++i) {
    threads.emplace_back(work);
  }
  work();
  for (auto& thread : threads) {
    thread.join();
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

// x % divisor for 0 < divisor < 2^63 with a multiply instead of a divide.
// The quotient estimate mulhi(x, floor((2^64 - 1) / divisor)) is at most 2
// below the quotient, so two conditional subtractions finish it.
class FastMod {
 public:
  FastMod() = default;

  explicit FastMod(int64_t divisor)
      : divisor_{static_cast<uint64_t>(divisor)},
        inverse_{~uint64_t{0} / static_cast<uint64_t>(divisor)} {}

  uint64_t operator()(uint64_t x) const {
    const auto quotient = static_cast<uint64_t>(
        (static_cast<unsigned __int128>(x) * inverse_) >> 64);
    auto remainder = x - quotient * divisor_;
    remainder -= divisor_ & (uint64_t{0} - (remainder >= divisor_));
    remainder -= divisor_ & (uint64_t{0} - (remainder >= divisor_));
    return remainder;
  }

  // torch.fmod: truncated toward zero, the sign of 'value'. Branch-free: the
  // signs of ids are random.
  int64_t fmod(int64_t value) const {
    const auto sign = static_cast<uint64_t>(value >> 63);
    const auto remainder =
        (*this)((static_cast<uint64_t>(value) ^ sign) - sign);
    return static_cast<int64_t>((remainder ^ sign) - sign);
  }

  // torch.remainder: in [0, divisor).
  int64_t remainder(int64_t value) const {
    const auto r = fmod(value);
    return r + (static_cast<int64_t>(divisor_) & (r >> 63));
  }

 private:
  uint64_t divisor_{1};
  uint64_t inverse_{~uint64_t{0}};
};

// What a value goes through before it is stored: torch.remainder by the
// slot's modulus if modulus > 0. The trainer's hash is the decoder's
// (PackOptions::hashSize), so the host applies none.
struct Reduction {
  explicit Reduction(int64_t modulus)
      : modulus{modulus},
        modulusMod{modulus > 0 ? FastMod(modulus) : FastMod()} {}

  bool any() const {
    return modulus > 0;
  }

  // Whether every value in [min, max] is left as it is.
  bool keeps(int64_t min, int64_t max) const {
    return modulus <= 0 || (min >= 0 && max < modulus);
  }

  void apply(const int64_t* values, int64_t count, int64_t* out) const {
    for (int64_t i = 0; i < count; ++i) {
      out[i] = modulusMod.remainder(values[i]);
    }
  }

  int64_t modulus{0};
  FastMod modulusMod;
};

// The modulus the host reduces slot 'k' by: none when reducing is off, and
// none with a hash, whose remainder the graph must take after it.
int64_t
storedModulus(const PackRecipe& recipe, const PackOptions& options, int64_t k) {
  return options.reduce && options.hashSize == 0 ? recipe.slots.at(k).modulus
                                                 : 0;
}

// Values handled at a time by the fused check-and-store: small enough that a
// block read for its min and max is still in L1 when it is stored.
constexpr int64_t kBlockValues = 1'024;

SlotRange& extend(SlotRange& range, const SlotRange& other) {
  range.min = std::min(range.min, other.min);
  range.max = std::max(range.max, other.max);
  return range;
}

// Min and max of 'count' values, with SIMD.
SlotRange minMax(const int64_t* values, int64_t count) {
  using Batch = xsimd::batch<int64_t>;
  constexpr int64_t kLanes = Batch::size;
  SlotRange range;
  int64_t i = 0;
  if (count >= kLanes) {
    auto low = Batch::load_unaligned(values);
    auto high = low;
    for (i = kLanes; i + kLanes <= count; i += kLanes) {
      const auto batch = Batch::load_unaligned(values + i);
      low = xsimd::min(low, batch);
      high = xsimd::max(high, batch);
    }
    range.min = xsimd::reduce_min(low);
    range.max = xsimd::reduce_max(high);
  }
  for (; i < count; ++i) {
    range.min = std::min(range.min, values[i]);
    range.max = std::max(range.max, values[i]);
  }
  return range;
}

// 'count' values, each through 'reduction', a block at a time: calls
// fn(block, blockCount, blockRange) for each block. A block whose values the
// reduction keeps is passed as it is. Returns false as soon as fn does.
template <typename Fn>
bool forEachBlock(
    const int64_t* values,
    int64_t count,
    const Reduction& reduction,
    const Fn& fn) {
  int64_t reduced[kBlockValues];
  for (int64_t begin = 0; begin < count; begin += kBlockValues) {
    const auto n = std::min(kBlockValues, count - begin);
    const auto* block = values + begin;
    auto range = minMax(block, n);
    if (reduction.any() && !reduction.keeps(range.min, range.max)) {
      reduction.apply(block, n, reduced);
      block = reduced;
      range = minMax(block, n);
    }
    if (!fn(block, n, range)) {
      return false;
    }
  }
  return true;
}

SlotRange
valueRange(const int64_t* values, int64_t count, const Reduction& reduction) {
  SlotRange range;
  forEachBlock(values, count, reduction, [&](const int64_t*, int64_t, auto r) {
    extend(range, r);
    return true;
  });
  return range;
}

// Width in bits that holds every value of 'range' as an offset from its
// minimum.
int32_t widthOf(const SlotRange& range) {
  if (range.min >= range.max) {
    return 0;
  }
  const auto span =
      static_cast<uint64_t>(range.max) - static_cast<uint64_t>(range.min);
  if (span <= std::numeric_limits<uint8_t>::max()) {
    return 8;
  }
  if (span <= std::numeric_limits<uint16_t>::max()) {
    return 16;
  }
  if (span <= std::numeric_limits<uint32_t>::max()) {
    return 32;
  }
  return 64;
}

// Exact bits that hold the span of 'range', for the bit-packing estimate.
int32_t exactBitsOf(const SlotRange& range) {
  if (range.min >= range.max) {
    return 0;
  }
  const auto span =
      static_cast<uint64_t>(range.max) - static_cast<uint64_t>(range.min);
  return 64 - __builtin_clzll(span);
}

int32_t widthIndex(int64_t bits) {
  switch (bits) {
    case 8:
      return 1;
    case 16:
      return 2;
    case 32:
      return 3;
    case 64:
      return 4;
    default:
      return 0;
  }
}

SlotFrame exactFrame(const SlotRange& range) {
  if (range.min > range.max) {
    return {};
  }
  return {range.min, widthOf(range)};
}

// The frame for a re-encode: 'range' widened by an eighth of its span on each
// side, clamped to int64, so that a slot near a width limit goes to the next
// width rather than re-encoding again soon.
SlotFrame widenedFrame(const SlotRange& range) {
  const auto span =
      static_cast<uint64_t>(range.max) - static_cast<uint64_t>(range.min);
  const auto margin = span / 8;
  constexpr auto kMin = std::numeric_limits<int64_t>::min();
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  SlotRange widened;
  widened.min =
      static_cast<uint64_t>(range.min) - static_cast<uint64_t>(kMin) < margin
      ? kMin
      : static_cast<int64_t>(static_cast<uint64_t>(range.min) - margin);
  widened.max =
      static_cast<uint64_t>(kMax) - static_cast<uint64_t>(range.max) < margin
      ? kMax
      : static_cast<int64_t>(static_cast<uint64_t>(range.max) + margin);
  return exactFrame(widened);
}

// Whether every value of 'range' is stored exactly in 'frame'.
bool fits(const SlotRange& range, const SlotFrame& frame) {
  if (range.min > range.max) {
    return true;
  }
  if (range.min < frame.base) {
    return false;
  }
  const auto offset =
      static_cast<uint64_t>(range.max) - static_cast<uint64_t>(frame.base);
  return frame.bits == 64 ||
      offset < (frame.bits == 0 ? uint64_t{1} : uint64_t{1} << frame.bits);
}

template <typename T>
void storeOffsets(const int64_t* values, int64_t count, uint64_t base, T* out) {
  for (int64_t i = 0; i < count; ++i) {
    out[i] = static_cast<T>(static_cast<uint64_t>(values[i]) - base);
  }
}

// Stores 'count' values at 'out' as offsets in 'frame', a block at a time,
// checking each block against the frame before storing it. Returns the
// values' range if all of them fit, else nullopt.
std::optional<SlotRange> storeChecked(
    const int64_t* values,
    int64_t count,
    const Reduction& reduction,
    const SlotFrame& frame,
    uint8_t* out) {
  SlotRange range;
  const auto base = static_cast<uint64_t>(frame.base);
  int64_t index{0};
  const bool ok = forEachBlock(
      values, count, reduction, [&](const int64_t* block, int64_t n, auto r) {
        if (!fits(r, frame)) {
          return false;
        }
        extend(range, r);
        switch (frame.bits) {
          case 8:
            storeOffsets(block, n, base, out + index);
            break;
          case 16:
            storeOffsets(
                block, n, base, reinterpret_cast<uint16_t*>(out) + index);
            break;
          case 32:
            storeOffsets(
                block, n, base, reinterpret_cast<uint32_t*>(out) + index);
            break;
          case 64:
            storeOffsets(
                block, n, base, reinterpret_cast<uint64_t*>(out) + index);
            break;
          default:
            break;
        }
        index += n;
        return true;
      });
  if (!ok) {
    return std::nullopt;
  }
  return range;
}

template <typename T>
T loadOffset(const uint8_t* data, int64_t index) {
  T value;
  std::memcpy(&value, data + index * sizeof(T), sizeof(T));
  return value;
}

// A slice of one slot's values within one of its segments, the unit of work
// of the range and write passes. 'index' is where the slice starts among the
// slot's values.
struct Chunk {
  int32_t slot;
  int32_t segment;
  int64_t begin;
  int64_t end;
  int64_t index;
};

// The slot's values from virtual position 'start' (among the slot's values,
// across its segments) for 'count' values, appended to 'out'.
void copyValues(
    const std::vector<SlotSegment>& segments,
    int64_t start,
    int64_t count,
    std::vector<int64_t>& out) {
  int64_t segmentStart{0};
  for (const auto& segment : segments) {
    if (count == 0) {
      return;
    }
    const auto segmentEnd = segmentStart + segment.count;
    if (start < segmentEnd) {
      const auto from = start - segmentStart;
      const auto n = std::min(count, segment.count - from);
      out.insert(out.end(), segment.values + from, segment.values + from + n);
      start += n;
      count -= n;
    }
    segmentStart = segmentEnd;
  }
  TORCH_CHECK(count == 0, "PackEncoder: a row runs past the slot's values");
}

// Each row's start and length after the slot's window, given its length
// 'length' and, for a guarded slot, its group's minimum 'groupMin'.
std::pair<int64_t, int64_t>
rowWindow(const PackSlot& slot, int64_t length, int64_t groupMin) {
  const auto guarded = slot.guardGroup >= 0 ? groupMin : length;
  const auto kept =
      slot.maxLength >= 0 ? std::min(guarded, slot.maxLength) : guarded;
  return {slot.keepTail ? guarded - kept : 0, kept};
}

// The windowed slots of one execution, trimmed.
struct Trimmed {
  // Per slot, its trimmed values if its window cut any.
  std::vector<std::vector<int64_t>> values;
  std::vector<uint8_t> trimmed;
  std::vector<int64_t> slotLengths;
  // The lengths input with the windowed slots' rows trimmed, int64.
  at::Tensor lengths;
  int64_t droppedValues{0};
};

Trimmed trimWindows(
    const PackRecipe& recipe,
    const std::vector<std::vector<SlotSegment>>& segments,
    const std::vector<int64_t>& slotLengths,
    const at::Tensor& lengths,
    int32_t numThreads) {
  const auto numSlots = static_cast<int64_t>(recipe.slots.size());
  TORCH_CHECK(
      lengths.numel() % recipe.lengthsRows == 0,
      "PackEncoder: lengths input is not ",
      recipe.lengthsRows,
      " rows");
  const auto numRows = lengths.numel() / recipe.lengthsRows;
  const auto rowLengths = lengths.to(at::kCPU, at::kLong).contiguous();
  const auto* rowLength = rowLengths.const_data_ptr<int64_t>();
  auto lengthsOf = [&](const PackSlot& slot) {
    return rowLength + slot.lengthsRow * numRows;
  };

  std::unordered_map<int32_t, std::vector<int64_t>> groupMins;
  for (const auto& slot : recipe.slots) {
    if (slot.guardGroup < 0) {
      continue;
    }
    auto [it, inserted] = groupMins.try_emplace(
        slot.guardGroup,
        std::vector<int64_t>(numRows, std::numeric_limits<int64_t>::max()));
    const auto* slotRows = lengthsOf(slot);
    for (int64_t r = 0; r < numRows; ++r) {
      it->second[r] = std::min(it->second[r], slotRows[r]);
    }
  }

  Trimmed trimmed;
  trimmed.values.resize(numSlots);
  trimmed.trimmed.assign(numSlots, false);
  trimmed.slotLengths = slotLengths;
  auto newLengths = rowLengths.clone();
  auto* newRowLength = newLengths.mutable_data_ptr<int64_t>();
  std::vector<int64_t> windowed;
  for (int64_t k = 0; k < numSlots; ++k) {
    if (recipe.slots[k].hasWindow()) {
      windowed.push_back(k);
    }
  }
  runParallel(
      numThreads, static_cast<int64_t>(windowed.size()), [&](int64_t i) {
        const auto k = windowed[i];
        const auto& slot = recipe.slots[k];
        const auto* slotRows = lengthsOf(slot);
        const auto* groupMin = slot.guardGroup >= 0
            ? groupMins.at(slot.guardGroup).data()
            : nullptr;
        auto* newRows = newRowLength + slot.lengthsRow * numRows;
        std::vector<int64_t> starts(numRows);
        int64_t rowStart{0};
        int64_t numKept{0};
        for (int64_t r = 0; r < numRows; ++r) {
          const auto [start, kept] =
              rowWindow(slot, slotRows[r], groupMin ? groupMin[r] : 0);
          starts[r] = rowStart + start;
          newRows[r] = kept;
          numKept += kept;
          rowStart += slotRows[r];
        }
        TORCH_CHECK(
            rowStart == slotLengths.at(k),
            "PackEncoder: row lengths of slot ",
            k,
            " do not add up to its length");
        // A window that cuts nothing leaves the values where they are.
        if (numKept == slotLengths.at(k)) {
          return;
        }
        auto& out = trimmed.values[k];
        out.reserve(numKept);
        for (int64_t r = 0; r < numRows; ++r) {
          copyValues(segments[k], starts[r], newRows[r], out);
        }
        trimmed.trimmed[k] = true;
      });
  for (auto k : windowed) {
    if (trimmed.trimmed[k]) {
      trimmed.droppedValues +=
          slotLengths.at(k) - static_cast<int64_t>(trimmed.values[k].size());
      trimmed.slotLengths.at(k) =
          static_cast<int64_t>(trimmed.values[k].size());
    }
  }
  trimmed.lengths = newLengths;
  return trimmed;
}

// Where each slot's data goes, from the slots' frames and lengths.
struct Layout {
  std::vector<int64_t> byteOffsets;
  int64_t dataOffset{0};
  int64_t totalBytes{0};
};

Layout layoutOf(
    const std::vector<SlotFrame>& frames,
    const std::vector<int64_t>& slotLengths) {
  const auto numSlots = static_cast<int64_t>(frames.size());
  Layout layout;
  layout.byteOffsets.resize(numSlots);
  int64_t byteOffset =
      alignUp((kPackHeaderWords + numSlots * kPackDescWords) * 8, kPackAlign);
  layout.dataOffset = byteOffset;
  for (int64_t k = 0; k < numSlots; ++k) {
    layout.byteOffsets[k] = byteOffset;
    byteOffset =
        alignUp(byteOffset + slotLengths[k] * frames[k].bits / 8, kPackAlign);
  }
  layout.totalBytes = byteOffset;
  return layout;
}

void writeHeader(
    int64_t* words,
    const PackRecipe& recipe,
    const PackOptions& options,
    const std::vector<SlotFrame>& frames,
    const std::vector<int64_t>& slotLengths,
    const Layout& layout) {
  const auto numSlots = static_cast<int64_t>(frames.size());
  std::fill(words, words + layout.dataOffset / 8, 0);
  int64_t valueOffset{0};
  for (int64_t k = 0; k < numSlots; ++k) {
    auto* desc = words + kPackHeaderWords + k * kPackDescWords;
    const auto modulus = storedModulus(recipe, options, k);
    desc[kPackDescByteOffset] = layout.byteOffsets.at(k);
    desc[kPackDescByteLength] = slotLengths.at(k) * frames[k].bits / 8;
    desc[kPackDescValueCount] = slotLengths.at(k);
    desc[kPackDescBits] = frames[k].bits;
    desc[kPackDescBase] = frames[k].base;
    desc[kPackDescFlags] =
        frames[k].bits != 0 && modulus > 0 ? kPackReduced : 0;
    desc[kPackDescValueOffset] = valueOffset;
    desc[kPackDescHash] = options.hashSize;
    valueOffset += slotLengths.at(k);
    // The alignment padding after the slot's data.
    const auto dataEnd = desc[kPackDescByteOffset] + desc[kPackDescByteLength];
    const auto next =
        k + 1 < numSlots ? layout.byteOffsets.at(k + 1) : layout.totalBytes;
    std::memset(reinterpret_cast<uint8_t*>(words) + dataEnd, 0, next - dataEnd);
  }
  words[kPackHeaderMagic] = kPackMagic;
  words[kPackHeaderNumSlots] = numSlots;
  words[kPackHeaderNumValues] = valueOffset;
  words[kPackHeaderDataOffset] = layout.dataOffset;
  words[kPackHeaderTotalBytes] = layout.totalBytes;
}

at::Tensor allocateWords(int64_t numWords, bool pinned) {
  return at::empty(
      {numWords}, at::TensorOptions().dtype(at::kLong).pinned_memory(pinned));
}

} // namespace

std::string PackRecipe::toString() const {
  int32_t numUsed{0};
  int32_t numReduced{0};
  for (const auto& slot : slots) {
    numUsed += slot.used;
    numReduced += slot.modulus > 0;
  }
  std::string out = fmt::format(
      "packed input {} (user input {}): {} slots, {} read, {} reduced\n",
      inputName,
      inputIndex,
      slots.size(),
      numUsed,
      numReduced);
  for (size_t k = 0; k < slots.size(); ++k) {
    const auto& slot = slots[k];
    out += fmt::format(
        "  slot {}: length {} {}{}{}\n",
        k,
        slot.lengthInput >= 0 ? "input" : "constant",
        slot.lengthInput >= 0 ? slot.lengthInput : slot.constLength,
        slot.used ? "" : ", unread",
        slot.modulus > 0 ? fmt::format(", % {}", slot.modulus) : "");
    if (slot.hasWindow()) {
      out.pop_back();
      out += fmt::format(
          ", lengths row {}{}{}\n",
          slot.lengthsRow,
          slot.guardGroup >= 0 ? fmt::format(", guard {}", slot.guardGroup)
                               : "",
          slot.maxLength >= 0 ? fmt::format(
                                    ", max {} {}",
                                    slot.maxLength,
                                    slot.keepTail ? "tail" : "head")
                              : "");
    }
  }
  return out;
}

std::string PackRecipe::toJson() const {
  nlohmann::json json;
  json["inputName"] = inputName;
  json["inputIndex"] = inputIndex;
  json["lengthsInput"] = lengthsInput;
  json["lengthsRows"] = lengthsRows;
  auto& slotsJson = json["slots"] = nlohmann::json::array();
  for (const auto& slot : slots) {
    slotsJson.push_back(
        {{"lengthInput", slot.lengthInput},
         {"constLength", slot.constLength},
         {"used", slot.used},
         {"modulus", slot.modulus},
         {"lengthsRow", slot.lengthsRow},
         {"guardGroup", slot.guardGroup},
         {"maxLength", slot.maxLength},
         {"keepTail", slot.keepTail}});
  }
  return json.dump();
}

std::vector<int64_t> packSlotLengths(
    const PackRecipe& recipe,
    const std::vector<c10::IValue>& inputs) {
  std::vector<int64_t> lengths;
  lengths.reserve(recipe.slots.size());
  for (const auto& slot : recipe.slots) {
    if (slot.lengthInput < 0) {
      lengths.push_back(slot.constLength);
      continue;
    }
    TORCH_CHECK(
        slot.lengthInput < static_cast<int32_t>(inputs.size()),
        "packSlotLengths: no user input for a slot length: ",
        slot.lengthInput);
    lengths.push_back(inputs[slot.lengthInput].toInt());
  }
  return lengths;
}

PackEncoder::PackEncoder(PackRecipe recipe, PackOptions options)
    : recipe_(std::move(recipe)), options_(options) {}

std::vector<SlotFrame> PackEncoder::frames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return frames_;
}

std::vector<SlotRange> PackEncoder::ranges() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ranges_;
}

PackEncoder::Result PackEncoder::encode(
    const std::vector<std::vector<SlotSegment>>& inputSegments,
    const at::Tensor& rowLengths,
    PackStats* stats) {
  const auto start = std::chrono::steady_clock::now();
  const auto numSlots = static_cast<int64_t>(recipe_.slots.size());
  TORCH_CHECK(
      static_cast<int64_t>(inputSegments.size()) == numSlots,
      "PackEncoder: one segment list per slot needed, got ",
      inputSegments.size());
  PackStats local;
  auto& st = stats != nullptr ? *stats : local;
  st = PackStats{};
  const auto writeThreads =
      options_.writeThreads > 0 ? options_.writeThreads : options_.numThreads;
  std::vector<Reduction> reductions;
  reductions.reserve(numSlots);
  for (int64_t k = 0; k < numSlots; ++k) {
    reductions.emplace_back(storedModulus(recipe_, options_, k));
  }
  auto modulusOf = [&](int64_t slot) -> const Reduction& {
    return reductions[slot];
  };

  Result result;
  result.slotLengths.assign(numSlots, 0);
  for (int64_t k = 0; k < numSlots; ++k) {
    for (const auto& segment : inputSegments[k]) {
      TORCH_CHECK(segment.count >= 0, "PackEncoder: negative segment length");
      result.slotLengths[k] += segment.count;
    }
    st.inputBytes += result.slotLengths[k] * 8;
  }

  // Windows replace a slot that they cut by one segment of its kept values.
  std::vector<std::vector<SlotSegment>> trimmedSegments;
  const auto* segments = &inputSegments;
  Trimmed trimmed;
  if (recipe_.lengthsInput >= 0) {
    TORCH_CHECK(
        rowLengths.defined(), "PackEncoder: the recipe's windows need lengths");
    const auto windowStart = std::chrono::steady_clock::now();
    trimmed = trimWindows(
        recipe_,
        inputSegments,
        result.slotLengths,
        rowLengths,
        options_.numThreads);
    trimmedSegments = inputSegments;
    for (int64_t k = 0; k < numSlots; ++k) {
      if (trimmed.trimmed[k]) {
        trimmedSegments[k] = {
            {trimmed.values[k].data(),
             static_cast<int64_t>(trimmed.values[k].size())}};
      }
    }
    segments = &trimmedSegments;
    result.slotLengths = trimmed.slotLengths;
    result.rowLengths = trimmed.lengths;
    st.windowUs = microsSince(windowStart);
    st.droppedValues = trimmed.droppedValues;
  }
  const auto& slotLengths = result.slotLengths;

  std::vector<Chunk> chunks;
  std::vector<std::vector<int32_t>> slotChunks(numSlots);
  for (int64_t k = 0; k < numSlots; ++k) {
    if (!recipe_.slots[k].used) {
      continue;
    }
    int64_t index{0};
    for (size_t s = 0; s < (*segments)[k].size(); ++s) {
      const auto count = (*segments)[k][s].count;
      for (int64_t begin = 0; begin < count; begin += options_.chunkValues) {
        const auto end = std::min(begin + options_.chunkValues, count);
        slotChunks[k].push_back(static_cast<int32_t>(chunks.size()));
        chunks.push_back(
            {static_cast<int32_t>(k),
             static_cast<int32_t>(s),
             begin,
             end,
             index + begin});
      }
      index += count;
    }
  }
  auto chunkValues = [&](const Chunk& chunk) {
    return (*segments)[chunk.slot][chunk.segment].values + chunk.begin;
  };
  // The exact range of each slot in 'slots', measured on 'numThreads'.
  auto measure = [&](const std::vector<int64_t>& slots) {
    std::vector<int32_t> work;
    for (auto k : slots) {
      work.insert(work.end(), slotChunks[k].begin(), slotChunks[k].end());
    }
    std::vector<SlotRange> chunkRanges(work.size());
    runParallel(
        options_.numThreads, static_cast<int64_t>(work.size()), [&](int64_t i) {
          const auto& chunk = chunks[work[i]];
          chunkRanges[i] = valueRange(
              chunkValues(chunk),
              chunk.end - chunk.begin,
              modulusOf(chunk.slot));
        });
    std::vector<SlotRange> slotRanges(numSlots);
    for (size_t i = 0; i < work.size(); ++i) {
      extend(slotRanges[chunks[work[i]].slot], chunkRanges[i]);
    }
    return slotRanges;
  };

  // The frames: kept ones, or measured now. Only one thread measures them
  // the first time; the others wait for it.
  std::vector<SlotFrame> frames;
  std::unique_lock<std::mutex> measureLock(mutex_);
  st.measured = !options_.keepFrames || frames_.empty();
  if (!st.measured) {
    frames = frames_;
    measureLock.unlock();
  } else {
    if (!options_.keepFrames) {
      measureLock.unlock();
    }
    const auto rangeStart = std::chrono::steady_clock::now();
    std::vector<int64_t> all(numSlots);
    std::iota(all.begin(), all.end(), 0);
    const auto slotRanges = measure(all);
    frames.resize(numSlots);
    for (int64_t k = 0; k < numSlots; ++k) {
      if (recipe_.slots[k].used && slotLengths[k] > 0) {
        frames[k] = exactFrame(slotRanges[k]);
      }
    }
    st.rangeUs = microsSince(rangeStart);
  }

  const auto layoutStart = std::chrono::steady_clock::now();
  auto layout = layoutOf(frames, slotLengths);
  auto capacityWords = layout.totalBytes / 8;
  if (options_.keepFrames) {
    capacityWords = alignUp(
        static_cast<int64_t>(
            static_cast<double>(capacityWords) * (1 + options_.bufferHeadroom)),
        kPackAlign / 8);
  }
  auto buffer = allocateWords(capacityWords, options_.pinned);
  auto* words = buffer.mutable_data_ptr<int64_t>();
  writeHeader(words, recipe_, options_, frames, slotLengths, layout);
  st.layoutUs = microsSince(layoutStart);

  // Write, checking each block against its slot's frame.
  const auto writeStart = std::chrono::steady_clock::now();
  std::vector<SlotRange> chunkRanges(chunks.size());
  auto failed = std::make_unique<std::atomic<bool>[]>(numSlots);
  for (int64_t k = 0; k < numSlots; ++k) {
    failed[k].store(false, std::memory_order_relaxed);
  }
  auto writeChunks = [&](const std::vector<int32_t>& work) {
    runParallel(
        writeThreads, static_cast<int64_t>(work.size()), [&](int64_t i) {
          const auto& chunk = chunks[work[i]];
          const auto k = chunk.slot;
          if (failed[k].load(std::memory_order_relaxed)) {
            return;
          }
          const auto& frame = frames[k];
          auto* out = reinterpret_cast<uint8_t*>(words) +
              layout.byteOffsets[k] + chunk.index * frame.bits / 8;
          auto range = storeChecked(
              chunkValues(chunk),
              chunk.end - chunk.begin,
              modulusOf(k),
              frame,
              out);
          if (!range.has_value()) {
            failed[k].store(true, std::memory_order_relaxed);
            return;
          }
          chunkRanges[work[i]] = *range;
        });
  };
  std::vector<int32_t> allChunks(chunks.size());
  std::iota(allChunks.begin(), allChunks.end(), 0);
  writeChunks(allChunks);
  st.writeUs = microsSince(writeStart);

  // Re-encode the slots whose values left their frame, and move the slots
  // after the first of them.
  std::vector<int64_t> reencoded;
  for (int64_t k = 0; k < numSlots; ++k) {
    if (failed[k].load()) {
      reencoded.push_back(k);
    }
  }
  if (!reencoded.empty()) {
    TORCH_CHECK(!st.measured, "PackEncoder: measured frame does not fit");
    const auto reencodeStart = std::chrono::steady_clock::now();
    const auto slotRanges = measure(reencoded);
    for (auto k : reencoded) {
      frames[k] = widenedFrame(slotRanges[k]);
      failed[k].store(false);
    }
    const auto first = reencoded.front();
    const auto oldLayout = layout;
    layout = layoutOf(frames, slotLengths);
    st.rewrittenBytes = oldLayout.totalBytes - oldLayout.byteOffsets.at(first);
    if (layout.totalBytes / 8 > capacityWords) {
      // Keep what is written up to the first moved slot.
      capacityWords = alignUp(
          static_cast<int64_t>(
              static_cast<double>(layout.totalBytes / 8) *
              (1 + options_.bufferHeadroom)),
          kPackAlign / 8);
      auto grown = allocateWords(capacityWords, options_.pinned);
      std::memcpy(
          grown.mutable_data_ptr<int64_t>(),
          words,
          oldLayout.byteOffsets.at(first));
      buffer = std::move(grown);
      words = buffer.mutable_data_ptr<int64_t>();
      st.regrownBuffers = 1;
    }
    writeHeader(words, recipe_, options_, frames, slotLengths, layout);
    std::vector<int32_t> moved;
    for (int64_t k = first; k < numSlots; ++k) {
      moved.insert(moved.end(), slotChunks[k].begin(), slotChunks[k].end());
    }
    writeChunks(moved);
    for (int64_t k = first; k < numSlots; ++k) {
      TORCH_CHECK(
          !failed[k].load(), "PackEncoder: re-encoded slot does not fit");
    }
    st.reencodedSlots = static_cast<int32_t>(reencoded.size());
    st.reencodeUs = microsSince(reencodeStart);
  }

  std::vector<SlotRange> slotRanges(numSlots);
  for (size_t i = 0; i < chunks.size(); ++i) {
    extend(slotRanges[chunks[i].slot], chunkRanges[i]);
  }
  {
    if (!measureLock.owns_lock()) {
      measureLock.lock();
    }
    if (options_.keepFrames) {
      if (st.measured) {
        frames_ = frames;
      } else {
        // Another thread may have widened a slot meanwhile: keep the wider.
        for (auto k : reencoded) {
          if (frames[k].bits > frames_[k].bits ||
              (frames[k].bits == frames_[k].bits &&
               frames[k].base < frames_[k].base)) {
            frames_[k] = frames[k];
          }
        }
      }
    }
    ranges_ = slotRanges;
    measureLock.unlock();
  }

  result.packed = capacityWords == layout.totalBytes / 8
      ? buffer
      : buffer.narrow(0, 0, layout.totalBytes / 8);
  st.packedBytes = layout.totalBytes;
  for (int64_t k = 0; k < numSlots; ++k) {
    ++st.slotsByWidth[widthIndex(frames[k].bits)];
    st.storedDataBytes += slotLengths[k] * frames[k].bits / 8;
    if (slotLengths[k] > 0 && recipe_.slots[k].used) {
      const auto needed = widthOf(slotRanges[k]);
      st.neededDataBytes += slotLengths[k] * needed / 8;
      st.bitPackedDataBytes +=
          (slotLengths[k] * exactBitsOf(slotRanges[k]) + 7) / 8;
      st.widerThanNeeded += needed < frames[k].bits;
    }
  }
  st.totalUs = microsSince(start);
  return result;
}

at::Tensor encodePacked(
    const PackRecipe& recipe,
    const at::Tensor& values,
    c10::ArrayRef<int64_t> slotLengths,
    const PackOptions& options,
    PackStats* stats) {
  TORCH_CHECK(
      values.dim() == 1 && values.scalar_type() == at::kLong &&
          values.is_contiguous() && values.device().is_cpu(),
      "encodePacked: values must be 1-d contiguous int64 on the host");
  TORCH_CHECK(
      slotLengths.size() == recipe.slots.size(),
      "encodePacked: one length per slot needed, got ",
      slotLengths.size());
  std::vector<std::vector<SlotSegment>> segments;
  segments.reserve(slotLengths.size());
  const auto* source = values.const_data_ptr<int64_t>();
  int64_t offset{0};
  for (auto length : slotLengths) {
    segments.push_back({{source + offset, length}});
    offset += length;
  }
  TORCH_CHECK(
      offset == values.numel(),
      "encodePacked: slot lengths do not cover the values of ",
      recipe.inputName);
  auto plain = recipe;
  plain.lengthsInput = -1;
  auto oneShot = options;
  oneShot.keepFrames = false;
  return PackEncoder(std::move(plain), oneShot)
      .encode(segments, {}, stats)
      .packed;
}

void encodePackedInputs(
    const std::vector<PackRecipe>& recipes,
    std::vector<c10::IValue>& inputs,
    const PackOptions& options,
    std::vector<PackStats>* stats) {
  auto oneShot = options;
  oneShot.keepFrames = false;
  for (const auto& recipe : recipes) {
    TORCH_CHECK(
        recipe.inputIndex < static_cast<int32_t>(inputs.size()) &&
            inputs[recipe.inputIndex].isTensor(),
        "encodePackedInputs: no tensor input for ",
        recipe.inputName);
    const auto lengths = packSlotLengths(recipe, inputs);
    const auto values =
        inputs[recipe.inputIndex].toTensor().to(at::kCPU).contiguous();
    TORCH_CHECK(
        values.numel() ==
            std::accumulate(lengths.begin(), lengths.end(), int64_t{0}),
        "encodePackedInputs: slot lengths do not cover the values of ",
        recipe.inputName);
    std::vector<std::vector<SlotSegment>> segments;
    const auto* source = values.const_data_ptr<int64_t>();
    int64_t offset{0};
    for (auto length : lengths) {
      segments.push_back({{source + offset, length}});
      offset += length;
    }
    PackStats packStats;
    at::Tensor rowLengths;
    if (recipe.lengthsInput >= 0) {
      rowLengths = inputs[recipe.lengthsInput].toTensor();
    }
    auto result =
        PackEncoder(recipe, oneShot).encode(segments, rowLengths, &packStats);
    inputs[recipe.inputIndex] = result.packed;
    if (recipe.lengthsInput >= 0) {
      inputs[recipe.lengthsInput] =
          result.rowLengths.to(rowLengths.device(), rowLengths.scalar_type());
      for (size_t k = 0; k < recipe.slots.size(); ++k) {
        if (recipe.slots[k].hasWindow()) {
          inputs[recipe.slots[k].lengthInput] = result.slotLengths[k];
        }
      }
    }
    if (stats != nullptr) {
      stats->push_back(packStats);
    }
  }
}

at::Tensor decodePacked(const at::Tensor& packed) {
  TORCH_CHECK(
      packed.dim() == 1 && packed.scalar_type() == at::kLong &&
          packed.device().is_cpu() && packed.numel() >= kPackHeaderWords,
      "decodePacked: not an encoded host tensor");
  const auto* words = packed.const_data_ptr<int64_t>();
  TORCH_CHECK(words[0] == kPackMagic, "decodePacked: bad magic");
  const auto numSlots = words[1];
  auto values = at::empty({words[2]}, at::kLong);
  auto* out = values.mutable_data_ptr<int64_t>();
  const auto* bytes = reinterpret_cast<const uint8_t*>(words);
  for (int64_t k = 0; k < numSlots; ++k) {
    const auto* desc = words + kPackHeaderWords + k * kPackDescWords;
    const auto* data = bytes + desc[0];
    const auto count = desc[2];
    const auto base = static_cast<uint64_t>(desc[4]);
    const auto hash = desc[kPackDescHash];
    auto* slotOut = out + desc[6];
    for (int64_t i = 0; i < count; ++i) {
      uint64_t offset{0};
      switch (desc[3]) {
        case 8:
          offset = data[i];
          break;
        case 16:
          offset = loadOffset<uint16_t>(data, i);
          break;
        case 32:
          offset = loadOffset<uint32_t>(data, i);
          break;
        case 64:
          offset = loadOffset<uint64_t>(data, i);
          break;
        default:
          break;
      }
      const auto value = static_cast<int64_t>(base + offset);
      slotOut[i] = hash > 0 ? value % hash : value;
    }
  }
  return values;
}

void setPackedHash(at::Tensor& packed, int64_t hashSize) {
  TORCH_CHECK(
      packed.dim() == 1 && packed.scalar_type() == at::kLong &&
          packed.device().is_cpu() && packed.numel() >= kPackHeaderWords,
      "setPackedHash: not an encoded host tensor");
  TORCH_CHECK(hashSize >= 0, "setPackedHash: negative hash ", hashSize);
  auto* words = packed.mutable_data_ptr<int64_t>();
  TORCH_CHECK(
      words[kPackHeaderMagic] == kPackMagic, "setPackedHash: bad magic");
  const auto numSlots = words[kPackHeaderNumSlots];
  for (int64_t k = 0; k < numSlots; ++k) {
    auto* desc = words + kPackHeaderWords + k * kPackDescWords;
    TORCH_CHECK(
        hashSize == 0 || (desc[kPackDescFlags] & kPackReduced) == 0,
        "setPackedHash: slot ",
        k,
        " is stored reduced by its modulus, which must follow the hash");
    desc[kPackDescHash] = hashSize;
  }
}

} // namespace torch::wave
