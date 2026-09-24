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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>

#include "folly/container/F14Map.h"

#include "velox/dwio/nimble/common/RadixSort.h"
#include "velox/dwio/nimble/encodings/subintsplit/RowFrame.h"

namespace facebook::nimble::subintsplit {

// The decoder reproduces exactly this permutation by re-sorting the key
// section, so nothing needs storing. Must be a stable sort so the encoded
// bytes do not move under it.
std::vector<uint32_t> buildKeyOrder(std::span<const uint64_t> key) {
  const auto rowCount = static_cast<uint32_t>(key.size());
  const int keyBits = significantBits(key);
  std::vector<uint32_t> order;
  order.reserve(rowCount);

  // A key that fits in 32 bits travels packed with its row above it, so each
  // radix pass reads the key from the item being moved rather than a random
  // row of the key section.
  if (keyBits <= 32) {
    std::vector<uint64_t> packed;
    packed.reserve(rowCount);
    for (uint32_t row = 0; row < rowCount; ++row) {
      packed.push_back((key[row] << 32) | row);
    }
    RadixSort<uint64_t> sorter;
    sorter.sortStable(
        std::span<uint64_t>(packed),
        [](uint64_t item) { return item >> 32; },
        keyBits);
    for (const uint64_t item : packed) {
      order.push_back(static_cast<uint32_t>(item));
    }
    return order;
  }

  for (uint32_t row = 0; row < rowCount; ++row) {
    order.push_back(row);
  }
  RadixSort<uint32_t> sorter;
  sorter.sortStable(
      std::span<uint32_t>(order),
      [key](uint32_t row) { return key[row]; },
      keyBits);
  return order;
}

namespace {

// Every element of the scratch is written before it is read, so it is
// allocated without being cleared.
void gather(std::span<uint64_t> values, std::span<const uint32_t> order) {
  const size_t count = values.size();
  const auto scratch = std::make_unique_for_overwrite<uint64_t[]>(count);
  for (size_t i = 0; i < order.size(); ++i) {
    scratch[i] = values[order[i]];
  }
  std::copy(scratch.get(), scratch.get() + count, values.begin());
}

void scatter(std::span<uint64_t> values, const std::vector<uint32_t>& order) {
  std::vector<uint64_t> scratch(values.size());
  for (size_t i = 0; i < order.size(); ++i) {
    scratch[order[i]] = values[i];
  }
  std::copy(scratch.begin(), scratch.end(), values.begin());
}

} // namespace

void buildKeyRunState(std::span<const uint64_t> keys, KeyRunState& out) {
  const size_t count = keys.size();
  folly::F14FastMap<uint64_t, uint32_t> runOf;
  runOf.reserve(count / 8);
  out.runOfRow.resize(count);
  for (size_t i = 0; i < count; ++i) {
    const auto inserted =
        runOf.emplace(keys[i], static_cast<uint32_t>(runOf.size()));
    out.runOfRow[i] = inserted.first->second;
  }
  out.runValues.resize(runOf.size());
  for (const auto& entry : runOf) {
    out.runValues[entry.second] = entry.first;
  }

  const auto runs = static_cast<uint32_t>(out.runValues.size());
  std::vector<uint32_t> rank(runs);
  std::iota(rank.begin(), rank.end(), 0u);
  const auto& runValues = out.runValues;
  std::sort(rank.begin(), rank.end(), [&runValues](uint32_t a, uint32_t b) {
    return runValues[a] < runValues[b];
  });
  out.sortedRank.resize(runs);
  for (uint32_t i = 0; i < runs; ++i) {
    out.sortedRank[rank[i]] = i;
  }

  out.runStart.assign(runs + 1, 0);
  for (size_t i = 0; i < count; ++i) {
    ++out.runStart[out.sortedRank[out.runOfRow[i]] + 1];
  }
  std::partial_sum(
      out.runStart.begin(), out.runStart.end(), out.runStart.begin());
}

namespace {

class KeyDerivedTransform : public SectionTransform {
 public:
  TransformId id() const override {
    return TransformId::KeyDerived;
  }

  void apply(
      std::span<uint64_t> values,
      const TransformContext& context,
      TransformState& /*state*/) const override {
    NIMBLE_CHECK(
        context.keySection.size() == values.size(),
        "Key-derived transform needs a key section covering the same rows.");
    if (!context.keyOrder.empty()) {
      NIMBLE_CHECK_EQ(
          context.keyOrder.size(),
          values.size(),
          "Key-derived transform needs an order covering the same rows.");
      gather(values, context.keyOrder);
      return;
    }
    const auto order = buildKeyOrder(context.keySection);
    gather(values, order);
  }

  void invert(
      std::span<uint64_t> values,
      const TransformContext& context,
      const TransformState& /*state*/) const override {
    NIMBLE_CHECK(
        context.keySection.size() == values.size(),
        "Key-derived transform needs a key section covering the same rows.");
    const auto keys = context.keySection;
    const size_t count = values.size();
    if (count == 0) {
      return;
    }

    // The permutation this undoes is a stable sort by the key, so a row sits
    // at its key's run start plus its rank within that run. Undoing it is
    // therefore a k-way merge (walk rows in order, take the next value from
    // that row's run) rather than a scatter.
    //
    // `given` reflects whether the caller supplied the whole run-bookkeeping
    // bundle (dense run ids, run values, sorted run order, per-run starts)
    // through TransformContext, shared across every section keyed on the same
    // block's key. The fields are documented as independently optional, but
    // run ids alone do not imply the rest: indexing an empty keyRunStart
    // would read past the end.
    KeyDerivedScratch localScratch;
    KeyDerivedScratch& scratch = context.keyDerivedScratch != nullptr
        ? *context.keyDerivedScratch
        : localScratch;
    const bool given = !context.keyRunIds.empty() &&
        !context.keyRunSortedRank.empty() && !context.keyRunStart.empty();
    if (!given) {
      buildKeyRunState(keys, scratch.local);
    }
    const std::span<const uint32_t> runOfRow = given
        ? context.keyRunIds
        : std::span<const uint32_t>(scratch.local.runOfRow);
    const std::span<const uint64_t> runValues = given
        ? context.keyRunValues
        : std::span<const uint64_t>(scratch.local.runValues);
    const std::span<const uint32_t> position = given
        ? context.keyRunSortedRank
        : std::span<const uint32_t>(scratch.local.sortedRank);
    const std::span<const uint32_t> runStart = given
        ? context.keyRunStart
        : std::span<const uint32_t>(scratch.local.runStart);
    NIMBLE_CHECK_EQ(
        runOfRow.size(), count, "Key-derived needs one run id per row.");

    // Copies runStart because several sections may share it read-only, while
    // the merge below consumes cursor by incrementing it.
    scratch.cursor.assign(runStart.begin(), runStart.end());
    scratch.rows.resize(count);
    for (size_t i = 0; i < count; ++i) {
      scratch.rows[i] = values[scratch.cursor[position[runOfRow[i]]]++];
    }
    std::copy(scratch.rows.begin(), scratch.rows.end(), values.begin());
  }

  // Where a row went is its rank in the sort of the key section, and the key
  // section reaches the reader in original order, so that rank is derivable
  // without reading a single transformed value.
  PositionMapping positionMapping() const override {
    return PositionMapping::Permuted;
  }

  void positionMap(
      const TransformContext& context,
      const TransformState& /*state*/,
      std::span<uint32_t> positions) const override {
    const size_t count = positions.size();
    if (count == 0) {
      return;
    }

    // Where a row went is its run's start plus how many rows of that run came
    // before it, so this counts rather than sorts.
    std::vector<uint32_t> derivedIds;
    std::vector<uint64_t> derivedValues;
    const bool given = !context.keyRunIds.empty();
    if (!given) {
      NIMBLE_CHECK(
          context.keySection.size() == count,
          "Key-derived transform needs a key section covering the same rows.");
      folly::F14FastMap<uint64_t, uint32_t> runOf;
      runOf.reserve(count / 8);
      derivedIds.resize(count);
      for (size_t i = 0; i < count; ++i) {
        const auto inserted = runOf.emplace(
            context.keySection[i], static_cast<uint32_t>(runOf.size()));
        derivedIds[i] = inserted.first->second;
      }
      derivedValues.resize(runOf.size());
      for (const auto& entry : runOf) {
        derivedValues[entry.second] = entry.first;
      }
    }
    const std::span<const uint32_t> runOfRow =
        given ? context.keyRunIds : std::span<const uint32_t>(derivedIds);
    const std::span<const uint64_t> runValues =
        given ? context.keyRunValues : std::span<const uint64_t>(derivedValues);
    const size_t runs = runValues.size();

    // Runs are laid out in the key's order, which run ids do not carry.
    std::vector<uint32_t> rank(runs);
    std::iota(rank.begin(), rank.end(), 0u);
    std::sort(rank.begin(), rank.end(), [runValues](uint32_t a, uint32_t b) {
      return runValues[a] < runValues[b];
    });
    std::vector<uint32_t> position(runs);
    for (uint32_t i = 0; i < runs; ++i) {
      position[rank[i]] = i;
    }

    std::vector<uint32_t> cursor(runs + 1, 0);
    for (size_t i = 0; i < count; ++i) {
      ++cursor[position[runOfRow[i]] + 1];
    }
    std::partial_sum(cursor.begin(), cursor.end(), cursor.begin());
    for (size_t i = 0; i < count; ++i) {
      positions[i] = cursor[position[runOfRow[i]]]++;
    }
  }

  // A probe costs the position map, which is built once, and then a single
  // indirection.
  bool supportsPointAccess() const override {
    return true;
  }

  bool needsKeySection() const override {
    return true;
  }
};

// The row frame as a transform. A column whose values climb by a steady step
// per row spends its high bits on that climb; subtracting the line leaves
// residuals whose sections are narrow. Every row stays addressable, since the
// inverse is one multiply-add by the row index.
//
// apply() fits the line when the state carries none and leaves the values
// untouched when nothing fits, so an empty codebook means not applied.
class RowFrameTransform : public SectionTransform {
 public:
  TransformId id() const override {
    return TransformId::RowFrame;
  }

  void apply(
      std::span<uint64_t> values,
      const TransformContext& context,
      TransformState& state) const override {
    checkWidth(context.width);
    if (state.codebook.empty()) {
      const auto frame =
          context.width == 32 ? fitNarrowed(values) : fitWide(values);
      if (!frame.active()) {
        return;
      }
      state.codebook = {frame.slope, frame.base};
    }
    addLine(values, context, state, /*sign=*/~uint64_t{0});
  }

  void invert(
      std::span<uint64_t> values,
      const TransformContext& context,
      const TransformState& state) const override {
    checkWidth(context.width);
    if (!state.codebook.empty()) {
      addLine(values, context, state, /*sign=*/1);
    }
  }

  PositionMapping positionMapping() const override {
    return PositionMapping::InPlace;
  }

  bool supportsPointAccess() const override {
    return true;
  }

  bool transformsWholeValue() const override {
    return true;
  }

 private:
  static void checkWidth(int width) {
    NIMBLE_CHECK(
        width == 32 || width == 64,
        fmt::format("Row frame needs a 32- or 64-bit column, got {}", width));
  }

  static subintsplit::RowFrame fitWide(std::span<const uint64_t> values) {
    return subintsplit::fitRowFrame(values);
  }

  static subintsplit::RowFrame fitNarrowed(std::span<const uint64_t> values) {
    const std::vector<uint32_t> narrowed(values.begin(), values.end());
    return subintsplit::fitRowFrame(std::span<const uint32_t>(narrowed));
  }

  // Adds sign * (slope * row + base) to every value, modulo the width. sign
  // is 1 or all ones, so subtraction is the same loop.
  static void addLine(
      std::span<uint64_t> values,
      const TransformContext& context,
      const TransformState& state,
      uint64_t sign) {
    NIMBLE_CHECK_EQ(
        state.codebook.size(), size_t{2}, "Row frame stores slope, base.");
    const uint64_t mask =
        context.width >= 64 ? ~uint64_t{0} : (uint64_t{1} << context.width) - 1;
    const uint64_t slope = state.codebook[0] * sign;
    uint64_t predicted =
        (state.codebook[0] * context.firstRow + state.codebook[1]) * sign;
    for (auto& value : values) {
      value = (value + predicted) & mask;
      predicted += slope;
    }
  }
};

const KeyDerivedTransform kKeyDerived;
const RowFrameTransform kRowFrame;

} // namespace

void SectionTransform::positionMap(
    const TransformContext& /*context*/,
    const TransformState& /*state*/,
    std::span<uint32_t> /*positions*/) const {
  NIMBLE_UNREACHABLE(
      "Only a transform with a computable position mapping can say where a "
      "row went without reading the transformed data.");
}

std::string toString(TransformId id) {
  switch (id) {
    case TransformId::None:
      return "None";
    case TransformId::KeyDerived:
      return "KeyDerived";
    case TransformId::RowFrame:
      return "RowFrame";
  }
  return "Unknown";
}

const SectionTransform* transformFor(TransformId id) {
  switch (id) {
    case TransformId::None:
      return nullptr;
    case TransformId::KeyDerived:
      return &kKeyDerived;
    case TransformId::RowFrame:
      return &kRowFrame;
  }
  NIMBLE_UNREACHABLE(
      fmt::format(
          "Unsupported SubIntSplit transform id: {}", static_cast<int>(id)));
}

} // namespace facebook::nimble::subintsplit

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
