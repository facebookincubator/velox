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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "velox/dwio/nimble/common/Exceptions.h"

// Reversible rewrites of one SubIntSplit section within one block. How a
// transform undoes itself decides both the bytes stored and whether a single
// row can be read without reconstructing the section.
namespace facebook::nimble::subintsplit {

/// Identifies a transform on the wire. Dense, appended to, never renumbered:
/// these values are persisted, and a reader that cannot recognise one must
/// fail rather than decode.
enum class TransformId : uint8_t {
  /// No transform.
  None = 0,
  /// Stable sort of the section by another section's value.
  KeyDerived = 1,
  // 2 to 7 are retired ids, left as a gap rather than reused so
  // transformForRaw rejects them.
  /// Subtracts a fitted line, slope * row + base, from every whole value
  /// before the column is split. Recorded by the stream's row-frame header
  /// flag rather than in the per-section id array, so transformForRaw rejects
  /// it there.
  RowFrame = 8,
};

/// How a transform relates an original row to where its value ended up.
enum class PositionMapping : uint8_t {
  /// Values are rewritten where they stand.
  InPlace,
  /// Rows move, but where a row went is derivable from the key section
  /// (stored in original order) without reading the transformed data.
  Permuted,
};

/// Returns the name of a transform id, for logging and test failures.
std::string toString(TransformId id);

/// Bookkeeping for undoing a key-derived permutation. A block's key section
/// produces exactly one of these, and every section keyed on it may share it
/// rather than each rebuilding it (see TransformContext::keyRunIds).
struct KeyRunState {
  /// Run id of each row's key, in original row order.
  std::vector<uint32_t> runOfRow;
  /// The value each run id stands for.
  std::vector<uint64_t> runValues;
  /// Sorted rank of each run id among the distinct keys, ascending.
  std::vector<uint32_t> sortedRank;
  /// Prefix-sum run starts in sorted-rank order; runValues.size() + 1
  /// entries.
  std::vector<uint32_t> runStart;
};

/// Fills `out` with the KeyRunState for `keys`.
void buildKeyRunState(std::span<const uint64_t> keys, KeyRunState& out);

/// Scratch KeyDerivedTransform::invert reuses across calls instead of
/// allocating fresh buffers each time.
struct KeyDerivedScratch {
  /// Run bookkeeping built here when TransformContext did not supply one.
  KeyRunState local;
  /// Working copy of the run starts, consumed (incremented) during the merge
  /// that undoes the permutation.
  std::vector<uint32_t> cursor;
  /// Values in original row order, filled by the merge before being copied
  /// back into the section.
  std::vector<uint64_t> rows;
};

/// Everything a transform needs about the block beyond the section itself.
/// The keyRun* and keyOrder spans are optional caches: empty means the
/// transform must derive them from keySection itself.
struct TransformContext {
  /// The section this transform is keyed on, already decoded and in original
  /// row order. Empty for transforms that do not use one.
  std::span<const uint64_t> keySection;
  /// Bit width of the section being transformed, which sets the width of any
  /// codebook entry.
  int width{0};
  /// Column row of the first value, for transforms whose inverse depends on
  /// where a row sits rather than only on its value.
  uint64_t firstRow{0};
  /// Dense run ids for the key section, one per row; a transform needing the
  /// key's value order takes it from keyRunValues instead.
  std::span<const uint32_t> keyRunIds;
  /// The value each run id stands for. As many entries as there are runs.
  std::span<const uint64_t> keyRunValues;
  /// The permutation that stably sorts keySection, one row index per row.
  std::span<const uint32_t> keyOrder;
  /// Sorted rank of each run, indexed by run id, ascending among the
  /// distinct keys.
  std::span<const uint32_t> keyRunSortedRank;
  /// Prefix-sum run starts in sorted-rank order; runValues.size() + 1
  /// entries.
  std::span<const uint32_t> keyRunStart;
  /// Reusable scratch for KeyDerivedTransform::invert. Null means allocate
  /// locally; non-owning otherwise.
  KeyDerivedScratch* keyDerivedScratch = nullptr;
};

/// The permutation that stably sorts `key`, ties keeping their original row
/// order. Exposed so a caller sharing one key across several sections can
/// build it once and pass it back through TransformContext::keyOrder.
std::vector<uint32_t> buildKeyOrder(std::span<const uint64_t> key);

/// State a transform produces at encode and needs back at decode. The row
/// frame keeps its slope and base in the codebook; a key-derived permutation
/// stores nothing.
struct TransformState {
  /// Values the transform needs back at decode.
  std::vector<uint64_t> codebook;
};

/// Rewrites one section, reversibly.
class SectionTransform {
 public:
  virtual ~SectionTransform() = default;

  /// Identifies this transform on the wire.
  virtual TransformId id() const = 0;

  /// Rewrites `values`, one section, in place and fills `state` with whatever
  /// `invert` will need.
  virtual void apply(
      std::span<uint64_t> values,
      const TransformContext& context,
      TransformState& state) const = 0;

  /// Restores the original values in place; must exactly reproduce `apply`'s
  /// input.
  virtual void invert(
      std::span<uint64_t> values,
      const TransformContext& context,
      const TransformState& state) const = 0;

  /// Decides how a reader addresses a single row.
  virtual PositionMapping positionMapping() const = 0;

  /// Fills `positions[i]` with the offset the value of original row i was
  /// stored at. Defined only where positionMapping() is Permuted; throws
  /// otherwise.
  virtual void positionMap(
      const TransformContext& context,
      const TransformState& state,
      std::span<uint32_t> positions) const;

  /// Whether a single row can be read without reconstructing the section.
  virtual bool supportsPointAccess() const = 0;

  /// Whether this transform needs a key section. Selection uses this to know
  /// whether it must hold a section back unpermuted.
  virtual bool needsKeySection() const {
    return false;
  }

  /// Whether this transform applies to whole values before the column is
  /// split rather than to one section.
  virtual bool transformsWholeValue() const {
    return false;
  }
};

/// Returns the transform for `id`, or nullptr for `TransformId::None`.
/// Throws if `id` is not recognised, because decoding with the wrong transform
/// silently produces wrong values.
const SectionTransform* transformFor(TransformId id);

/// Returns the transform for a raw wire byte, validating it first.
inline const SectionTransform* transformForRaw(uint8_t rawId) {
  // File-format check: the byte came off the wire, so an unrecognised id
  // (including retired ones) must be reported as a file error, not fall
  // through to transformFor's internal check.
  NIMBLE_CHECK_FILE(
      rawId <= static_cast<uint8_t>(TransformId::KeyDerived),
      fmt::format("Unsupported SubIntSplit transform id: {}", rawId));
  return transformFor(static_cast<TransformId>(rawId));
}

} // namespace facebook::nimble::subintsplit

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
